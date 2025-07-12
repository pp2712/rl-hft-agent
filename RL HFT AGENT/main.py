import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class OrderBookEnvironment(gym.Env):
    """
    Custom Gym environment for HFT trading with order book data
    """

    def __init__(self,
                 data: pd.DataFrame,
                 initial_cash: float = 100000,
                 max_inventory: int = 100,
                 tick_size: float = 0.01,
                 lot_size: int = 1):

        super(OrderBookEnvironment, self).__init__()

        # Market data
        self.data = data.copy()
        self.tick_size = tick_size
        self.lot_size = lot_size

        # Trading parameters
        self.initial_cash = initial_cash
        self.max_inventory = max_inventory

        # State variables
        self.current_step = 0
        self.cash = initial_cash
        self.inventory = 0
        self.last_trade_price = 0
        self.pnl = 0
        self.unrealized_pnl = 0
        self.total_trades = 0

        # Order book features (Level 2 data)
        self.lookback_window = 10
        self.price_history = deque(maxlen=self.lookback_window)
        self.spread_history = deque(maxlen=self.lookback_window)
        self.volume_history = deque(maxlen=self.lookback_window)

        # Action space: 0=hold, 1=buy, 2=sell, 3=cancel_all
        self.action_space = spaces.Discrete(4)

        # Observation space: [order_book_features, inventory, cash_ratio, pnl_ratio, time_features]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),  # Detailed feature space
            dtype=np.float32
        )

        # Transaction costs
        self.transaction_cost = 0.0001  # 1 basis point

        # Reset environment
        self.reset()

    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.cash = self.initial_cash
        self.inventory = 0
        self.last_trade_price = 0
        self.pnl = 0
        self.unrealized_pnl = 0
        self.total_trades = 0

        # Initialize history
        self.price_history.clear()
        self.spread_history.clear()
        self.volume_history.clear()

        # Fill initial history
        for i in range(self.lookback_window):
            row = self.data.iloc[i]
            mid_price = (row['bid_price_1'] + row['ask_price_1']) / 2
            spread = row['ask_price_1'] - row['bid_price_1']
            volume = row['bid_volume_1'] + row['ask_volume_1']

            self.price_history.append(mid_price)
            self.spread_history.append(spread)
            self.volume_history.append(volume)

        return self._get_observation(), {}

    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}

        # Get current market data
        current_data = self.data.iloc[self.current_step]

        # Calculate reward before action
        prev_total_value = self.cash + self.inventory * self._get_mid_price()

        # Execute action
        reward = self._execute_action(action, current_data)

        # Update state
        self.current_step += 1
        self._update_history(current_data)

        # Calculate new total value and PnL
        current_mid_price = self._get_mid_price()
        current_total_value = self.cash + self.inventory * current_mid_price
        self.unrealized_pnl = self.inventory * (
                    current_mid_price - self.last_trade_price) if self.last_trade_price > 0 else 0

        # Additional reward components
        inventory_penalty = -abs(self.inventory) * 0.01  # Penalize large inventory
        spread_reward = self._calculate_spread_reward(current_data)

        total_reward = reward + inventory_penalty + spread_reward

        # Check if done
        done = (self.current_step >= len(self.data) - 1 or
                abs(self.inventory) >= self.max_inventory or
                self.cash < 0)

        info = {
            'cash': self.cash,
            'inventory': self.inventory,
            'pnl': self.pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'total_value': current_total_value
        }

        return self._get_observation(), total_reward, done, False, info

    def _execute_action(self, action, market_data):
        """Execute the given action"""
        reward = 0

        bid_price = market_data['bid_price_1']
        ask_price = market_data['ask_price_1']
        mid_price = (bid_price + ask_price) / 2

        if action == 1:  # Buy
            if self.cash >= ask_price * self.lot_size and self.inventory < self.max_inventory:
                cost = ask_price * self.lot_size * (1 + self.transaction_cost)
                self.cash -= cost
                self.inventory += self.lot_size
                self.total_trades += 1

                # Reward for buying at good price (below mid)
                if ask_price < mid_price:
                    reward += (mid_price - ask_price) * self.lot_size

                self.last_trade_price = ask_price

        elif action == 2:  # Sell
            if self.inventory >= self.lot_size:
                revenue = bid_price * self.lot_size * (1 - self.transaction_cost)
                self.cash += revenue
                self.inventory -= self.lot_size
                self.total_trades += 1

                # Reward for selling at good price (above mid)
                if bid_price > mid_price:
                    reward += (bid_price - mid_price) * self.lot_size

                # Reward for profitable trade
                if self.last_trade_price > 0:
                    trade_pnl = (bid_price - self.last_trade_price) * self.lot_size
                    self.pnl += trade_pnl
                    reward += trade_pnl * 0.1  # Scale reward

                self.last_trade_price = bid_price

        # Action 0 (hold) and 3 (cancel - not implemented in this simple version) have no immediate effect

        return reward

    def _calculate_spread_reward(self, market_data):
        """Calculate reward based on spread conditions"""
        spread = market_data['ask_price_1'] - market_data['bid_price_1']
        # Reward for trading when spread is tight
        if spread < np.mean(self.spread_history):
            return 0.1
        return 0

    def _get_mid_price(self):
        """Get current mid price"""
        if self.current_step < len(self.data):
            current_data = self.data.iloc[self.current_step]
            return (current_data['bid_price_1'] + current_data['ask_price_1']) / 2
        return self.price_history[-1]

    def _update_history(self, current_data):
        """Update price and volume history"""
        mid_price = (current_data['bid_price_1'] + current_data['ask_price_1']) / 2
        spread = current_data['ask_price_1'] - current_data['bid_price_1']
        volume = current_data['bid_volume_1'] + current_data['ask_volume_1']

        self.price_history.append(mid_price)
        self.spread_history.append(spread)
        self.volume_history.append(volume)

    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.data):
            # Return last known state
            return self._create_observation_vector(self.data.iloc[-1])

        current_data = self.data.iloc[self.current_step]
        return self._create_observation_vector(current_data)

    def _create_observation_vector(self, market_data):
        """Create observation vector from market data and internal state"""

        # Order book features (Level 2)
        order_book_features = [
            market_data['bid_price_1'], market_data['ask_price_1'],
            market_data['bid_volume_1'], market_data['ask_volume_1'],
            market_data['bid_price_2'], market_data['ask_price_2'],
            market_data['bid_volume_2'], market_data['ask_volume_2'],
            market_data['bid_price_3'], market_data['ask_price_3'],
            market_data['bid_volume_3'], market_data['ask_volume_3']
        ]

        # Spread and mid price
        spread = market_data['ask_price_1'] - market_data['bid_price_1']
        mid_price = (market_data['bid_price_1'] + market_data['ask_price_1']) / 2

        # Technical indicators
        price_momentum = (list(self.price_history)[-1] - list(self.price_history)[0]) / list(self.price_history)[
            0] if len(self.price_history) > 0 else 0
        spread_avg = np.mean(self.spread_history) if len(self.spread_history) > 0 else spread
        volume_avg = np.mean(self.volume_history) if len(self.volume_history) > 0 else market_data['bid_volume_1'] + \
                                                                                       market_data['ask_volume_1']

        # Portfolio state
        cash_ratio = self.cash / self.initial_cash
        inventory_ratio = self.inventory / self.max_inventory
        pnl_ratio = self.pnl / self.initial_cash
        unrealized_pnl_ratio = self.unrealized_pnl / self.initial_cash

        # Time features
        time_progress = self.current_step / len(self.data)

        # Combine all features
        observation = np.array(order_book_features + [
            spread, mid_price, price_momentum, spread_avg, volume_avg,
            cash_ratio, inventory_ratio, pnl_ratio, unrealized_pnl_ratio,
            time_progress
        ], dtype=np.float32)

        return observation


class TradingCallback(BaseCallback):
    """Custom callback for monitoring training progress"""

    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Log episode information
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][-1]
            if 'total_value' in info:
                self.portfolio_values.append(info['total_value'])

        return True


def generate_synthetic_orderbook_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic order book data for testing"""
    np.random.seed(42)

    # Base price around 100
    base_price = 100.0

    # Generate price movements
    price_changes = np.random.normal(0, 0.1, n_samples)
    prices = base_price + np.cumsum(price_changes)

    data = []
    for i, price in enumerate(prices):
        # Generate bid-ask spread (0.01 to 0.05)
        spread = np.random.uniform(0.01, 0.05)

        # Level 1
        bid_price_1 = price - spread / 2
        ask_price_1 = price + spread / 2
        bid_volume_1 = np.random.randint(100, 1000)
        ask_volume_1 = np.random.randint(100, 1000)

        # Level 2
        bid_price_2 = bid_price_1 - np.random.uniform(0.01, 0.03)
        ask_price_2 = ask_price_1 + np.random.uniform(0.01, 0.03)
        bid_volume_2 = np.random.randint(50, 500)
        ask_volume_2 = np.random.randint(50, 500)

        # Level 3
        bid_price_3 = bid_price_2 - np.random.uniform(0.01, 0.03)
        ask_price_3 = ask_price_2 + np.random.uniform(0.01, 0.03)
        bid_volume_3 = np.random.randint(50, 500)
        ask_volume_3 = np.random.randint(50, 500)

        data.append({
            'timestamp': i,
            'bid_price_1': round(bid_price_1, 2),
            'ask_price_1': round(ask_price_1, 2),
            'bid_volume_1': bid_volume_1,
            'ask_volume_1': ask_volume_1,
            'bid_price_2': round(bid_price_2, 2),
            'ask_price_2': round(ask_price_2, 2),
            'bid_volume_2': bid_volume_2,
            'ask_volume_2': ask_volume_2,
            'bid_price_3': round(bid_price_3, 2),
            'ask_price_3': round(ask_price_3, 2),
            'bid_volume_3': bid_volume_3,
            'ask_volume_3': ask_volume_3,
        })

    return pd.DataFrame(data)


def train_hft_agent(data: pd.DataFrame,
                    total_timesteps: int = 100000,
                    learning_rate: float = 1e-4):
    """Train the HFT RL agent"""

    # Create environment
    env = OrderBookEnvironment(data)

    # Verify environment
    check_env(env)

    # Create DQN agent
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256, 128])
    )

    # Create callback
    callback = TradingCallback()

    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model, callback


def evaluate_agent(model, data: pd.DataFrame, episodes: int = 10):
    """Evaluate the trained agent"""

    env = OrderBookEnvironment(data)

    episode_rewards = []
    episode_info = []

    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_info.append(info)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Final PnL = {info['pnl']:.2f}, "
              f"Total Trades = {info['total_trades']}")

    return episode_rewards, episode_info


def plot_training_results(callback: TradingCallback):
    """Plot training results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Portfolio values
    if callback.portfolio_values:
        axes[0, 0].plot(callback.portfolio_values)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].grid(True)

    # Episode rewards
    if callback.episode_rewards:
        axes[0, 1].plot(callback.episode_rewards)
        axes[0, 1].set_title('Episode Rewards')
        axes[0, 1].set_xlabel('Episodes')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)

    # Episode lengths
    if callback.episode_lengths:
        axes[1, 0].plot(callback.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episodes')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)

    # Moving average of rewards
    if callback.episode_rewards and len(callback.episode_rewards) > 10:
        window = min(100, len(callback.episode_rewards) // 10)
        moving_avg = pd.Series(callback.episode_rewards).rolling(window=window).mean()
        axes[1, 1].plot(moving_avg)
        axes[1, 1].set_title(f'Moving Average Reward (window={window})')
        axes[1, 1].set_xlabel('Episodes')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """Main training and evaluation pipeline"""

    print("=== HFT Reinforcement Learning Agent ===")
    print("Generating synthetic order book data...")

    # Generate synthetic data
    train_data = generate_synthetic_orderbook_data(n_samples=8000)
    test_data = generate_synthetic_orderbook_data(n_samples=2000)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Train the agent
    print("\nTraining the agent...")
    model, callback = train_hft_agent(train_data, total_timesteps=50000)

    # Plot training results
    print("\nPlotting training results...")
    plot_training_results(callback)

    # Evaluate the agent
    print("\nEvaluating the agent...")
    episode_rewards, episode_info = evaluate_agent(model, test_data, episodes=5)

    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average PnL: {np.mean([info['pnl'] for info in episode_info]):.2f}")
    print(f"Average Total Trades: {np.mean([info['total_trades'] for info in episode_info]):.1f}")
    print(f"Average Final Portfolio Value: {np.mean([info['total_value'] for info in episode_info]):.2f}")

    # Save the model
    model.save("hft_dqn_model")
    print("\nModel saved as 'hft_dqn_model'")

    return model, callback


if __name__ == "__main__":
    # Run the complete pipeline
    model, callback = main()

    # Example of loading and using the saved model
    print("\n=== Loading and Testing Saved Model ===")
    loaded_model = DQN.load("hft_dqn_model")

    # Test with a small dataset
    test_data = generate_synthetic_orderbook_data(n_samples=1000)
    test_rewards, test_info = evaluate_agent(loaded_model, test_data, episodes=3)

    print(f"Loaded model test - Average reward: {np.mean(test_rewards):.2f}")