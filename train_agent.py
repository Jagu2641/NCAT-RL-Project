import gym
from stable_baselines3 import PPO
from cat_env import CATEnv


def main():
    """
    Train a PPO agent on the CATEnv environment and save the model.
    """
    # Register the custom environment with Gym (optional for SB3, but good practice)
    env = CATEnv(student_profile='Average', question_bank_path='question_bank.csv')

    # Wrap in a vectorized environment for SB3
    from stable_baselines3.common.env_util import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    # Instantiate the PPO agent
    model = PPO('MlpPolicy', vec_env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=20000)

    # Save the trained model
    model.save('ppo_cat_agent.zip')
    print('Model saved as ppo_cat_agent.zip')


if __name__ == '__main__':
    main() 