import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from cat_env import CATEnv
import os

PROFILES = ['Novice', 'Emerging', 'Average', 'Above Average', 'Expert']
N_EPISODES = 10
PLOT_DIR = 'plots'


def evaluate_agent(model_path='ppo_cat_agent.zip', question_bank_path='question_bank.csv'):
    """
    Evaluate the trained PPO agent on unseen student profiles and save plots.
    """
    model = PPO.load(model_path)
    os.makedirs(PLOT_DIR, exist_ok=True)

    for profile in PROFILES:
        all_theta_est = []
        all_true_theta = []
        all_response_times = []
        all_difficulties = []
        for ep in range(N_EPISODES):
            env = CATEnv(student_profile=profile, question_bank_path=question_bank_path)
            obs = env.reset()
            theta_ests = []
            true_thetas = []
            response_times = []
            difficulties = []
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                theta_ests.append(env.theta_est)
                true_thetas.append(env.true_theta)
                if env.response_times:
                    response_times.append(env.response_times[-1])
                if env.correct_history:
                    idx = len(env.correct_history) - 1
                    q_idx = list(env.asked_questions)[idx]
                    diff = env.question_bank.iloc[q_idx]['difficulty']
                    difficulties.append(diff)
            all_theta_est.append(theta_ests)
            all_true_theta.append(true_thetas)
            all_response_times.append(response_times)
            all_difficulties.append(difficulties)
        # Plot theta estimation
        plt.figure(figsize=(8, 5))
        mean_theta_est = np.mean(all_theta_est, axis=0)
        mean_true_theta = np.mean(all_true_theta, axis=0)
        plt.plot(mean_theta_est, label='Estimated θ')
        plt.plot(mean_true_theta, label='True θ', linestyle='--')
        plt.xlabel('Step')
        plt.ylabel('Theta')
        plt.title(f'Theta Estimation - {profile}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'theta_est_{profile}.png'))
        plt.close()
        # Plot response times
        plt.figure(figsize=(8, 5))
        mean_resp_time = np.mean(all_response_times, axis=0)
        plt.plot(mean_resp_time, label='Mean Response Time')
        plt.xlabel('Step')
        plt.ylabel('Response Time (s)')
        plt.title(f'Response Time - {profile}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'response_time_{profile}.png'))
        plt.close()
        # Plot question difficulties
        plt.figure(figsize=(8, 5))
        mean_difficulty = np.mean(all_difficulties, axis=0)
        plt.plot(mean_difficulty, label='Mean Question Difficulty')
        plt.xlabel('Step')
        plt.ylabel('Difficulty')
        plt.title(f'Question Difficulty - {profile}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'difficulty_{profile}.png'))
        plt.close()
        print(f'Plots saved for profile: {profile}')

if __name__ == '__main__':
    evaluate_agent() 