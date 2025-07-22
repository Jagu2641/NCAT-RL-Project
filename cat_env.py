import gym
from gym import spaces
import numpy as np
from simulated_student import SimulatedStudent
from load_question_bank import load_question_bank

class CATEnv(gym.Env):
    """
    Custom Gym environment for RL-based Computerized Adaptive Testing (CAT).
    Observation: [theta_est, mean_time, std_time, last_5_accuracies, last_5_response_times]
    Action: Index of question to select (0-499)
    """
    def __init__(self, student_profile='Average', question_bank_path='question_bank.csv'):
        super(CATEnv, self).__init__()
        self.question_bank = load_question_bank(question_bank_path)
        self.n_questions = len(self.question_bank)
        self.action_space = spaces.Discrete(self.n_questions)
        # Observation: theta_est, mean_time, std_time, last_5_accuracies, last_5_response_times
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0] + [0.0]*5 + [0.0]*5),
            high=np.array([1.0, 60.0, 60.0] + [1.0]*5 + [60.0]*5),
            dtype=np.float32
        )
        self.student_profile = student_profile
        self.max_steps = 20
        self.reset()

    def reset(self):
        self.student = SimulatedStudent(self.student_profile)
        self.true_theta = self.student.get_true_theta()
        self.asked_questions = set()
        self.correct_history = []
        self.response_times = []
        self.difficulties_correct = []
        self.theta_est = 0.5  # Start with neutral estimate
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        last_5_acc = self.correct_history[-5:] if len(self.correct_history) >= 5 else [0]*(5-len(self.correct_history)) + self.correct_history
        last_5_time = self.response_times[-5:] if len(self.response_times) >= 5 else [0]*(5-len(self.response_times)) + self.response_times
        mean_time = np.mean(self.response_times) if self.response_times else 0.0
        std_time = np.std(self.response_times) if self.response_times else 0.0
        obs = np.array([
            self.theta_est,
            mean_time,
            std_time,
            *last_5_acc,
            *last_5_time
        ], dtype=np.float32)
        return obs

    def step(self, action):
        action = int(action)  # Ensure action is always an int
        done = False
        info = {}
        if action in self.asked_questions:
            # Penalize repeated questions
            reward = -2.0
            correct = 0
            response_time = 5.0
        else:
            self.asked_questions.add(action)
            q_row = self.question_bank.iloc[action]
            difficulty = q_row['difficulty']
            correct, response_time = self.student.answer_question(difficulty)
            self.correct_history.append(correct)
            self.response_times.append(response_time)
            if correct:
                self.difficulties_correct.append(difficulty)
            # Update theta estimate: mean of difficulties of correct answers, else 0.5
            if self.difficulties_correct:
                self.theta_est = float(np.mean(self.difficulties_correct))
            else:
                self.theta_est = 0.5
            # Reward: -abs(theta_est - true_theta) - 0.1 * response_time
            reward = -abs(self.theta_est - self.true_theta) - 0.1 * response_time
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        obs = self._get_obs()
        return obs, reward, done, info 