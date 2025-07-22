import numpy as np

class SimulatedStudent:
    """
    Simulates a student with a true ability parameter (theta) and response behavior.
    Profiles:
        - Novice: 0.1–0.3
        - Emerging: 0.3–0.45
        - Average: 0.45–0.55
        - Above Average: 0.55–0.7
        - Expert: 0.7–1.0
    """
    PROFILE_RANGES = {
        'Novice': (0.1, 0.3),
        'Emerging': (0.3, 0.45),
        'Average': (0.45, 0.55),
        'Above Average': (0.55, 0.7),
        'Expert': (0.7, 1.0)
    }

    def __init__(self, profile: str):
        if profile not in self.PROFILE_RANGES:
            raise ValueError(f"Unknown profile: {profile}")
        self.profile = profile
        self.true_theta = np.random.uniform(*self.PROFILE_RANGES[profile])

    def answer_question(self, difficulty: float):
        """
        Simulate answering a question:
        - Correctness: Bernoulli(logistic(theta - difficulty))
        - Response time: Longer if question is harder than ability
        Returns: (correct: int, response_time: float)
        """
        # Logistic probability of correct answer
        prob_correct = 1 / (1 + np.exp(-(self.true_theta - difficulty) * 8))
        correct = np.random.rand() < prob_correct
        # Response time: base + penalty for difficulty above ability + noise
        base_time = 2.0  # seconds
        penalty = max(0, difficulty - self.true_theta) * 8  # scale penalty
        response_time = base_time + penalty + np.random.normal(0, 0.5)
        response_time = max(0.5, response_time)  # Clamp to minimum
        return int(correct), response_time

    def get_true_theta(self):
        return self.true_theta 