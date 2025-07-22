# Neural Computerized Adaptive Testing (NCAT) with Response Time Patterns Using Reinforcement Learning

## 🎯 Objective
This project implements a Reinforcement Learning (RL)-powered Computerized Adaptive Testing (CAT) system that adaptively selects math questions based on both:
- The correctness of a student’s answers
- Their response time per question

The system estimates and improves its understanding of a student’s ability level (θ) more accurately over time.

---

## 📁 Project Structure

```
.
├── simulated_student.py         # Simulates students with different ability profiles
├── load_question_bank.py        # Loads the question bank from CSV
├── cat_env.py                   # Custom Gym environment for CAT
├── train_agent.py               # Trains PPO agent on CATEnv
├── evaluate.py                  # Evaluates agent and saves plots
├── question_bank.csv            # Math question bank (provided)
├── plots/                       # Auto-generated evaluation plots
└── README.md                    # This file
```

---

## 🔧 Setup & Installation

1. **Clone the repository and navigate to the project folder.**
2. **Install dependencies:**

```bash
pip install numpy pandas matplotlib gym stable-baselines3
```

3. **Ensure `question_bank.csv` is present in the project directory.**

---

## 🚀 Usage

### 1. **Train the RL Agent**

This will train a PPO agent to select questions adaptively and save the model as `ppo_cat_agent.zip`.

```bash
python train_agent.py
```

### 2. **Evaluate the Trained Agent**

This will run the agent on unseen student profiles and generate plots in the `plots/` directory.

```bash
python evaluate.py
```

---

## 📘 File Descriptions

- **simulated_student.py**  
  Simulates students with varying ability profiles and response behaviors.

- **load_question_bank.py**  
  Loads the question bank from CSV into a pandas DataFrame.

- **cat_env.py**  
  Custom Gym environment for RL-based CAT, modeling the adaptive testing process.

- **train_agent.py**  
  Trains a PPO agent using `stable-baselines3` on the CAT environment.

- **evaluate.py**  
  Evaluates the trained agent on all student profiles, collects results, and saves plots.

- **plots/**  
  Contains auto-generated evaluation plots for each student profile.

- **question_bank.csv**  
  The math question bank with 500 questions and difficulty values.

---

## 📊 Output Plots

After running `evaluate.py`, the following plots will be saved in `plots/` for each student profile:
- Estimated vs True θ over steps
- Response time per step
- Question difficulty per step

---

## 🧩 Dependencies
- numpy
- pandas
- matplotlib
- gym
- stable-baselines3

---

## 🙋‍♂️ Notes
- The code is modular and beginner-friendly, with comments and docstrings throughout.
- You can adjust the number of training steps or evaluation episodes in `train_agent.py` and `evaluate.py`.
- For best results, use a GPU-enabled environment for faster RL training (optional).

---

## 📬 Questions?
Open an issue or contact the project maintainer for help. 