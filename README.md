# Cliff Walking: Q-Learning vs. SARSA Comparison

This project is a Reinforcement Learning (RL) demonstration that compares two classic algorithms—**Q-Learning** (Off-policy) and **SARSA** (On-policy)—in the "Cliff Walking" Gridworld environment.

Built with **Streamlit** for an interactive and modern visualization experience.

## 🔗 Live Demo
Check out the live interactive dashboard here:  
**[👉 RL Demo: Cliff Walking](https://drlhw2git-ev2jnj8gq25plzvdnatjbl.streamlit.app/)**

## 🌟 Features

- **Interactive Dashboard**: Adjust hyperparameters (Alpha, Gamma, Epsilon) and training episodes on the fly.
- **Real-time Training**: Observe the learning progress with live status updates.
- **Comparative Analytics**: Visual comparison of reward convergence using interactive Plotly charts.
- **Policy Visualization**: See the final learned path visualized with arrows (↑↓←→) to understand the behavioral differences.

## 🏗️ Environment: Cliff Walking

The environment is a 4x12 grid:
- **Start**: (3, 0)
- **Goal**: (3, 11)
- **Cliff**: The area between the start and goal at the bottom of the grid.
- **Rewards**: 
    - Each step: -1
    - Falling into the cliff: -100 (resets to start)
    - Reaching the goal: 0

## 🧠 Algorithm Behavior

### Q-Learning (Off-policy)
- **Nature**: Learns the optimal policy independent of the agent's current exploration.
- **Result**: Finds the shortest path right along the cliff edge.
- **Observation**: The reward curve is more volatile during training because exploration (ε) occasionally causes the agent to step into the cliff.

### SARSA (On-policy)
- **Nature**: Learns the value of the policy it is actually following, including the exploration steps.
- **Result**: Learns a safer, more conservative path (usually moving up to the second row) to avoid the risk of falling.
- **Observation**: Once converged, it achieves more stable and higher rewards during training compared to Q-Learning.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
Clone the repository and install the dependencies:
```bash
pip install streamlit plotly numpy pandas
```

### Running the App
```bash
python -m streamlit run app.py
```

## 📂 Project Structure
- `app.py`: Streamlit interface and visualization logic.
- `rl_logic.py`: Implementation of the Gridworld environment and RL agents.
- `.gitignore`: Configured to keep the repository clean from cache files.

---
Created for DRL HW2 assignment.
