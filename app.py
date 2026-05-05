import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from rl_logic import CliffWalkingEnv, QLearningAgent, SarsaAgent

st.set_page_config(page_title="RL Demo: Cliff Walking", layout="wide")

# Custom CSS for modern look
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        color: white;
        border: none;
    }
    .metric-card {
        background-color: rgba(30, 41, 59, 0.7);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🪨 Cliff Walking: Q-Learning vs SARSA")
st.markdown("Comparing **Off-policy** (Q-Learning) and **On-policy** (SARSA) in a classic Gridworld.")

# Sidebar Parameters
st.sidebar.header("Hyperparameters")
alpha = st.sidebar.slider("Learning Rate (α)", 0.0, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.0, 1.0, 0.9)
epsilon = st.sidebar.slider("Exploration (ε)", 0.0, 0.5, 0.1)
episodes = st.sidebar.number_input("Total Episodes", 100, 2000, 500)

env = CliffWalkingEnv()

def run_training(agent_type):
    env = CliffWalkingEnv()
    if agent_type == "Q-Learning":
        agent = QLearningAgent(env, alpha, gamma, epsilon)
    else:
        agent = SarsaAgent(env, alpha, gamma, epsilon)
    
    rewards = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for ep in range(episodes):
        state = env.start_state
        total_reward = 0
        done = False
        steps = 0

        if agent_type == "Q-Learning":
            while not done and steps < 200:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(state, action)
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1
        else:
            action = agent.choose_action(state)
            while not done and steps < 200:
                next_state, reward, done = env.step(state, action)
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                total_reward += reward
                steps += 1
        
        rewards.append(total_reward)
        if ep % 50 == 0:
            progress_bar.progress((ep + 1) / episodes)
            status_text.text(f"Training {agent_type}: Episode {ep+1}/{episodes}")

    progress_bar.empty()
    status_text.empty()
    return rewards, agent

def plot_rewards(q_rewards, sarsa_rewards):
    fig = go.Figure()
    if q_rewards:
        fig.add_trace(go.Scatter(y=q_rewards, name="Q-Learning", line=dict(color='#10b981', width=1.5)))
    if sarsa_rewards:
        fig.add_trace(go.Scatter(y=sarsa_rewards, name="SARSA", line=dict(color='#f59e0b', width=1.5)))
    
    fig.update_layout(
        title="Accumulated Reward per Episode",
        xaxis_title="Episodes",
        yaxis_title="Total Reward",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def visualize_policy(agent, title):
    # Create a 4x12 grid to show the best action in each state
    grid = np.zeros((4, 12))
    # Actions: 0:↑, 1:↓, 2:←, 3:→
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    
    policy_grid = []
    for r in range(4):
        row = []
        for c in range(12):
            state = (r, c)
            if state == (3, 11):
                row.append("🏁")
            elif state in [(3, i) for i in range(1, 11)]:
                row.append("🔥")
            elif state == (3, 0):
                row.append("🚀")
            else:
                q_values = agent.q_table.get(state, [0, 0, 0, 0])
                best_action = np.argmax(q_values)
                row.append(arrows[best_action])
        policy_grid.append(row)
    
    df = pd.DataFrame(policy_grid)
    st.subheader(f"Final Policy: {title}")
    st.table(df)

# Main Interface
col1, col2 = st.columns([1, 1])

if 'q_rewards' not in st.session_state: st.session_state.q_rewards = []
if 'sarsa_rewards' not in st.session_state: st.session_state.sarsa_rewards = []
if 'q_agent' not in st.session_state: st.session_state.q_agent = None
if 'sarsa_agent' not in st.session_state: st.session_state.sarsa_agent = None

with col1:
    if st.button("Train Q-Learning"):
        st.session_state.q_rewards, st.session_state.q_agent = run_training("Q-Learning")

with col2:
    if st.button("Train SARSA"):
        st.session_state.sarsa_rewards, st.session_state.sarsa_agent = run_training("SARSA")

# Charts
if st.session_state.q_rewards or st.session_state.sarsa_rewards:
    st.plotly_chart(plot_rewards(st.session_state.q_rewards, st.session_state.sarsa_rewards), use_container_width=True)

# Policy Visualization
if st.session_state.q_agent or st.session_state.sarsa_agent:
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        if st.session_state.q_agent:
            visualize_policy(st.session_state.q_agent, "Q-Learning (Optimal/Risky)")
    with v_col2:
        if st.session_state.sarsa_agent:
            visualize_policy(st.session_state.sarsa_agent, "SARSA (Safe/Conservative)")

# Analysis Content
st.divider()
a1, a2 = st.columns(2)
with a1:
    st.info("### Q-Learning (Off-policy)\n"
            "Learns the value of the **optimal policy** regardless of exploration.\n\n"
            "- **Behavior**: Chooses the path right next to the cliff.\n"
            "- **Observation**: The reward curve is volatile because exploration occasionally drops it into the cliff.")
with a2:
    st.warning("### SARSA (On-policy)\n"
               "Learns the value of the **actual policy** including exploration.\n\n"
               "- **Behavior**: Learns that the cliff edge is 'dangerous' due to exploration probability.\n"
               "- **Observation**: The reward curve is smoother and stays higher once converged.")
