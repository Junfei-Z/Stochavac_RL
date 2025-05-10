# 🧠 Stochavac-RL: Stochastic Vaccine Allocation with Reinforcement Learning

This repository provides a minimal and extensible framework for simulating vaccine allocation in stochastic epidemic environments using reinforcement learning. The model is based on individual-level contact networks with heterogeneous agent properties, where disease progression is governed by a Continuous-Time Markov Process (CTMP).

## 📌 Features

- Graph-based contact network with group and individual-level heterogeneity
- Stochastic infection, progression, and recovery dynamics
- Vaccination policies with constrained budgets and adaptive strategies
- Reinforcement learning optimization (REINFORCE, PPO)
- Modular codebase for experimentation and extension

## 📐 Problem Formulation

We formulate vaccine allocation as a **stochastic optimal control problem** over a CTMP defined on a contact graph. The objective is to learn a policy that minimizes cumulative infections or deaths by dynamically selecting individuals for vaccination under limited capacity.

## 📁 Project Structure

```plaintext
Stochavac-RL/
├── config.py              # Global configuration and hyperparameters
├── graph_model.py         # Generates heterogeneous contact network with groups and individual properties
├── epidemic_env.py        # Simulation environment with CTMP-based epidemic dynamics
├── policy_baseline.py     # Heuristic or ODE-based baseline vaccination policy
├── rl_agent.py            # Reinforcement learning logic (e.g., PPO, REINFORCE)
├── simulator.py           # Evaluation and rollout utilities
├── main.py                # Main script to run training, evaluation, or simulation
└── utils.py               # Helper functions: reward computation, logging, metrics, etc.
```




## 💡 Idea
- https://openreview.net/pdf?id=S5Yo6w3n3f - a kind of interesting combination with neural ode and refinforcement learning(for smooth controlloing ) , maybe can be used here
