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

## 🧪 Project Structure

