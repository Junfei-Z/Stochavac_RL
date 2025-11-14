# Simplified Prompt for GPT-4/Claude (Copy-Paste Ready)

Generate a professional TikZ figure showing a three-stage machine learning pipeline for vaccine allocation:

**Stage 1 (GREEN): Macro-Level ODE Expert**
- Input: Population groups (3 circles)
- Process: ODE solver with optimal control equations
- Output: 1000 expert trajectories
- Badge: "Tractable: 30D, deterministic, seconds"

**Stage 2 (BLUE): Diffusion Prior Learning**
- Input: Macro trajectories from Stage 1
- Process: (a) Lifting algorithm (group→individual network), (b) Diffusion model training (Transformer, T=1000 steps)
- Output: Probability distribution p_θ(a|s)
- Badge: "Learning: Transform, minutes"

**Stage 3 (ORANGE): Prior-Guided RL**
- Input: CTMP environment (network with 2000 nodes in states S,E,I,R,V,D)
- Process: PPO agent with loss = L_PPO + β·KL(π||p_θ)
- Key: Blue dashed arrow from Stage 2's p_θ into KL term
- Output: Optimized policy π*
- Badge: "Stochastic: 12000D, hours, 2× faster"

**Visual requirements:**
1. Horizontal left-to-right flow
2. Rounded rectangle boxes with drop shadows for each stage
3. Thick arrows between stages labeled with data types
4. Blue dashed arrow from Stage 2 to Stage 3 (prior regularization)
5. Color scheme: Stage 1=#2ECC71, Stage 2=#3498DB, Stage 3=#E67E22
6. Include small icons: ODE curves, network graphs, neural network layers
7. Add equations: "dS_g/dt=...", "w_i=0.6·k_i+0.4·r_i", "L=L_PPO+β·KL"
8. Width: 180mm, Height: 90mm

Generate complete LaTeX TikZ code ready to compile.
