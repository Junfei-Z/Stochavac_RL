# Prompt for Generating Three-Stage Pipeline Figure

## Context
I need a high-quality, publication-ready figure for a machine learning paper showing a three-stage framework for vaccine allocation. The pipeline combines optimal control, diffusion models, and reinforcement learning.

## Overall Requirements
- **Style**: Professional academic paper figure (Nature/Science/NeurIPS quality)
- **Format**: TikZ code (LaTeX) or high-resolution vector graphic description
- **Layout**: Horizontal flow from left to right, three distinct stages
- **Color scheme**: Use professional academic colors (blues, greens, oranges) with clear visual hierarchy
- **Size**: Should fit in a two-column paper (width ~180mm)

## Three Stages to Visualize

### **Stage M1: Macro-Level ODE Expert Generation**

**Visual representation:**
- Input box (left): "Initial Conditions"
  - Icon: Small population groups (3 colored circles representing baseline/high-risk/high-contact groups)
  - Text: "Population groups (g=1,2,3)", "Contact matrix C", "N=2000"

- Process box (center):
  - Title: "M1: ODE Expert Generation"
  - Icon: Graph showing smooth ODE curves (S, E, I, R compartments declining/rising)
  - Equations snippet: "dS_g/dt = -λ_g S_g - u_g S_g"
  - Text: "Solve population-level optimal control"
  - Badge: "Tractable (30 states, deterministic)"

- Output box (right):
  - Title: "Macro Trajectories"
  - Icon: Multiple trajectory lines in 3D space
  - Text: "K=1000 expert demos", "{τ_macro^(k)}"
  - Visual: Small thumbnail showing 3 groups with different vaccination strategies (color-coded)

**Key visual elements:**
- Use GREEN color scheme for M1 (represents "tractable/solvable")
- Show continuous smooth curves
- Include small icons of optimal control symbols (∂, min, constraint brackets)

---

### **Stage M2: Diffusion Prior Learning**

**Visual representation:**

**Part 2a: Lifting (top sub-box)**
- Input: Macro trajectories from M1
- Process icon: Network graph transforming
  - Left side: 3 large circles (groups)
  - Right side: Dense network with ~100 small nodes
- Algorithm badge: "Degree-Risk Weighted Lifting"
- Formula snippet: "w_i = 0.6·k_i + 0.4·risk_i"
- Text: "Map group→individual actions"

**Part 2b: Diffusion Training (bottom sub-box)**
- Central icon: Diffusion process visualization
  - Left: Clean action vector (clear bars)
  - Middle: Noisy versions (T=1000 steps, gradient from clear to fuzzy)
  - Right: Transformer icon (multi-layer blocks)
- Process flow: "Forward diffusion → Reverse learning → p_θ(a|s)"
- Architecture badge: "Transformer (4 layers, d=128)"
- Loss function: "ℒ = 𝔼[||ε - ε_θ||²]"

**Output:**
- Title: "Diffusion Prior"
- Icon: Probabilistic distribution (Gaussian curve over action space)
- Text: "p_θ(a|s)", "Strategy distribution"
- Visual: Heatmap showing conditional probabilities

**Key visual elements:**
- Use BLUE color scheme for M2 (represents "learning/modeling")
- Show transformation from discrete (groups) to continuous (individuals)
- Include diffusion noise visualization (gradually blurred images)
- Arrows showing data flow: Macro → Lifting → Micro → Diffusion → Prior

---

### **Stage M3: Prior-Guided Reinforcement Learning**

**Visual representation:**

**Top section: Environment**
- Icon: Contact network with nodes in different states (S, E, I, R, V, D)
  - Color-code nodes by state: Gray=S, Yellow=E, Red=I, Blue=V, Green=R, Black=D
  - Show edges connecting nodes
- Badge: "CTMP Simulator"
- Text: "Individual-level stochastic dynamics", "N=2000 nodes, 6^N states"

**Middle section: Agent Architecture**
- Central box: "PPO Policy π_φ(a|s)"
  - Icon: Neural network diagram (3 layers)
  - Input: State features (group counts, network structure)
  - Output: Action probabilities (N-dimensional)

**Bottom section: Training Loop**
- Circular arrow showing RL loop:
  1. "Observe state s_t" →
  2. "Sample action a_t ~ π_φ" →
  3. "Execute in CTMP" →
  4. "Receive reward r_t" →
  5. "Update policy"

- Loss function box (highlighted):
  - "ℒ = ℒ_PPO + β(t)·KL(π_φ || p_θ) - λ·H(π_φ)"
  - Show β decay curve: β(t) = 1.0 → 0.6 (exponential decay)
  - Arrow from M2 diffusion prior p_θ into this KL term (in BLUE)

**Output:**
- Title: "Optimized Policy π*"
- Icon: Network with highlighted optimal vaccination targets (nodes marked with stars)
- Badge: "Network-aware, individual-level"
- Metrics: "Sample efficiency: 2× faster than RL-only"

**Key visual elements:**
- Use ORANGE/RED color scheme for M3 (represents "optimization/action")
- Show feedback loop clearly (circular arrows)
- Emphasize KL regularization with blue arrow from M2 to M3
- Include before/after comparison (random policy vs optimized)

---

## Connecting Elements

### **Arrows between stages:**
1. **M1 → M2**:
   - Thick arrow labeled "1000 macro trajectories"
   - Data flow icon (database symbol)

2. **M2 → M3**:
   - Two parallel arrows:
     - Top arrow: "Lifted expert data 𝒟" (for warmstart)
     - Bottom arrow: "Diffusion prior p_θ(a|s)" (for KL regularization) - make this BLUE and DASHED

3. **Within stages**: Use thinner arrows to show internal data flow

### **Visual hierarchy:**
- **Stage boxes**: Use rounded rectangles with drop shadows
- **Sub-processes**: Use lighter background colors, smaller boxes
- **Key equations**: Use yellow highlight boxes
- **Badges/tags**: Use small colored pills (like GitHub badges)

---

## Additional Visual Elements

### **Top annotation bar:**
- Show the progression of key properties:
  - M1: "Deterministic | 30D | Seconds" (green)
  - M2: "Learning | Transform | Minutes" (blue)
  - M3: "Stochastic | 12000D | Hours" (orange)

### **Bottom legend:**
- Color codes for node states (S, E, I, R, V, D)
- Line styles: Solid = data flow, Dashed = regularization
- Symbols: Circle = state, Rectangle = process, Rhombus = decision

### **Annotations:**
Add small callout boxes for key insights:
1. Near M1: "Leverages tractable ODE-based optimal control"
2. Near M2: "Learns strategy distribution, not single policy"
3. Near M3: "KL regularization improves sample efficiency by 2×"

---

## Technical Specifications

### **Dimensions:**
- Total width: 180mm (two-column figure)
- Height: ~90mm (maintain ~2:1 aspect ratio)
- Margins: 2mm between stage boxes

### **Fonts:**
- Stage titles: Bold, 11pt
- Labels: Regular, 9pt
- Equations: Computer Modern Math, 8pt

### **Colors (professional palette):**
- M1 (ODE): #2ECC71 (emerald green) for boxes, #27AE60 for accents
- M2 (Diffusion): #3498DB (bright blue) for boxes, #2980B9 for accents
- M3 (RL): #E67E22 (orange) for boxes, #D35400 for accents
- Backgrounds: Very light versions (#F0F0F0 with tint)
- Text: #2C3E50 (dark blue-gray)

### **TikZ-specific requests (if generating code):**
```latex
% Use these packages
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,shadows,decorations.pathreplacing}

% Define custom styles
\tikzstyle{stage} = [rectangle, rounded corners, draw, fill=..., drop shadow, minimum height=2.5cm]
\tikzstyle{data} = [cylinder, draw, fill=..., aspect=0.3]
\tikzstyle{process} = [rectangle, draw, fill=...]
```

---

## Example Structure (Pseudocode)

```
[Figure: Three-Stage Pipeline]

┌─────────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
│   M1: ODE Expert    │──────▶│   M2: Diffusion      │──────▶│   M3: Prior-Guided  │
│   Generation        │       │   Prior Learning     │       │   RL                │
│                     │       │                      │       │                     │
│  ┌──────────────┐   │       │  ┌────────────────┐  │       │  ┌──────────────┐   │
│  │ Initial      │   │       │  │ Lifting:       │  │       │  │ CTMP         │   │
│  │ Conditions   │   │       │  │ Macro→Micro    │  │       │  │ Environment  │   │
│  └──────────────┘   │       │  └────────────────┘  │       │  └──────────────┘   │
│         ↓           │       │         ↓            │       │         ↓           │
│  ┌──────────────┐   │       │  ┌────────────────┐  │       │  ┌──────────────┐   │
│  │ ODE Solver   │   │       │  │ Diffusion      │  │       │  │ PPO Agent    │   │
│  │ + Optimal    │   │       │  │ Transformer    │  │       │  │ π_φ(a|s)     │   │
│  │ Control      │   │       │  │ Training       │  │       │  └──────────────┘   │
│  └──────────────┘   │       │  └────────────────┘  │       │         ↑           │
│         ↓           │       │         ↓            │       │         │KL reg     │
│  ┌──────────────┐   │       │  ┌────────────────┐  │       │  ┌──────────────┐   │
│  │ 1000 Expert  │───┼──────▶│  │ p_θ(a|s)       │──┼───────┼─▶│ Prior (blue) │   │
│  │ Trajectories │   │       │  │ Distribution   │  │       │  └──────────────┘   │
│  └──────────────┘   │       │  └────────────────┘  │       │                     │
└─────────────────────┘       └──────────────────────┘       └─────────────────────┘
   GREEN (Tractable)             BLUE (Learning)              ORANGE (Optimization)
```

---

## Desired Output Format

Please provide:
1. **Complete TikZ code** that compiles to a publication-ready figure
2. **Alternative SVG description** if TikZ is too complex
3. **Detailed visual specification** for manual creation in tools like Inkscape/Illustrator

The figure should be self-contained and clearly convey:
- The three-stage nature of our approach
- Data flow from macro (ODE) to micro (CTMP)
- The role of diffusion as a bridge and prior
- The computational trade-offs at each stage

---

## Success Criteria

A successful figure should allow a reader to:
1. Understand the pipeline flow in 10 seconds
2. Identify the three key stages and their purposes
3. See how ODE experts guide RL through diffusion prior
4. Grasp the scale transition (population → individual)
5. Appreciate the methodological innovation (prior-guided RL)

The figure should be suitable for:
- Main paper figure (high visibility)
- Conference presentation slide
- Poster at ICML/NeurIPS/ICLR

**Style inspiration**:
- AlphaGo Nature paper (clear pipeline with domain knowledge + RL)
- DDPM paper (diffusion process visualization)
- Graph neural network papers (network + learning architecture)
