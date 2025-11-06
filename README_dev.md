# Project Plan: Prior-Guided RL + Diffusion for Vaccine Allocation (with `ProtectorPrevent` as expert)

> 交给 Claude Code 直接执行的实现说明。把本文黏到仓库根目录的 `README_dev.md`，让 Claude 按步骤实现即可。

---

## 0) 目标与里程碑

**目标**：在个体级（CTMP/接触网络）上，构建 "**专家(Ode) → 条件扩散(imitate) → RL(PPO, prior-guided)**" 的最小可运行管线，并在若干场景上跑出比专家更好的闭环策略。

**三个里程碑**
1. `M1` 复现专家轨迹（基于 `ProtectorPrevent` 的 3 组 ODE）→ 导出 macro 级 `(s,a)` 序列
2. `M2` 宏观→微观 **Lifting + CTMP 重放** → 得到 micro 级 `(s,a,s',r)` 数据集
3. `M3` 训练 **条件扩散** 模型与 **PPO(先验KL)** → 完成对比实验与可视化报告

---

## 1) 仓库结构与环境

```bash
vaccine-diffusion-rl/
├─ third_party/ProtectorPrevent/
├─ envs/
│  └─ vax_env.py
├─ expert/
│  └─ export_expert_trajectories.py
├─ lifting/
│  └─ lifting.py
├─ diffusion/
│  ├─ model.py
│  ├─ train.py
│  └─ sample.py
├─ rl/
│  ├─ ppo.py
│  └─ train_ppo.py
├─ data/
├─ conf/
├─ notebooks/
└─ README_dev.md
```

**依赖**：`python>=3.10, numpy, scipy, torch, gymnasium, networkx, einops, hydra-core, tqdm, matplotlib, wandb(optional)`

---

## 2) 接入专家代码（M1）

### 2.1 拉取子模块

```bash
git submodule add https://github.com/raarghal/ProtectorPrevent third_party/ProtectorPrevent
```

### 2.2 统一"专家仿真"接口

```python
def simulate_episode_macro(seed:int, scenario:dict) -> dict:
    '''
    Returns a trajectory dictionary:
      S,E,I,R,V: arrays with shape (T+1, G)
      U: (T, G) group-level vaccine allocation
      meta: metadata including R0, supply_per_day, etc.
    '''
```

### 2.3 导出专家轨迹

批量采样场景（R0, supply, infection rate），生成约 1000 条轨迹到 `data/macro_expert/`。

---

## 3) 宏观→微观 Lifting + CTMP 重放（M2）

### 3.1 环境定义

`envs/vax_env.py`: 基于 gymnasium 环境接口，定义 state, action, step()。

### 3.2 Lifting 模块

```python
def lift_macro_to_micro(Ug, state, graph, rule='risk', eps=1e-8):
    N = state['N']; groups = state['group_id']
    eligible = (state['vaccinated'] == 0).astype(float)
    deg = graph.degree_array()
    risk = state.get('risk', np.ones(N))
    w = 0.6*deg + 0.4*risk
    a = np.zeros(N)
    for g in [0,1,2]:
        idx = np.where(groups==g)[0]
        w_g = w[idx] * eligible[idx]
        s = w_g.sum()
        if s < eps or Ug[g] <= 0: continue
        a[idx] += Ug[g] * (w_g / (s+eps))
    a = np.minimum(a, eligible)
    scale = min(1.0, state['supply_today'] / (a.sum()+eps))
    return a * scale
```

### 3.3 微观轨迹重放

`expert/replay_to_micro.py`:
读取宏观轨迹 → 调用 `lift_macro_to_micro()` → 在 `VaxEnv` 里运行 → 保存 `(s,a,s',r)`。

---

## 4) 条件扩散模型（M3-Part1）

`diffusion/model.py`: Transformer/UNet-based ϵ-prediction 网络。
`diffusion/train.py`: MSE(ϵ,ϵ̂)+供应约束正则。
`diffusion/sample.py`: 条件采样并做可行性投影。

---

## 5) PPO with Prior-Guided KL（M3-Part2）

`rl/ppo.py`: 标准 PPO + KL 正则：

```python
loss_actor = loss_ppo_clip + beta * kl_divergence(pi_theta(a|s), pi_prior(a|s))
beta = beta0 * decay**update_step
```

扩散模型提供 `pi_prior`。

---

## 6) 可行性投影

```python
def project_feasible(a_t, state):
    a_t = np.clip(a_t, 0, None)
    a_t *= (state['vaccinated']==0)
    for g in [0,1,2]:
        idx = (state['group_id']==g)
        s = a_t[idx].sum() + 1e-8
        target = state['macro_quota'][g]
        a_t[idx] *= (target / s)
    total = a_t.sum() + 1e-8
    a_t *= (state['supply_today'] / total)
    return np.minimum(a_t, 1.0)
```

---

## 7) 评估与验收

1. **学习曲线**：三方法 (`RL-only`, `BC+RL`, `Diffusion+RL`)
2. **策略对比**：扩散采样 vs 专家
3. **鲁棒性箱线图**：R0、感染率、配额扰动

验收条件：
- M1：1000 宏观轨迹
- M2：500 微观轨迹，可行率100%
- M3：扩散 MAE<0.15，`Diffusion+RL` 收敛更快、泛化更好

---

## 8) Claude 执行顺序

1. 接入子模块，封装 `simulate_episode_macro()`
2. 实现 `VaxEnv` + `Lifting`，生成 micro 轨迹
3. 训练 `diffusion` 模型
4. 实现 `PPO + Prior-KL`，三组对比
5. 输出三图 + `README_results.md`

---

执行完此文件后，整个管线即实现 "专家 → 扩散模仿 → Prior-Guided RL"。
