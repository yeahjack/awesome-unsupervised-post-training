# TT-VLA: On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning

> **加入 Survey 时间：** 2026-04-14

**论文：** arXiv 2601.06748v3
**作者：** Changyu Liu, Yiyang Liu, Taowen Wang, Qiao Zhuang, James Chenhao Liang, Wenhao Yang, Renjing Xu, Qifan Wang, Dongfang Liu, Cheng Han
**日期：** 2026-04-07

| 方法 | 载体 Carrier | 阶段 Regime | 层级 Level |
|------|----------------|-------------|------------|
| TT-VLA | Policy Opt. | test-time | Traj. |

---

## 审计更新

TT-VLA **不属于 strict UPT**。它的 policy update 发生在 test time 没错，但 reward 来自 **environment interaction + external progress estimator (VLAC)**，而不是完全 internal-only 的模型自举信号。

| When to Adapt | Streaming Continual Adaptation without immediate reset |
|---|---|
| 触发单位 Trigger Unit | arriving episode / environment interaction |
| 参数/状态持久性 Persistence | policy parameters accumulate across online episodes |
| 与推理关系 Inference Coupling | interleaved act-and-adapt during deployment |
| 输入可见性 Input Visibility | Online |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | No Immediate Reset |
| 证据状态 Evidence Status | note-explicit |
| 辅助分类 Timing Regime | Streaming Continual Adaptation |
| 可见数据范围 Visibility Scope | Streaming prefix only |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Streaming Continual Adaptation`；`Visibility Scope=Streaming prefix only`。
- **两轴编码：** `Input Visibility=Online`；`Update Persistence=Cumulative`；`Reset Boundary=No Immediate Reset`。

- **更新何时触发：** 更新在真实 deployment / interaction loop 中由新 episode 持续触发，而不是先收集完整 cohort 再统一优化。
- **服务当前样本还是后续样本：** 当前 episode 的反馈既服务当前 rollout，也会累积到后续 episodes 的策略中。
- **参数/状态是否累积：** 策略参数跨 online episodes 持续保留，默认不 reset。
- **reset 边界：** 因此它是典型的 deployment-time online adaptation。

## 1. UPT 归属理由

TT-VLA 更适合作为 **面向 VLA 的 environment-grounded test-time adaptation** 来讨论。

原因：

- **有真实的 inference-time policy adaptation**：在部署期间做 online PPO update。
- **reward 依赖环境与外部 progress model**：
  - task progress `p_t` 来自 progress estimator `Φ`；
  - 论文明确说用 **VLAC** 作为 progress regressor；
  - reward 是 progress difference。
- **不是 pure unlabeled prompt / model-internal generation**：
  - 需要机器人动作执行后的 observation；
  - 需要 environment state 变化；
  - 需要外部 progress predictor。

因此它与 survey 的 strict UPT 目标对象不同，更靠近 embodied / robotics setting 下的 online RL adaptation。

---

## 2. 解决的问题

作者想解决的核心问题是：

- 现有 VLA 多在训练时完成 SFT 或 training-time RL，部署时 policy 固定；
- 在真实机器人场景中，环境和分布不断变化，固定 policy 容易失效；
- 传统 sparse terminal reward 不适合 test-time adaptation，因为信号太晚。

---

## 3. 方法介绍

### 3.1 稠密进度奖励（Dense Progress-Based Reward）

- 令 `p_t = Φ(o_0:t+1, l)` 表示当前任务进度；
- per-step reward 定义为 `r_t = p_t - p_{t-1}`。

这个 reward 来自 progress estimator 对 observation history + instruction 的评估，而不是模型自身的输出一致性。

### 3.2 无 value 的 PPO（Value-free PPO）

- 由于 test-time episode 很短、无法稳定学习 value function，作者移除 PPO 里的 value loss；
- 进一步令 advantage 直接等于即时 reward，从而做 **value-free PPO**。

### 3.3 整体流程（Overall Pipeline）

- pretrained VLA 执行动作；
- 环境返回新 observation；
- progress estimator 计算 dense reward；
- policy 在线更新，再用于后续动作生成。

---

## 4. 数据集

### 仿真环境（Simulation）

基于 **ManiSkill 3** 的 pick-and-place 任务，并覆盖三类 generalization：

- **Execution**
- **Vision**
- **Semantics**

### 真实环境（Real-world）

- 9 个未见 real-world pick-and-place tasks

基础模型包括：

- **Nora**
- **OpenVLA**
- **OpenVLA-RL**
- **TraceVLA**

---

## 5. 评估指标与主要结果

主指标为 **success rate**。

模拟环境中，TT-VLA 对不同 backbone 都有增益。例如：

- **OpenVLA**
  - Execution 平均：**36.39% -> 39.83%**
- **OpenVLA-RL**
  - Execution 平均：**81.53% -> 84.17%**
- **TraceVLA**
  - Execution 平均：**26.94% -> 28.97%**

真实世界实验中，基于 **OpenVLA**：

- Execution：**40.42% -> 42.08%**
- Vision：**54.12% -> 57.08%**
- Semantics：**43.33% -> 46.25%**

作者还做了诊断比较：

- 与 **TLM**、**TTRL** 对比时，TT-VLA 更优；
- 论文据此认为 consensus-style pseudo-label 不适合 VLA，因为动作空间与环境反馈结构不同。

---

## 6. UPT Survey 定位

TT-VLA 的 survey 位置建议为：

- **strict UPT 外的 embodied / environment-grounded 相邻工作**

它的重要性不在于纳入主表，而在于提醒：

- 一旦进入 VLA / embodied setting，reward 很容易转向 environment progress、task success、critic prediction；
- 这与当前 survey 的“no external ground truth / verifier / tool / environment reward”口径已经明显不同。

因此它更适合作为 **boundary contrast case**，说明为什么 robotics/VLA test-time RL 不能直接与 text/MLLM strict UPT 混在一起。
