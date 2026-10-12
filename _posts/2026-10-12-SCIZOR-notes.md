---
title: "SCIZOR: A Self-Supervised Approach to DataCuration for Large-Scale Imitation Learning"
date: 2026-10-12
tags:
    - data curation
    - paper-notes
link: https://arxiv.org/abs/2505.22626
---

问题：现有的data curation方法粒度太粗。

+ dataset级：对于整个数据集进行手动加权
+ trajectory级：默认一条轨迹要么好要么坏，但实际上一条轨迹中可能包含一次失败的抓取尝试，而随后是一次成功的尝试，不应该被完全丢弃

所以文章提出curation的最小单位应该是state-action pair，文章里也叫做transition。但是有效筛选这个pair也是有挑战性的，因为机器人演示中通常缺少足够的奖励标注，不好评估每个交互步骤的质量。文章提出了一种子监督的方法SCIZOR来解决，主要针对两种互补的低质数据：

+ suboptimal： 比如碰撞、抖动等错误或不良工作，可能会强化错误来教坏策略
+ redundant： 过渡重复某种模式，可能会盖过其他有信息量且多样的样本，稀释学习信号

![方法总览](/images/scizor-overview.png)

### Suboptimal Transition Removal via Progress Estimation

insight：对于正确的操作而言，任务的进度应该随时间而增长。虽然没有任务标签，但是我们可以通过两次观测的时间戳来判断进度。

如果间隔了三秒，但模型预测任务只推进了一秒，就说明这三秒内出现了失败、卡住等情况，属于次优交互。

这就需要我们来训练一个进度的预测器 $f: S_{i:i+T} \to T_p$， 输入是从时间步$i$到$i+T$的子轨迹$S_{i:i+T}$，输出是预测进度$T_p$。我们定义这段子轨迹的 suboptimality score 得分为$V_{i:i+T} = T - T_p$，这里$T$为实际的时间间隔

我们希望能对每一个 transition 定义一个 suboptimality score， 满足：

+ 有当前 transition 所属的子轨迹来预测任务进度
+ 考虑未来影响（随着时间递减），因为当前的一个次优操作也是可能影响到后来的
+ 考虑 transition 所属的整条轨迹的质量

我们假设子轨迹中每个 transition 的贡献相同，那么得分 $V_{i:i+T}$ 均匀分配到每个 transition 上，由此可计算出单个 transition 的得分：$\hat V_i = \sum_{t=i-T}^{i} \frac{V_{t:t+T}}{T}$。再考虑到未来影响，得到 $V_i = \sum_{t=i}^{T} \gamma^{t-i} \hat V_t$，其中 $\gamma$ 是衰减系数。最后我们将每个 transition 的得分与整个轨迹的平均分融合得到最终得分：
$$
V_i^{final} = \alpha \cdot V_i + (1 - \alpha) \cdot \frac{1}{N} \sum_{j=1}^{N} V_j
$$

最后引入一个阈值 $\epsilon_{s}$ 剔除次优样本即可。

最后说一下进度预测器的训练。我们将时间差 $\Delta t$ 分为五档，分别为[0, 0.5), [0.5, 1.0), [1.0, 2.0), [2.0, 5.0), [5.0, +$\infty$)。对于每一个训练样本，我们随机挑选一档，并在此区间内均匀采样一个 $\Delta t$，从轨迹里取出 $t$ 和 $t+\Delta t$时刻的两帧，并给一个标签（几档），模型的任务就是看这两帧并输出对应编号。