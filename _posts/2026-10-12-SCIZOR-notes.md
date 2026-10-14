---
title: "SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning"
date: 2026-10-12
tags:
    - data curation
    - paper-notes
link: https://arxiv.org/abs/2505.22626
---

问题：现有的data curation方法粒度太粗。

+ dataset级：对整个数据集进行手动加权
+ trajectory级：默认一条轨迹要么好要么坏，但实际上一条轨迹中可能包含一次失败的抓取尝试，但随后是一次成功的尝试，不应该被完全丢弃

所以文章提出curation的最小单位应该是state-action pair，文中也叫做transition。但是有效筛选这个pair也是有挑战性的，因为机器人演示中通常缺少足够的奖励标注，不好评估每个交互步骤的质量。文章提出了一种自监督的方法SCIZOR来解决，主要针对两种互补的低质数据：

+ suboptimal： 比如碰撞、抖动等错误或不良动作，可能会强化错误行为，教坏策略
+ redundant： 过度重复某种模式，可能会盖过其他有信息量且多样的样本，稀释学习信号

![方法总览](/images/scizor-overview.png)

### Suboptimal Transition Removal via Progress Estimation

insight：对于正确的操作而言，任务的进度应该随时间而增长。虽然没有进度标签，但是我们可以通过两次观测的时间戳来判断进度。

如果间隔了三秒，但模型预测任务只推进了一秒，就说明这三秒内出现了失败、卡住等情况，属于次优交互。

这就需要我们来训练一个进度的预测器 $f: S_{i:i+T} \to T_p$， 输入是从时间步$i$到$i+T$的子轨迹$S_{i:i+T}$，输出是预测进度$T_p$。我们定义这段子轨迹的 suboptimality score 为 $V_{i:i+T} = T - T_p$，这里 $T$ 为实际的时间间隔。

我们希望能对每一个 transition 定义一个 suboptimality score， 满足：

+ 由当前 transition 所属的子轨迹来预测任务进度
+ 考虑未来影响（随着时间递减），因为当前的次优操作也可能影响到后续
+ 考虑 transition 所属的整条轨迹的质量

我们假设子轨迹中每个 transition 的贡献相同，那么得分 $V_{i:i+T}$ 均匀分配到每个 transition 上，由此可计算出单个 transition 的得分：$\hat V_i = \sum_{t=i-T}^{i} \frac{V_{t:t+T}}{T}$。再考虑到未来影响，得到 $V_i = \sum_{t=i}^{T} \gamma^{t-i} \hat V_t$，其中 $\gamma$ 是衰减系数。最后我们将每个 transition 的得分与整个轨迹的平均分融合得到最终得分：
$$
V_i^{final} = \alpha \cdot V_i + (1 - \alpha) \cdot \frac{1}{N} \sum_{j=1}^{N} V_j
$$

最后引入一个阈值 $\epsilon_{s}$ 剔除次优样本即可。

补充一下进度预测器的训练。我们将时间差 $\Delta t$ 分为五档，分别为[0, 0.5), [0.5, 1.0), [1.0, 2.0), [2.0, 5.0), [5.0, +$\infty$)。对于每一个训练样本，我们随机挑选一档，并在此区间内均匀采样一个 $\Delta t$，从轨迹里取出 $t$ 和 $t+\Delta t$时刻的两帧，并给一个标签（第几档），模型的任务就是看这两帧并输出对应编号。

### Similarity-Based State-Action Deduplication

insight：某些片段可能在视觉上是相似的，但是在任务意图或动作执行上不同。所以我们去重时必须同时考虑视觉状态和动作。

之前的研究主要集中于通过仅基于视觉特征的聚类来去除语义相似的数据，但这并不适合机器人模仿学习等序列决策任务（核心在于动作动力学，比如只看夹爪的某一帧图像，它可能是松手也可能是抓取，需结合动作信息）。文章将语义去重的思想拓展到了模仿学习领域，具体而言，我们首先将数据集划分为互不重叠的子轨迹$(S_{i:i+T}, a_{i:i+T})$。鉴于不同数据集之间记录频率的差异，在每个片段中均匀采样$N$张 RGB 图像。并使用 Cosmos 视频编码器（一种预训练模型，能从视频中编码时间和语义信息，提取成一维的视频特征向量$z_v$）。随后我们将末端执行器位姿的动作拼接到视觉向量上，形成 state-action 特征$z_{v+a}$。

不过这里$z_{v+a} = [z_v \mid \Delta \mathrm{action}]$，即将视频特征向量和动作向量（左右臂、夹爪）直接拼接。文章也承认了这种结合方式可能存在不足。

然后我们用 K-means 聚类将语义相似的 state-action chunks 分组。在每个聚类内，我们计算各个块之间的 cosine distances，并对每个块，将其相似度得分定义为同一聚类中与其他块的最小距离。再引入阈值$\epsilon_{d}$，将最大相似度超过阈值的块识别为重复项，在训练中用重复掩码过滤掉即可。（不过这样会把离得最近的两个样本都剔除掉，但应该保留一个样本？文章没有提到）。