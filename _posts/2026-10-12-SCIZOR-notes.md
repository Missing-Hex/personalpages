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