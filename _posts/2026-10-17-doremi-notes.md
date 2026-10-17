---
title: "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining"
date: 2026-10-17
tags:
    - data curation
    - data mixing
    - paper-notes
link: https://arxiv.org/abs/2305.10429
---

问题：预训练数据的 domain 配比极大地影响模型效果，但这个配比到底怎么定？

+ The Pile 的做法：22 个 domain 的权重靠人工直觉拍板，作者自己也承认可能次优
+ PaLM / GLaM 的做法：靠下游任务调权重——要训上千个不同配比的模型，非常贵；而且等于"同一套题既当练习册又当考卷"，有过拟合到评测集的风险

所以文章要的是：**不依赖任何下游任务**，还能让模型在**所有** domain 上都不差的配比。

### Three Steps

![DoReMi 三步流程](/images/doremi-overview.png)

insight：先训一个很小的模型，但**不要那个模型**——要的是它试出来的那套配比数字。

+ **Step 1**：用参考权重 $\alpha_{\text{ref}}$ 训一个 reference model（文章里 280M），它的作用是提供每个域"正常的 loss 水平"
+ **Step 2**：再训一个同样大的 proxy model，用 Group DRO 每步动态调整域权重，最后返回权重
+ **Step 3**：用这套权重重新采样数据，训大模型（8B）

代价是训大模型 FLOPs 的 **8%**。

这里最容易套错的类比是**蒸馏**——但方向是反的。蒸馏是大模型教小模型，知识进参数，产物是模型；DoReMi 是小模型给大模型提建议，产物是一组采样概率，那个小模型用完即弃。一句话：**权重才是目的，模型是容器。**

论文里那句原话值得划出来：*"rather than using the robust LM, we take the domain weights produced by DRO training."*——DRO-LM 原本的产品是鲁棒模型，DoReMi 只要它的副产品。

### The Minimax Objective

$$
\min_\theta \max_{\alpha \in \Delta^k} \sum_{i=1}^{k} \alpha_i \cdot \left[\frac{1}{\sum_{x \in D_i} |x|} \sum_{x \in D_i} \left(\ell_\theta(x) - \ell_{\text{ref}}(x)\right)\right]
$$

逐层拆开看：

+ 最内层 $\ell_\theta(x) - \ell_{\text{ref}}(x)$ 是 **excess loss**（超额损失）
+ 中间按域内 token 总数平均（不是按样本数——长文本一个 token 一票，跟短文本公平）
+ 内层 $\max_\alpha$：在单纯形上取最坏分布
+ 外层 $\min_\theta$：补短板

**为什么不用裸 loss，这是整篇最漂亮的地方。** 不同 domain 的熵天生不同，GitHub 代码的 loss 天生就比 Wikipedia 高。只看裸 loss，模型会永远偏向"天生难的域"，但难不等于值得学。减掉参考模型的 loss，量出来的才是"这个域**还有多少可学的空间**"。

用三个样本说明：

| 样本 | $\ell_{\text{ref}}$ | $\ell_\theta$ | excess | 判断 |
|---|---|---|---|---|
| A | 1.0 | 3.0 | 2.0 | 参考学会了、代理还没 → **值得攻** |
| B | 5.0 | 5.5 | 0.5 | 两者都学不会（不可约损失高）→ 不值得 |
| C | 0.2 | 0.3 | 0.1 | 本来就容易 → 让出配额 |

只用 $\ell_\theta$ 会把 B 误当成最值得攻的域；只用 $-\ell_{\text{ref}}$ 又会偏好最容易的域。相减之后，只有 A 凸显出来。论文的消融也证实了：**这两个组件单独用都不行。**

### 为什么 $\max_\alpha$ 要写成加权求和

这里有个可以自己推一下的小结论：$\max_{\alpha \in \Delta^k} \sum_i \alpha_i v_i$ 其实**等价于** $\max_i v_i$——线性函数在单纯形上的最大值一定在顶点取到，也就是"全押最差的那个域"。那为什么还要写成加权求和？

+ 这是 DRO 的标准语言——"对分布求最坏"的自然写法，展开就是加权求和
+ 一般形式更普适：如果限制 $\alpha$ 不能离均匀太远，最坏分布就不一定在顶点
+ **算法需要 $\alpha$ 平滑移动**：每步硬跳 one-hot 会来回震荡，指数更新是 max 的平滑逼近（就像 softmax 是 argmax 的平滑版）

### Group DRO 怎么更新

每步的流程很短：

1. 从**均匀分布**采一个 minibatch（不管当前权重是多少——这样每个域都保证被观测到）
2. 算各域的超额损失，逐 token 算完按域内 token 总数平均，并**截断到非负**（Group DRO 要求损失非负）
3. 指数梯度上升：$\alpha_t' = \alpha_{t-1} \cdot \exp(\eta \lambda_t)$，其中 $\eta = 1$
4. 归一化 + 平滑：$\alpha_t = (1-c)\frac{\alpha_t'}{\sum_i \alpha_t'[i]} + c u$，其中 $c = 10^{-3}$，保证任何域都不会被精确归零
5. 用 $\alpha_t$ 加权的目标更新 proxy 模型
6. 最后返回整条轨迹的**平均** $\bar\alpha = \frac{1}{T}\sum_{t=1}^{T}\alpha_t$

注意第 1 步和第 5 步的分工：**采样是均匀的，加权只发生在损失上**。用一句话概括这个结构——**公平地看，偏心眼地学**。

### 为什么返回平均，而不是最后一步

这一点我一开始也没想明白。$\alpha_t$ 不是稳步走向终点的登山者，而是**围绕中心抖动的测量仪**——每步只依据当前 minibatch 的观测，噪声很大，而且没有"该停了"的机制。取最后一步等于让最后一小批数据的运气独裁。

每步是一次独立测量，平均让噪声正负相消。这也跟民调一样——报多期平均，不会只报最后一天。理论上，Nemirovski 等人的收敛保证本来就是给"轨迹平均"的，最后一步没有保证。

### Results

280M 的 reference/proxy，8B 的主模型：

+ The Pile：下游 one-shot 任务平均准确率提升 **6.5 个百分点**，达到基线精度只用 75k 步（基线要 200k），**快 2.6 倍**
+ GLaM 数据集：iterated DoReMi 第 2 轮就追平了**在评测集上调出来的 oracle 权重**——而 DoReMi 完全不知道下游任务
+ 优化权重的成本是主模型训练的 8%

配比的变化非常剧烈。The Pile 上：

| Domain | 原始权重 | DoReMi | 变化 |
|---|---|---|---|
| Pile-CC | 0.1121 | **0.6057** | +0.494 |
| ArXiv | 0.1052 | 0.0036 | −0.102 |
| PubMed Central | 0.1071 | 0.0046 | −0.103 |
| StackExchange | 0.0929 | 0.0153 | −0.078 |

**被砍得最狠的，恰好是人类觉得"信息密度最高"的学术、专业和代码域。** 而 Pile-CC 这个范围很广的网页文本独吞了 61%。这跟 ODM 后来学出的"趋向均匀"是相反的方向，别记混。

还有个反直觉的结果：**被降权的 domain，困惑度也一起改善了**。作者的解释是熵的两端都不需要太多样本——最低熵的域太容易，几个样本就学会；最高熵的域接近均匀先验（模型随机初始化时本来就爱输出均匀分布），也不需要多少样本去拟合。省下来的配额给中等熵的域，正迁移回所有域。

### 消融里两个有意思的发现

**proxy 不是越大越好。** 70M→150M→280M 提升，但 1B 的 proxy 反而变差。更极端的是，同规模下 **proxy 自己比主模型差**（1B proxy 甚至不如 1B 基线），但那套权重依然能让 1B 主模型受益。作者怀疑是 Group DRO 在大模型上不好使，因为 proxy 阶段是"损失加权"、主模型阶段是"重采样"，这两者并不等价。

**excess loss 缺一不可。** 只用 $\ell_\theta$（偏好代理模型最难的域）或只用 $-\ell_{\text{ref}}$（偏好参照模型最容易的域），都达不到 DoReMi 的效果。这个消融其实预言了后来的争论——ODM 用裸 loss 当奖励，被 ADO 批评"不考虑不可约损失"，而 DoReMi 的 excess loss 早就在处理这个问题了。

### 局限与后续

+ **跨 scale 迁移的机理是开放问题。** 论文 §6 自己写明："Understanding why the domain weights can be transferred across scales... are important questions for future work."
+ **域的定义很粗。** 文章按数据来源划分 domain，The Pile（22 域）的效果明显好于 GLaM（8 域）——粒度即空间。
+ **1B proxy 和 280M proxy 的落点不同**（Pile-CC vs OpenWebText 主导），说明权重空间有多个局部最优。

一个帮我理解迁移性的画面：**模型规模是锅，分词器和数据表示是食材。** 换锅（280M→8B），数据表示没变，"哪个域还没学够"的排序大概率仍然成立 → 跨锅迁移值得期待。换食材（换 tokenizer），数据被重新切分，可学性结构变了 → 跨食材迁移别指望。

这个画面正好预言了 ODM 后来的发现：**DoReMi 的权重跨 tokenizer 不迁移**。有意思的是，DoReMi 论文脚注 1 其实自己提了一句"domain weights 随 tokenizer 变化"——只是当时它是个脚注，后来被别人的复现实验做成了硬伤。

### 一点 robotics 的联想

**minimax 的"域"就是受保护名单。** 你定义哪些域，就等于声明哪些域不允许被牺牲——按任务分域，它拒绝牺牲任何任务；按具身分域，它拒绝让任何机器人型号变差。这是"安全侧"的目标函数，跟 ODM 那种"平均最优、允许战略性放弃"正好相对。

另外 DoReMi 的 minibatch 天然是**混合域**的，而 ODM 为了奖励可归因要求每个 minibatch 单域——后者在机器人这种小 batch 场景下会多一个坑。如果要迁移，DoReMi 的"均匀采样 + 损失加权"结构更稳。
