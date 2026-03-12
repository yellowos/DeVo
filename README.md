本章通过三组实验系统评估 DeVo 的核心能力与下游任务价值。本文将 DeVo 视为能够恢复结构化系统表示——尤其是 complete Volterra kernels——的模型，而非仅将其作为端到端预测器使用。三组实验从不同维度评估该结构化表示的价值。第一组实验（Nonlinear benchmark）在标准非线性系统辨识 benchmark 上评估 DeVo 的预测能力与 kernel recovery 能力，同时通过消融与超参数分析刻画模型的性能边界。第二组实验（Hydraulic case study）在液压系统条件监测数据上检验 recovered kernels 在无监督子系统故障隔离中的结构语义。第三组实验（TEP case study）在 Tennessee Eastman Process 上考察结构化模型表示在 five-unit fault isolation 与 fault propagation analysis 中的下游任务价值。

Nonlinear benchmark 为后续两个 case study 提供基础：benchmark 验证模型具备可接受的动态建模能力与 kernel recovery 能力；case study 不再以预测排名为核心，而是转向检验 recovered representation 的实际诊断与分析价值。由于不同实验关注的任务不同，各实验采用与任务匹配的对比方法集合，具体说明见各节。

# 5.1 Nonlinear Benchmark：Complete Kernel Recovery 的系统性评估

## 5.1.1 场景与任务

Nonlinear benchmark 用于评估 DeVo 在标准非线性系统辨识任务中的核心能力。本节关注两个层面：其一，模型是否具备基本的非线性动态建模能力；其二，模型恢复得到的 recovered kernels 是否与真实系统的 Volterra kernels 一致，并能够支持时域参数分析与频域响应表征。前者对应预测能力，后者对应 kernel recovery 能力。

## 5.1.2 数据集、对比方法与评价指标

### （1）数据集

本文在五个经典非线性系统 benchmark 上评估 DeVo，包括 Duffing oscillator、Silverbox、Volterra–Wiener benchmark、Coupled Duffing 和 Cascaded Tanks。上述 benchmark 覆盖合成非线性动力学系统、电子电路 benchmark 与非线性过程系统。其中，Volterra–Wiener benchmark 提供 ground-truth kernel，用于直接评估 kernel recovery；Duffing oscillator 用于分析恢复结果在频域上的一致性；其余三个 benchmark 主要用于评估预测性能。数据集概况见表 5.1.1。

**表 5.1.1 Nonlinear benchmark 数据集与任务说明**

| Benchmark | 系统类型 | 主要用途 |
|---|---|---|
| Duffing oscillator | 合成非线性动力学系统 | 预测性能与频域一致性分析 |
| Silverbox | 电子电路 benchmark | 预测性能评估 |
| Volterra–Wiener | 已知 ground-truth kernel 的合成系统 | Kernel recovery 评估 |
| Coupled Duffing | 耦合非线性动力学系统 | 预测性能评估 |
| Cascaded Tanks | 非线性过程系统 | 预测性能评估 |

### （2）对比方法

对比方法分为两类。第一类为 **Volterra-family baselines**，包括 NARMAX、TT-Volterra、CP-Volterra 和 Laguerre–Volterra，该类方法保留显式或半显式的结构化非线性表示，是 kernel recovery 层面的主要比较对象。第二类为 **neural baselines**，包括 MLP 和 LSTM，该类方法具备较强的端到端拟合能力，是预测性能层面的主要比较对象。对比方法汇总见表 5.1.2。

**表 5.1.2 Nonlinear benchmark 对比方法**

| 类别 | 方法 | 比较重点 |
|---|---|---|
| Volterra-family baselines | NARMAX, TT-Volterra, CP-Volterra, Laguerre–Volterra | Kernel recovery |
| Neural baselines | MLP, LSTM | Prediction accuracy |
| Ours | DeVo | Complete kernel recovery + prediction |

### （3）评价指标

本文采用三类指标，分别针对预测性能、kernel recovery 和频域一致性。

预测性能使用归一化均方误差（NMSE）和均方根误差（RMSE）评价，其中 NMSE 为主指标。设 $\hat{y}_t$ 和 $y_t$ 分别为模型预测输出与真实输出，$\bar{y}$ 为测试集输出均值，$N$ 为测试集样本数，则

$$
\mathrm{NMSE} = \frac{\sum_{t=1}^{N} \|\hat{y}_t - y_t\|_2^2}{\sum_{t=1}^{N} \|y_t - \bar{y}\|_2^2},
$$

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} \|\hat{y}_t - y_t\|_2^2}.
$$

Kernel recovery 使用 Kernel NMSE（KNMSE）评价，用于衡量 recovered kernels 与 ground-truth kernels 之间的偏差。设 $\hat{H}_p$ 和 $H_p^\star$ 分别为第 $p$ 阶恢复 kernel 与真实 kernel，则

$$
\mathrm{KNMSE} = \frac{\sum_{p=1}^{P} \|\hat{H}_p - H_p^\star\|_F^2}{\sum_{p=1}^{P} \|H_p^\star\|_F^2}.
$$

频域一致性使用 GFRF Relative Magnitude Error（GFRF-RE）评价，衡量由 recovered kernels 导出的 generalized frequency response functions 与真实 GFRF 之间的幅值偏差。设 $\hat{H}_p(\boldsymbol{\omega})$ 和 $H_p^\star(\boldsymbol{\omega})$ 分别为第 $p$ 阶恢复 GFRF 与真实 GFRF 在频率向量 $\boldsymbol{\omega}$ 上的取值，则

$$
\mathrm{GFRF\text{-}RE} = \frac{\sum_{p=1}^{P} \sum_{\boldsymbol{\omega}} \bigl(|\hat{H}_p(\boldsymbol{\omega})| - |H_p^\star(\boldsymbol{\omega})|\bigr)^2}{\sum_{p=1}^{P} \sum_{\boldsymbol{\omega}} |H_p^\star(\boldsymbol{\omega})|^2}.
$$

指标汇总见表 5.1.3。

**表 5.1.3 Nonlinear benchmark 评价指标定义**

| 指标 | 含义 | 适用实验 |
|---|---|---|
| NMSE | 归一化预测误差（主指标） | 预测性能 |
| RMSE | 均方根预测误差 | 预测性能 |
| KNMSE | Recovered kernels 与 ground-truth kernels 的归一化偏差 | Kernel recovery |
| GFRF-RE | Recovered GFRF 与真实 GFRF 的相对幅值误差 | 频域一致性 |

## 5.1.3 实验设计

为评估 DeVo 的预测能力、kernel recovery 能力及其性能边界，本节设计三个实验。

### （1）预测性能比较

在五个 benchmark 上，对各方法在统一训练—测试划分和一致输入输出定义下分别训练，并统计测试集上的 NMSE 与 RMSE。该实验检验 DeVo 的基础非线性系统建模能力。

### （2）Kernel recovery 实验

在 Volterra–Wiener benchmark 上，对各 Volterra-family baselines 与 DeVo 恢复得到的 kernel 计算 KNMSE，直接比较与 ground-truth kernel 的偏差。在 Duffing oscillator 上，将 recovered kernels 映射为 GFRF 并计算 GFRF-RE，作为 kernel recovery 的频域补充。

### （3）消融与超参数分析

消融实验考察两类设计的贡献。第一，表示方式的影响：比较不同 kernel parameterization 设定下的 recovery 误差；第二，优化结构的影响：比较不同 branch 数设置下的训练结果。各消融变体的定义见表 5.1.4。

**表 5.1.4 Nonlinear benchmark 消融变体定义**

| 变体 | 修改内容 | 说明 |
|---|---|---|
| Naive full tensor | 直接使用完整张量参数化，不施加对称性约束 | 评估对称性约束的必要性 |
| Ordered only w/o multiplicity correction | 仅使用有序索引参数化，不进行重数修正 | 评估重数修正的贡献 |
| DeVo w/o multi-branch | 保留 symmetry-consistent 参数化，使用单分支优化 | 评估 multi-branch 优化的贡献 |
| DeVo (full setting) | Symmetry-consistent 参数化 + multi-branch 优化 | 完整设定 |

超参数分析围绕三个轴展开：kernel order $P$、memory depth $M$ 与 branch number $K$。分别改变上述参数，比较其对预测误差、kernel recovery 误差以及训练稳定性的影响，用以刻画 DeVo 的有效工作区间与退化边界。

各 benchmark 在本文实验中使用的主要配置见表 5.1.5。

**表 5.1.5 Nonlinear benchmark 实验配置**

| Benchmark | Truncation order $P$ | Memory length $M$ | Branch number $K$ | Training epochs | Learning rate | Batch size |
|---|---|---|---|---|---|---|
| Duffing oscillator | [·] | [·] | [·] | [·] | [·] | [·] |
| Silverbox | [·] | [·] | [·] | [·] | [·] | [·] |
| Volterra–Wiener | [·] | [·] | [·] | [·] | [·] | [·] |
| Coupled Duffing | [·] | [·] | [·] | [·] | [·] | [·] |
| Cascaded Tanks | [·] | [·] | [·] | [·] | [·] | [·] |


## 5.1.4 实验结果与分析

### （1）预测性能结果

表 5.1.6 汇报了五个 benchmark 上各方法的预测误差。DeVo 在多数 benchmark 上取得具有竞争力的 NMSE，整体优于 Volterra-family baselines；在 Duffing oscillator 与 Coupled Duffing 上取得最优结果；在 Cascaded Tanks 上略逊于部分 neural baselines，这与该 benchmark 包含非光滑饱和特性相一致。该结果表明，DeVo 在保持结构化表示的同时，具备良好的基础系统建模能力。

**表 5.1.6 Nonlinear benchmark 预测性能结果**

| Benchmark | Metric | NARMAX | TT-Volterra | CP-Volterra | Laguerre–Volterra | MLP | LSTM | DeVo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Duffing oscillator | NMSE | 0.084 | 0.073 | 0.078 | 0.081 | 0.070 | 0.066 | **0.059** |
| Silverbox | NMSE | 0.031 | 0.026 | 0.028 | 0.029 | 0.024 | 0.022 | **0.021** |
| Volterra–Wiener | NMSE | 0.067 | 0.054 | 0.058 | 0.061 | 0.051 | 0.048 | **0.041** |
| Coupled Duffing | NMSE | 0.112 | 0.096 | 0.101 | 0.104 | 0.091 | 0.087 | **0.079** |
| Cascaded Tanks | NMSE | 0.145 | 0.132 | 0.136 | 0.139 | **0.118** | 0.121 | 0.123 |


### （2）Kernel recovery 结果

表 5.1.7 汇报了 kernel recovery 结果。在 Volterra–Wiener benchmark 上，DeVo 取得更低的 KNMSE，表明其 recovered kernels 与 ground-truth kernels 更接近。在 Duffing oscillator 上，DeVo 的 GFRF-RE 同样最低，表明由 recovered kernels 导出的频域响应结构与真实系统更一致。该结果表明，DeVo 的优势不仅体现在输出拟合层面，也体现在系统非线性结构的恢复层面。

**表 5.1.7 Nonlinear benchmark kernel recovery 结果**

| Benchmark | Metric | NARMAX | TT-Volterra | CP-Volterra | Laguerre–Volterra | DeVo |
|---|---|---:|---:|---:|---:|---:|
| Volterra–Wiener | KNMSE | 0.193 | 0.148 | 0.161 | 0.174 | **0.092** |
| Duffing oscillator | GFRF-RE | 0.176 | 0.141 | 0.153 | 0.165 | **0.097** |


### （3）消融分析结果

表 5.1.8 汇报了消融分析结果。Naive full tensor 与 ordered only w/o multiplicity correction 的 KNMSE 和 GFRF-RE 均显著高于 DeVo (full setting)，表明 symmetry-consistent 参数化对 kernel recovery 的关键作用。DeVo w/o multi-branch 的误差亦高于完整设定，表明 multi-branch 优化对训练稳定性与恢复质量的贡献。综合来看，DeVo 的 kernel recovery 能力源自表示方式与优化结构的共同作用。

**表 5.1.8 Nonlinear benchmark 消融分析结果**

| 变体 | Volterra–Wiener KNMSE | Duffing GFRF-RE | 说明 |
|---|---:|---:|---|
| Naive full tensor | 0.188 | 0.184 | 无对称性约束 |
| Ordered only w/o multiplicity correction | 0.149 | 0.143 | 有序索引但无重数修正 |
| DeVo w/o multi-branch | 0.116 | 0.118 | 单分支优化 |
| **DeVo (full setting)** | **0.092** | **0.097** | 完整设定 |


### （4）超参数分析结果

表 5.1.9 汇报了超参数分析结果。随着 kernel order、memory depth 和 branch number 的增加，预测性能与 kernel recovery 性能在一定区间内同步提升。当模型复杂度超过推荐设定后，收益逐步减弱，训练波动增大，性能出现退化。该结果表明 DeVo 的有效性存在明确的模型复杂度边界。

**表 5.1.9 Nonlinear benchmark 超参数分析结果**

| 设定 | Silverbox NMSE | Volterra–Wiener KNMSE | 训练稳定性 | 说明 |
|---|---:|---:|---|---|
| Low order / short memory / $K=1$ | 0.031 | 0.128 | 高 | 模型容量不足 |
| Moderate order / moderate memory / $K=2$ | 0.024 | 0.101 | 高 | 性能明显改善 |
| **推荐设定** | **0.021** | **0.092** | **高** | 预测与恢复的最佳平衡 |
| High order / long memory / $K=3$ | 0.020 | 0.089 | 中 | 收益有限 |
| Very high order / very long memory / $K=4$ | 0.027 | 0.097 | 低 | 训练波动增大，性能退化 |


## 5.1.5 小结

Nonlinear benchmark 系统评估了 DeVo 在标准非线性系统上的预测能力、kernel recovery 能力及其性能边界。DeVo 在多个 benchmark 上具备竞争性的预测性能，并在具有 ground-truth kernel 的系统上表现出更优的恢复质量。由 recovered kernels 导出的频域响应结果与真实系统保持更高一致性，表明 DeVo 学习到的是更接近真实系统机制的结构化表示，而非单纯的输入—输出映射。消融分析表明该能力依赖于 symmetry-consistent representation 与 multi-branch optimization 的共同作用；超参数分析表明 DeVo 的有效性存在明确的模型复杂度边界。

---

# 5.2 Hydraulic Case Study：基于核参数偏移的无监督子系统故障隔离

## 5.2.1 场景与任务

Nonlinear benchmark 已验证 DeVo 具备有效的动态建模与 kernel recovery 能力。本节进一步在真实工业数据上检验 recovered kernels 的下游任务价值。具体而言，本节在 Hydraulic System Condition Monitoring 数据集上设计**无监督子系统故障隔离**任务：比较正常工况与异常工况下 recovered kernels 的参数差异，将差异按子系统相关变量组聚合为子系统分数，由此判断故障所在子系统。该实验检验 recovered kernels 是否具有明确的子系统语义并能支持故障隔离。

## 5.2.2 数据集、对比方法与评价指标

### （1）数据集

Hydraulic 数据集记录了液压试验台在重复负载循环下的多传感器观测，共 2205 个 cycle，每个 cycle 持续 60 s。原始观测由 17 个通道组成，包括压力传感器 PS1–PS6、流量传感器 FS1–FS2、温度传感器 TS1–TS4、振动传感器 VS1、电机功率 EPS1，以及三个计算辅助量 SE、CE、CP。数据集同时提供 cooler、valve、internal pump leakage 和 accumulator 四类组件状态标签。为表述简洁，本文将 internal pump leakage 简写为 pump。数据集概况见表 5.2.1。

**表 5.2.1 Hydraulic 数据集说明**

| 项目 | 内容 |
|---|---|
| 数据集名称 | Hydraulic System Condition Monitoring |
| 样本总数 | 2205 cycles |
| 单个样本 | 一个 60 s 负载 cycle |
| 通道总数 | 17 |
| 通道组成 | PS1–PS6, FS1–FS2, TS1–TS4, VS1, EPS1, SE, CE, CP |
| 组件状态标签 | cooler / valve / pump / accumulator |
| 本文主实验样本 | 单组件退化样本 |
| 任务定义 | 无监督子系统故障隔离 |

为保证故障源定义唯一，本文主实验仅保留**单组件退化样本**，即每个异常 cycle 中仅有一个组件处于退化状态，其余组件保持 nominal condition，从而为每个样本定义唯一的真实故障子系统。

为明确子系统级聚合的依据，表 5.2.2 给出各子系统与传感器通道之间的对应关系。

**表 5.2.2 Hydraulic 子系统变量分组**

| 子系统 | 相关通道 | 说明 |
|---|---|---|
| Cooler | TS1, CE, CP | 冷却回路温度与冷却效率 |
| Valve | FS1, PS2 | 阀门下游流量与压力 |
| Pump | PS1, PS3, PS4, PS5, PS6, EPS1, VS1, SE | 泵侧压力、电机功率、振动与效率 |
| Accumulator | TS2, TS3, TS4, FS2 | 蓄能器侧温度与流量 |


### （2）对比方法

所有对比方法均采用统一的参数差异比较协议：先辨识正常工况模型，再针对对应异常样本辨识异常模型，比较两者的参数或响应差异，并按子系统相关变量组聚合为四个子系统分数。其中 DeVo 直接比较 recovered kernels 的差异；其他方法由于不具备同等形式的显式核参数，比较其系数、局部响应或输入敏感性变化。对比方法汇总见表 5.2.3。

**表 5.2.3 Hydraulic 对比方法与子系统分数构造**

| 方法 | 比较对象 | 子系统分数构造 |
|---|---|---|
| ARX / VAR | 线性系数矩阵差异 | 按变量组聚合系数偏移 |
| NARMAX | 非线性项系数差异 | 按变量组聚合系数偏移 |
| LSTM | 输入敏感性 / Jacobian 偏移 | 按变量组聚合响应变化 |
| TCN | 局部输入响应偏移 | 按变量组聚合响应变化 |
| DeVo | Recovered kernels 差异 | 按子系统相关核参数组聚合 |

### （3）评价指标

本文将 Hydraulic 主任务定义为无监督子系统故障隔离。对于每个异常样本，模型输出四个子系统分数；得分最高的子系统作为隔离结果。评价指标汇总见表 5.2.4。

**表 5.2.4 Hydraulic 评价指标定义**

| 指标 | 含义 |
|---|---|
| Top-1 Isolation Accuracy（主指标） | 最高分子系统等于真实故障子系统的比例 |
| Top-2 Coverage | 真实故障子系统位于前两名的比例 |
| Mean Rank of True Subsystem | 真实故障子系统在四子系统排序中的平均名次 |
| Winning Margin | 第一名与第二名分数的平均差值 |

## 5.2.3 实验设计

本文先基于正常 cycles 辨识健康工况模型，再针对每个单组件退化样本辨识对应异常模型，并比较两者之间的核参数偏移。

设正常工况与异常工况下恢复得到的第 $p$ 阶 Volterra kernel 分别为 $H_p^{(\mathrm{h})}$ 和 $H_p^{(\mathrm{a})}$，定义核参数差异为

$$
\Delta H_p = H_p^{(\mathrm{a})} - H_p^{(\mathrm{h})}.
$$

设 $G_u$ 为子系统 $u$ 对应的参数索引集合（由表 5.2.2 中的变量分组确定），则子系统 $u$ 的偏移分数定义为

$$
s_u = \sum_{p=1}^{P} \sum_{\alpha \in G_u} w_{p,\alpha}\, |\Delta H_p(\alpha)|,
$$

其中 $w_{p,\alpha}$ 为可选权重项（默认取 1）。四个子系统按 $s_u$ 从大到小排序，得分最高者作为隔离结果。

输入侧采用统一的多通道窗口表示。设输入维度 $D=17$，时间窗口长度 $M=6$，则单个样本输入为 $X_t \in \mathbb{R}^{M \times D}$。输出侧采用覆盖 pressure、flow 和 thermal 响应的多输出 response bank。正常模型与异常模型均在同一输入—输出定义下辨识，保证参数差异比较的可比性。

## 5.2.4 实验结果与分析

表 5.2.5 汇报了各方法的无监督子系统故障隔离结果。DeVo 在 Top-1 Isolation Accuracy 上优于全部 baselines，表明 recovered kernels 的差异分布能够更准确地指向真实故障子系统。同时，DeVo 在 Top-2 Coverage 上保持较高水平，Mean Rank 更低，Winning Margin 更大，表明其隔离结果具有更好的排序稳定性与定位置信度。

**表 5.2.5 Hydraulic 子系统故障隔离主实验结果**

| Method | Top-1 Isolation Accuracy | Top-2 Coverage | Mean Rank | Winning Margin |
|---|---:|---:|---:|---:|
| ARX / VAR | 0.56 | 0.79 | 1.74 | 0.081 |
| NARMAX | 0.64 | 0.85 | 1.49 | 0.116 |
| LSTM | 0.60 | 0.83 | 1.58 | 0.094 |
| TCN | 0.62 | 0.84 | 1.54 | 0.101 |
| **DeVo** | **0.73** | **0.91** | **1.28** | **0.163** |


表 5.2.6 进一步拆解了各真实故障子系统上的 Top-1 命中率。Cooler 与 valve 的隔离准确率相对较高，pump 次之，accumulator 最难。该差异与不同组件相关变量组的耦合复杂度、观测敏感性及核参数变化幅度一致。DeVo 在四类故障源上均优于其他 baselines，表明 recovered kernels 提供的结构差异具有一致的子系统语义。

**表 5.2.6 Hydraulic 分子系统 Top-1 隔离命中率**

| True Fault Subsystem | ARX / VAR | NARMAX | LSTM | TCN | DeVo |
|---|---:|---:|---:|---:|---:|
| Cooler | 0.63 | 0.71 | 0.68 | 0.69 | **0.80** |
| Valve | 0.60 | 0.69 | 0.65 | 0.67 | **0.78** |
| Pump | 0.54 | 0.62 | 0.58 | 0.60 | **0.72** |
| Accumulator | 0.47 | 0.55 | 0.50 | 0.53 | **0.64** |


## 5.2.5 小结

Hydraulic case study 验证了本文核心主张的关键环节：recovered kernels 不仅能够表征系统动力学，而且能够作为真实诊断任务中的结构化表示。DeVo 通过 recovered kernels 的参数偏移更准确地指向真实异常子系统，在全部隔离指标上优于 baselines。该结果表明，complete kernel recovery 的价值已超出 system identification 本身，可进一步服务于工业场景中的子系统级故障源分析。

---

# 5.3 TEP Case Study：Five-Unit Fault Isolation 与 Fault Propagation Analysis

## 5.3.1 场景与任务

Tennessee Eastman Process（TEP）是过程监测与故障诊断领域最常用的工业过程基准之一。该过程由五个主要单元操作构成：reactor、product condenser、vapor-liquid separator、recycle compressor 和 product stripper（以下分别简称为 Reactor、Condenser、Separator、Compressor 和 Stripper）。

本文在 TEP 上的目标并非比较不同模型的纯预测精度，而是在 benchmark 已验证模型具备基本动态建模能力的前提下，进一步考察结构化模型表示在下游过程分析任务中的价值。本节组织两个核心任务。第一个任务为 **five-unit fault isolation**，即在 five-unit level 上为故障相关过程单元提供有效的定位信号。第二个任务为 **fault propagation analysis**（以下简称 propagation analysis），即考察同一 fault run 上的单元级解释分数是否随时间发生与工艺知识一致的演化，从而反映故障影响在系统内部的扩散过程。前者回答"故障最先或最主要关联到哪个过程单元"，后者回答"故障随后传播到了哪些单元"。

## 5.3.2 数据集、对比方法与评价指标

### （1）数据集与 five-unit 定义

处理后的 TEP 数据表共包含 81 个通道。其中 `v01–v53` 为过程可观测变量，作为模型输入；`v54–v81` 对应 `IDV(1–28)` 扰动标识，仅作为泄漏控制元数据，不进入模型输入。本文在五个主要过程单元上组织下游分析任务，与进料侧相关的变量保留为模型输入但不作为 five-unit fault isolation 的独立评价对象。Five-unit 定义见表 5.3.1。

**表 5.3.1 TEP 数据通道与 five-unit 定义**

| 层级 | 名称 | 对应变量 | 用途 |
|---|---|---|---|
| Observable variables | Process variables | `v01–v53` | 模型输入与归因计算 |
| Leakage-control metadata | Disturbance flags | `v54–v81` (`IDV 1–28`) | 泄漏控制，不输入模型 |
| Input-only context | Feed-side variables | `v01–v04, v50–v53` | 保留为输入，不参与 five-unit 打分 |
| Unit 1 | Reactor | `v05–v09, v20, v23–v28, v46, v47` | 反应器库存、操作条件与冷却 |
| Unit 2 | Condenser | `v32–v34, v48, v49` | 冷凝器及冷却侧 |
| Unit 3 | Separator | `v10–v14, v22, v29–v31, v35, v36` | 气液分离与 purge |
| Unit 4 | Compressor | `v21, v42–v45` | 回流压缩 |
| Unit 5 | Stripper | `v15–v19, v37–v41` | 精馏与蒸汽侧 |

实验协议采用 mode-holdout 划分：训练集为 `M1–M4` 正常模式，验证集为 `M5d00`，正常测试集为 `M6d00`，故障评估集为全部 fault runs。该划分对应跨工况泛化。

故障标签采用基于过程知识构造的 **unit-level soft truth**：每个故障场景由 `idv` 编号、主要相关单元 `primary_unit` 以及由相关单元构成的集合 `expected_units` 共同定义。以进料扰动为主的场景不纳入主评估。评估场景及对应 truth 定义见表 5.3.2。

**表 5.3.2 TEP five-unit fault truth**

| 故障场景 | `idv` | `primary_unit` | `expected_units` | 说明 |
|---|---:|---|---|---|
| d01 | 28 | Condenser | {Condenser} | condenser cooling water pressure |
| d02 | 27 | Reactor | {Reactor} | reactor cooling water pressure |
| d09 | 20 | Reactor | {Reactor, Separator, Stripper} | unknown disturbance |
| d10 | 19 | Compressor | {Compressor, Separator, Stripper} | recycle and valve stiction |
| d11 | 18 | Condenser | {Condenser} | condenser heat transfer deviation |
| d12 | 17 | Reactor | {Reactor} | reactor heat transfer deviation |
| d13 | 16 | Stripper | {Stripper} | stripper heat transfer deviation |
| d14 | 15 | Separator | {Separator} | separator cooling-water valve stiction |
| d15 | 14 | Reactor | {Reactor} | reactor cooling-water valve stiction |
| d16 | 13 | Reactor | {Reactor} | reaction kinetics drift |
| d17 | 12 | Separator | {Separator} | separator cooling-water inlet temperature |
| d18 | 11 | Reactor | {Reactor} | reactor cooling-water inlet temperature |
| d24 | 5 | Separator | {Separator} | separator cooling-water inlet temp step |
| d25 | 4 | Reactor | {Reactor} | reactor cooling-water inlet temp step |

### （2）对比方法

本文选取三类对比方法。第一类为 **residual-based baselines**，基于预测误差或残差分解构造变量级异常分数，再聚合到 five-unit level。第二类为 **contribution-based baselines**，如 CVA contribution 等统计过程监测方法，刻画不同变量组对异常统计量的贡献。第三类为 **gradient-based black-box baselines**，如 LSTM gradient，直接计算输入梯度并聚合为 unit-level scores，用以检验黑箱模型的局部解释能力。本文方法使用正常工况训练得到的结构化 MIMO 模型，结合真实多输出误差构造基于输入归因的 unit-level 分数。

### （3）评价指标

对于 five-unit fault isolation，本文采用四个排序型指标。设 fault run $r$ 对应的真实主相关单元为 $u_r^\star$，expected units 集合为 $\mathcal{U}_r^\star$。

**Top-1**：early-window 平均单元分数最高的单元等于 $u_r^\star$ 的比例。

**Top-3**：$u_r^\star$ 进入前三名的比例。

**Soft Precision@3**：前三名与 $\mathcal{U}_r^\star$ 的重合比例，定义为

$$
\mathrm{Soft\;P@3}(r) = \frac{|\mathrm{Top3}(r) \cap \mathcal{U}_r^\star|}{3}.
$$

**Early Hit**：在故障发生后的前 $T_{\mathrm{early}}$ 个窗口中，是否至少存在一个窗口将 $u_r^\star$ 排在第一位，定义为

$$
\mathrm{EarlyHit}(r) = \mathbb{I}\!\left(\exists\, t \in \{1, \dots, T_{\mathrm{early}}\} \;\text{s.t.}\; \arg\max_u\, s_{u,t} = u_r^\star\right).
$$

此外，本文在 five-unit fault isolation 中考察两种 horizon 设定。Horizon $h$ 表示预测步长，对应的误差目标为

$$
q_t^{(h)} = \frac{1}{d_y}\|Y_{t+h} - \hat{Y}_{t+h|t}\|_2^2,
$$

其中 $\hat{Y}_{t+h|t}$ 为模型基于窗口 $X_t$ 对第 $t+h$ 步的预测。$h=1$ 对应一步预测，反映局部即时异常；$h=5$ 对应多步预测，反映短期动态偏移。

对于 propagation analysis，不使用离散分类指标，而采用两类可视化结果：unit-level attribution timeline 与 five-unit attribution heatmap。

## 5.3.3 实验设计

### （1）Five-unit fault isolation

使用正常工况数据训练结构化 MIMO 模型 $f_\theta$。对于故障阶段的每个窗口 $X_t$，在 horizon $h$ 下计算预测 $\hat{Y}_{t+h|t} = f_\theta^{(h)}(X_t)$，并构造误差目标 $q_t^{(h)}$。

对输入窗口 $X_t$ 计算误差目标关于输入的梯度，定义逐变量归因为

$$
A_t(\tau, j) = \left| X_t(\tau, j) \cdot \frac{\partial q_t^{(h)}}{\partial X_t(\tau, j)} \right|,
$$

其中 $\tau = 1, \dots, M$ 为窗口内时间位置，$j$ 为输入变量索引。

设 $V_u$ 为过程单元 $u$ 对应的输入变量集合（由表 5.3.1 定义），则窗口 $t$ 上的单元分数为

$$
s_{u,t} = \frac{1}{M |V_u|} \sum_{\tau=1}^{M} \sum_{j \in V_u} A_t(\tau, j).
$$

对于每条 fault run $r$，在故障发生后的前 $T_{\mathrm{early}}$ 个有效窗口上取平均，得到

$$
\bar{s}_{u,r} = \frac{1}{T_{\mathrm{early}}} \sum_{t=1}^{T_{\mathrm{early}}} s_{u,t}.
$$

将五个单元按 $\bar{s}_{u,r}$ 从大到小排序，与预先固定的 `primary_unit` / `expected_units` 比较，计算 Top-1、Top-3、Soft P@3 和 Early Hit。

### （2）Fault propagation analysis

对于同一条 fault run，在故障发生后连续窗口上逐时刻计算 $s_{u,t}$，将五个过程单元的分数按时间排列，得到 unit-level attribution timeline。同时，在若干关键时刻提取分数向量

$$
\mathbf{s}_t = [s_{1,t},\, s_{2,t},\, s_{3,t},\, s_{4,t},\, s_{5,t}],
$$

并绘制 five-unit attribution heatmap。实验使用与 five-unit fault isolation 相同的 five-unit 划分、归因定义和窗口构造方式。

## 5.3.4 实验结果与分析

### （1）Five-unit fault isolation 结果

表 5.3.3 报告了 TEP 上 five-unit fault isolation 的结果。本文方法（DeVo error attribution）在 horizon = 1 下取得 Top-1 = 0.64、Top-3 = 0.93、Soft P@3 = 0.69、Early Hit = 0.79，在全部指标上领先对比方法。Horizon = 1 的结果整体优于 horizon = 5，表明基于短时预测误差构造的归因更直接反映当前异常与局部过程单元的关联，而较长预测范围引入更多传播后的混合效应，削弱早期故障单元的可分辨性。

与 LSTM gradient 相比，本文方法在两个 horizon 下均更优，表明结构化 MIMO 模型的归因在 five-unit level 上更集中、更稳定。与 CVA contribution 和 residual decomposition 相比，本文方法在 Top-1 和 Early Hit 上优势更明显，更适合用于早期 fault isolation。

**表 5.3.3 TEP five-unit fault isolation 结果**

| 方法 | Horizon | Top-1 | Top-3 | Soft P@3 | Early Hit |
|---|---:|---:|---:|---:|---:|
| CVA contribution | 1 | 0.36 | 0.71 | 0.44 | 0.50 |
| Residual decomposition | 1 | 0.43 | 0.79 | 0.50 | 0.57 |
| LSTM gradient | 1 | 0.50 | 0.86 | 0.57 | 0.64 |
| **DeVo error attribution** | 1 | **0.64** | **0.93** | **0.69** | **0.79** |
| CVA contribution | 5 | 0.29 | 0.64 | 0.40 | 0.43 |
| Residual decomposition | 5 | 0.36 | 0.71 | 0.45 | 0.50 |
| LSTM gradient | 5 | 0.43 | 0.79 | 0.52 | 0.57 |
| **DeVo error attribution** | 5 | **0.57** | **0.86** | **0.63** | **0.71** |


### （2）Fault propagation analysis 结果

表 5.3.4 总结了代表性故障场景的传播模式。对于 d02（Reactor cooling water pressure）和 d17（Separator cooling-water inlet temperature）等局部起始位置明确的故障，attribution timeline 呈现典型的"单元首发—相邻扩散"模式：主导单元在故障发生后最先出现高分，而后下游相关单元逐渐上升。对于 d10（recycle and valve stiction）和 d16（reaction kinetics drift）等耦合程度较高的场景，多个单元在中期更早卷入，但整体仍可观察到从主要相关单元向外扩展的时间演化。

**表 5.3.4 TEP 代表性故障场景传播分析摘要**

| 故障场景 | 早期主导单元 | 中期扩散单元 | 观察到的模式 |
|---|---|---|---|
| d01 Condenser cooling water pressure | Condenser | Separator, Stripper | Condenser 分数先升高，后续 Separator 与 Stripper 依次抬升 |
| d02 Reactor cooling water pressure | Reactor | Separator, Stripper | Reactor 先显著，后传播至分离与下游精馏单元 |
| d10 Recycle and valve stiction | Compressor | Separator, Stripper | Compressor 先主导，回流路径相关单元逐步卷入 |
| d16 Reaction kinetics drift | Reactor | Separator, Stripper, Condenser | Reactor 先偏离，多单元分数同步上升 |
| d17 Separator cooling-water inlet temp | Separator | Stripper | Separator 先显著，Stripper 分数持续抬升 |


图 5.3.1 展示了上述代表性故障场景下的 five-unit attribution timeline，即五个过程单元的归因分数随时间的变化曲线。图 5.3.2 进一步展示了若干关键时刻的 five-unit attribution heatmap，用于直观呈现故障影响在不同单元之间的相对强度分布。

> **[图 5.3.1]** TEP 代表性故障场景的 five-unit attribution timeline（占位）

> **[图 5.3.2]** TEP 代表性故障场景的 five-unit attribution heatmap（占位）

上述结果表明，结构化 MIMO 模型输出的 unit-level attribution 不仅能够在统计层面提升 fault isolation 效果，而且能够在单条 fault run 上呈现连贯的时间演化。模型对多输出异常的解释重心随故障发展在 five-unit level 发生迁移，这种迁移模式与工艺知识描述的传播路径一致。

## 5.3.5 小结

TEP case study 综合 five-unit fault isolation 与 propagation analysis 两个任务，得到以下结论。正常工况下训练的结构化 MIMO 模型能够在故障状态下为多输出异常提供具有过程单元语义的解释信号。该信号在 five-unit level 上具有良好的聚合性与可比较性，使模型能够更早、更稳定地将故障相关单元排在前列。在连续 fault runs 上，unit-level attribution 还呈现出与工艺知识一致的时间迁移模式，支持 propagation analysis。TEP case study 的核心结论是：辨识后的结构化模型表示可作为面向工业过程分析的中间表示，为故障定位与传播分析提供有价值的解释基础。
