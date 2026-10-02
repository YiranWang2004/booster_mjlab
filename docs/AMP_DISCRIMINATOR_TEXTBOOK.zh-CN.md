# 从二分类到运动风格奖励：Booster K1 AMP 判别器

> 本文面向学过线性代数、微积分和基础概率论的本科生。不要求读者事先掌握 GAN 或 PPO。
>
> 实现核对日期：2026-10-02。主要对象为本仓库的 `Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1`。数值配置来自当前源码；某次实验的实际配置应以对应训练目录中的 `params/agent.yaml` 和 `params/env.yaml` 为准。
>
> 本文区分三类内容：**当前实现**是代码实际执行的计算；**教学简化**用于帮助推导；**理想化结论**有额外假设，不能直接当作实际神经网络的保证。

## 学习目标与阅读路线

完成本文后，读者应能够：

1. 解释 AMP 的正负样本从哪里来，以及为什么输入需要包含运动历史。
2. 从伯努利分布推导二元交叉熵，计算它对 logit 的梯度。
3. 写出本仓库的判别器、R1 梯度惩罚和风格奖励的完整公式。
4. 区分判别器学习与策略学习的梯度路径。
5. 解释双卡梯度同步、归一化统计和批次标准差之间的区别。
6. 根据日志提出可验证的诊断，而不是把某个指标越大或越小视为必然更好。

推荐阅读顺序：

- 初次接触：第 1～3 节 → 第 6 节 → 第 9 节 → 第 12 节。
- 理解全部数学细节：按顺序阅读第 1～11 节。
- 对照源码与实验：第 13～16 节。

## 1. 为什么需要一个“运动风格判别器”

### 1.1 任务完成得好，不代表动作自然

设机器人接收到速度命令，强化学习策略需要让实际速度接近命令。只给出速度跟踪奖励时，机器人可能找到满足目标、但动作不理想的办法，例如高频摆腿、明显滑脚或僵硬的上身姿态。

一种办法是逐项手写奖励，分别约束姿态、足部接触、关节运动等。另一种办法是给出一批参考动作，让模型学习“这些动作在统计上有什么共同特点”。

AMP，即 Adversarial Motion Priors，采用后一种思路：训练一个判别器，让它区分参考动作与策略生成的动作，再把判别结果转化为策略的风格奖励。

“参考动作”表示数据来源，并不自动意味着每一帧都物理可行，也不意味着参考动作对当前速度命令最优。

### 1.2 两个学习问题

整个系统同时包含两个不同的问题：

| 学习对象 | 输入 | 输出 | 要解决的问题 |
|---|---|---|---|
| 策略 actor | 策略观测和速度命令 | 关节控制动作 | 怎样运动能得到更高回报 |
| 判别器 | 一段 AMP 状态历史 | 属于参考动作的分数 | 这段运动来自参考数据还是策略 |

此外，critic 估计策略的未来回报，用于降低策略梯度估计的方差。critic 不负责判断动作来自哪个数据源。

两者形成反馈：判别器学习识别策略与参考之间的差异；策略为了获得风格奖励，倾向于生成更接近参考分布的运动。

~~~mermaid
flowchart LR
    C["速度命令和策略观测"] --> P["策略 actor"]
    P --> A["关节动作"]
    A --> S["物理仿真"]
    S --> H["AMP 运动历史"]
    E["参考动作历史"] --> T["判别器训练"]
    H --> T
    H --> D["当前判别器"]
    D --> R["风格奖励"]
    S --> Q["任务奖励"]
    R --> U["PPO 策略更新"]
    Q --> U
    U --> P
~~~

图中“判别器 → 风格奖励 → PPO”是数值信息流，不表示存在一条穿过物理仿真的可微反向传播路径。

### 1.3 本实现的基本事实

当前标准 K1 AMP 使用：

| 项目 | 当前值 |
|---|---|
| 单帧 AMP 特征 | 30 维 |
| 历史帧数 | 10 |
| 控制时间步 | 0.02 s |
| 判别器输入 | 300 维 |
| 隐藏层 | 256、128，ReLU 激活 |
| 批次标准差特征 | 开启 |
| 输入运行归一化 | 开启 |
| 分类损失 | BCE |
| 梯度惩罚 | 参考样本上的 R1，系数 10 |
| 判别器优化器 | Adam |
| 风格混合权重 | 0.3，当前起止值相同 |

配置类还支持其他损失类型，但本文的主要推导只针对实际选中的 BCE 分支。

## 2. 数学符号与最小预备知识

### 2.1 符号表

| 符号 | 含义 |
|---|---|
| $t$ | 环境控制步索引 |
| $s_t,a_t$ | 环境状态、策略动作 |
| $\theta,\psi,\phi$ | actor、critic、判别器参数 |
| $\mathbf o_t$ | 单帧 AMP 特征 |
| $\mathbf x_t$ | 拼接后的 AMP 历史特征 |
| $\hat{\mathbf x}$ | 归一化后的历史特征 |
| $f_\phi,\ell$ | 判别器函数、它输出的 logit |
| $D_\phi=\sigma(f_\phi)$ | 预测为参考数据的概率 |
| $B$ | 一次更新抽取的参考样本数量 |
| $K$ | 历史帧数，当前为 10 |
| $\mathcal E,\mathcal C,\mathcal R$ | 参考、当前策略、回放样本集合 |
| $\alpha$ | 风格奖励混合权重 |
| $\lambda_{\mathrm{gp}}$ | 梯度惩罚系数 |
| $\gamma,\lambda_{\mathrm{GAE}}$ | 折扣因子、GAE 参数 |

不同公式里的“样本”通常是一个 300 维历史向量，而不是一个关节、一个标量或一整段动作文件。

### 2.2 从分数到概率

神经网络最后一层输出可以是任意实数：

$$
\ell\in(-\infty,+\infty).
$$

sigmoid 把它映射到 0 与 1 之间：

$$
D=\sigma(\ell)=\frac{1}{1+\exp(-\ell)}.
$$

反过来：

$$
\ell=\log\frac{D}{1-D}.
$$

所以 logit 是“参考概率与策略概率之比”的对数。比如 $\ell=0$ 对应 $D=0.5$，$\ell=\log 9$ 对应 $D=0.9$。

这里的概率来自一个不断变化的分类问题，不是经过验证的“动作安全概率”“不会摔倒概率”或“跟踪成功率”。

### 2.3 样本均值与期望

如果 $\mathbf x_i$ 从分布 $p$ 中抽样，则：

$$
\mathbb E_{\mathbf x\sim p}[g(\mathbf x)]
\approx\frac1n\sum_{i=1}^n g(\mathbf x_i).
$$

训练中几乎所有期望都由 minibatch 均值近似。抽样方式因此会改变实际优化的目标。

## 3. 把机器人运动变成判别器输入

### 3.1 单帧特征

标准 K1 每条腿有 6 个被选入 AMP 的关节：髋部 3 个、膝部 1 个、踝部 2 个。双腿合计 12 个。

当前单帧特征按以下顺序拼接：

$$
\mathbf o_t=
\begin{bmatrix}
\mathbf q_t^{leg}-\mathbf q_0^{leg}\\
\dot{\mathbf q}_t^{leg}-\dot{\mathbf q}_0^{leg}\\
\mathbf v_t^b\\
\mathbf g_t^b
\end{bmatrix}
\in\mathbb R^{12+12+3+3}
=\mathbb R^{30}.
$$

$\mathbf q_0$ 是配置的默认关节姿态，不是每个 episode 开始时随机得到的姿态。$\dot{\mathbf q}_0$ 是默认关节速度，当前通常为零。

设 $R_t$ 将躯干坐标系向量转换到世界坐标系，则：

$$
\mathbf v_t^b=R_t^\top\mathbf v_t^w,
\qquad
\mathbf g_t^b=R_t^\top(0,0,-1)^\top.
$$

使用躯干坐标系后，机器人面向世界的不同方向，不会仅因全局朝向变化就产生不同的前进速度特征。重力方向能描述躯干倾斜，却不能单独表示绝对 yaw 朝向。

当前 AMP 输入没有速度命令、躯干角速度、手臂状态、绝对位置或足部接触力。AMP 观测组没有开启观测噪声。

> **变体区别：** K1-Parallel 当前没有启用躯干线速度特征，其对应单帧维度为 27、10 帧维度为 270。不要把本文的 300 维输入直接套到该变体。

### 3.2 为什么需要历史

同一个关节姿态可能出现在抬腿和落腿两个阶段。即使包含关节速度，单帧也不容易表达步态是否连续、周期是否合理。

当前历史向量为：

$$
\mathbf x_t=
\operatorname{concat}
(\mathbf o_{t-9},\ldots,\mathbf o_t)
\in\mathbb R^{300}.
$$

虽然输入带有时间顺序，判别器仍是普通 MLP，不是循环网络。网络通过不同输入位置的权重学习时间关联。

控制间隔为 $\Delta t=0.02$ 秒，因此首尾跨度为：

$$
(K-1)\Delta t=9\times0.02=0.18\ \mathrm{s}.
$$

“10 帧覆盖约 0.2 秒”可作口头近似，精确的首尾时间差是 0.18 秒。

### 3.3 episode 边界

新 episode 没有足够历史时，用第一帧重复填充：

$$
\mathbf x_0=[\mathbf o_0,\ldots,\mathbf o_0].
$$

随后逐步替换为新观测。若仿真返回的是终止后重置的观测，代码会把该环境整段历史重填为重置观测，避免连接两个 episode。

参考动作文件开头也通过重复首帧补足历史。这样正负样本的边界处理方式一致，但开头样本的时间变化仍与正常运动阶段不同。

## 4. 正负样本与回放：训练分布是如何定义的

### 4.1 参考样本

默认参考源为 `whirlwind-ams/lafan_locomotion_k1`。当前配置使用镜像以及 0.8、0.9、1.0、1.1、1.2 倍速组合，再按训练时间步准备状态与速度。

参考关节位置需要减去与策略侧一致的默认姿态，所选关节、特征顺序和历史顺序也必须一致。否则判别器可能只学会识别格式差异。

设第 $i$ 个准备后的动作变体有 $T_i$ 帧、采样权重为 $w_i$，则：

$$
P(i,t)=\frac{w_i}{\sum_j w_j}\frac1{T_i}.
$$

默认变体等权时，先在变体之间等概率选择，再在该变体内部均匀选择历史。它不同于把所有帧拼接后无权重均匀抽样：后一种做法会给长动作更大权重。

### 4.2 当前样本和回放样本

一次判别器更新使用：

$$
\mathcal E=\{\mathbf x_i^E\}_{i=1}^{B},\quad
\mathcal C=\{\mathbf x_i^C\}_{i=1}^{B},\quad
\mathcal R=\{\mathbf x_i^R\}_{i=1}^{B}.
$$

其中参考是正样本；当前 rollout 与历史回放共同作为负样本：

$$
\mathcal P=\mathcal C\cup\mathcal R,\quad|\mathcal P|=2B.
$$

回放让判别器不只适应刚采集的策略动作，也能保留对过去策略动作的辨别能力。它并不保证完全消除遗忘。

当前回放池每卡最多保存 200000 个历史向量。未满时写入当前 rollout；已满时每轮随机插入最多 1000 个当前历史，覆盖较旧内容。

**回放更新发生在判别训练抽样之前，因此回放不一定全是过去轮次的数据。** 当前样本与回放样本允许重复，也不保证互不重叠。

### 4.3 批次大小的计算

设每卡环境数为 $N_{\mathrm{env}}$，每轮 rollout 长度为 $T$，PPO minibatch 数量为 $M$，则：

$$
B=\left\lfloor\frac{N_{\mathrm{env}}T}{M}\right\rfloor.
$$

例如每卡 4096 环境、24 步、4 个 minibatch：

$$
B=24576,\quad|\mathcal P|=49152,\quad
|\mathcal P\cup\mathcal E|=73728.
$$

当前每轮 5 个 learning epochs，每个 epoch 4 次更新，所以每轮执行 20 次联合更新。反复使用样本不意味着获得了同样数量的独立新经验。

任务名称中的 DA 是 PPO 的左右对称数据增强；参考动作的镜像增强是另一个配置。不要因为 PPO 批次经过 DA 扩大，就再次把上述 AMP 批次大小翻倍。

## 5. 输入归一化：不仅是“减均值除方差”

### 5.1 为什么要归一化

关节角度、关节速度和躯干速度的单位不同、数值范围不同。若直接混合输入，数值变化大的特征可能在优化初期占据更强影响。

当前实现对 300 个输入维度分别维护均值和方差：

$$
\hat x_j=\frac{x_j-\mu_j}{\sqrt{v_j}+\epsilon_n},
\qquad\epsilon_n=10^{-2}.
$$

注意分母是标准差加 $\epsilon_n$，不是 $\sqrt{v_j+\epsilon_n}$。统计量是每个历史位置分别维护的，不是先把 10 个时间位置混为一组。

归一化初值为均值 0、方差 1、计数 0。第一次更新会根据真实批次替换这些初始统计。

### 5.2 合并旧统计与新批次

旧统计为样本数 $N$、均值 $\mu$、总体方差 $v$。新批次为样本数 $n$、均值 $\mu_b$、总体方差 $v_b$。

令：

$$
N'=N+n,\qquad a=\frac{n}{N'},\qquad\delta=\mu_b-\mu.
$$

则：

$$
\mu'=(1-a)\mu+a\mu_b=\mu+a\delta,
$$

$$
\boxed{
v'=(1-a)v+av_b+a(1-a)\delta^2
}.
$$

最后一项表示新旧均值不同带来的组间方差。只平均两组方差会漏掉它。

**例 5-1：** 两组分别为 $[0,0]$ 与 $[2,2]$。各组方差都是 0，但合并数据 $[0,0,2,2]$ 的均值是 1，方差是 1。漏掉组间项就会错误地得到方差 0。

### 5.3 双卡合并统计

第 $r$ 张卡的新批次统计记为 $n_r,\mu_r,v_r$，则全局批次统计为：

$$
n=\sum_r n_r,\qquad
\mu_b=\frac{\sum_r n_r\mu_r}{n},
$$

$$
v_b=\frac{\sum_r n_r[v_r+(\mu_r-\mu_b)^2]}{n}.
$$

先得到全局批次统计，再与运行统计合并。即使两卡批次数量不同，也应按样本数加权，而不是简单平均两卡均值。

### 5.4 更新时机与权重

当前实现顺序是：

1. 使用已有统计完成分类前向。
2. 使用同一套已有统计完成梯度惩罚计算。
3. 用参考原始历史、策略原始历史依次更新统计。
4. 对之前构造好的损失反向传播并更新网络。

统计更新在 `no_grad` 下进行，不是通过梯度学习。下一次前向才会使用新统计。

参考批次有 $B$ 个、策略批次有 $2B$ 个，因此每次更新加入统计的数据总量按 1:2 计数。后面会看到，BCE 的类别总权重却是 1:1。二者并不矛盾：前者描述输入数据分布，后者定义分类优化目标。

统计量还会重复看到重采样的历史，所以其计数不是“独立物理状态的数量”。

## 6. 判别器网络与批次标准差

### 6.1 两层隐藏网络

对批次中的第 $i$ 个归一化样本：

$$
\mathbf h_i^{(1)}
=\operatorname{ReLU}(W_1\hat{\mathbf x}_i+\mathbf b_1),
\quad W_1\in\mathbb R^{256\times300},
$$

$$
\mathbf h_i^{(2)}
=\operatorname{ReLU}(W_2\mathbf h_i^{(1)}+\mathbf b_2),
\quad W_2\in\mathbb R^{128\times256}.
$$

其中 $\operatorname{ReLU}(z)=\max(0,z)$，逐元素应用。

这使网络能够学习非线性组合，例如“某个躯干速度是否与关节运动和倾斜状态相协调”，而不只是分别检查每个特征是否超限。

### 6.2 批次标准差

设本次前向批次大小为 $m$，则：

$$
\bar h_j=\frac1m\sum_{i=1}^m h_{ij}^{(2)},
$$

$$
s_{\mathcal B}
=\frac1{128}\sum_{j=1}^{128}
\sqrt{\frac1m\sum_{i=1}^m(h_{ij}^{(2)}-\bar h_j)^2}.
$$

标准差使用 `unbiased=False`，分母为 $m$，不是 $m-1$。若 $m=1$，实现直接令 $s_{\mathcal B}=0$。

将这个标量广播并拼接到所有样本：

$$
\tilde{\mathbf h}_i=[\mathbf h_i^{(2)};s_{\mathcal B}]
\in\mathbb R^{129}.
$$

最后：

$$
\ell_i=\mathbf w^\top\tilde{\mathbf h}_i+b,\qquad
D_i=\sigma(\ell_i).
$$

网络尺寸因此为：

$$
300\rightarrow256\rightarrow128\rightarrow129\rightarrow1.
$$

批次标准差给网络提供了本批隐藏特征离散程度的信息。它可以辅助识别缺乏变化的策略行为，但并不保证学到完整的动作多样性。

### 6.3 输出并非严格的逐样本函数

更准确的记法是：

$$
D_\phi(\mathbf x_i;\mathcal B),
$$

而不是仅写 $D_\phi(\mathbf x_i)$。相同样本放进不同批次，输出可能不同。

当前存在三个批次上下文：

| 用途 | 批次内容 | 标准差是否参与输入梯度计算 |
|---|---|---|
| BCE 分类 | 本卡 $2B$ 个负样本与 $B$ 个正样本一起输入 | 是 |
| R1 惩罚 | 本卡 $B$ 个参考样本 | 否，标准差被 detach |
| rollout 风格奖励 | 本卡当前所有环境的策略历史 | 不求梯度 |

归一化统计会跨卡合并；这里的隐藏特征标准差只在本卡当前批次中计算。二者是不同机制。

## 7. 从最大似然推导 BCE

### 7.1 单个样本

设标签 $y\in\{0,1\}$。给定预测概率 $D$，伯努利分布的似然是：

$$
P(y\mid\mathbf x)=D^y(1-D)^{1-y}.
$$

取负对数，得到要最小化的损失：

$$
L(\ell,y)
=-y\log D-(1-y)\log(1-D).
$$

代入 $D=\sigma(\ell)$：

$$
\boxed{
L(\ell,y)=\operatorname{softplus}(\ell)-y\ell
},
\qquad
\operatorname{softplus}(z)=\log(1+e^z).
$$

所以正样本使用 $\operatorname{softplus}(-\ell)$，负样本使用 $\operatorname{softplus}(\ell)$。

实现使用 `BCEWithLogitsLoss`，直接输入 logit，让库内部稳定计算，不需要先显式 sigmoid 再取对数。

### 7.2 梯度方向

因为：

$$
\frac{d}{d\ell}\operatorname{softplus}(\ell)=\sigma(\ell),
$$

故：

$$
\boxed{
\frac{\partial L}{\partial\ell}=D-y
}.
$$

对正样本 $y=1$，梯度为 $D-1\le0$，梯度下降倾向于增大 logit；对负样本 $y=0$，梯度为 $D\ge0$，梯度下降倾向于减小 logit。

这就是判别器“提高参考分数、降低策略分数”的数学来源。

### 7.3 当前代码如何平均

定义各组损失：

$$
L_E=\frac1B\sum_{i=1}^B\operatorname{softplus}(-\ell_i^E),
$$

$$
L_C=\frac1B\sum_{i=1}^B\operatorname{softplus}(\ell_i^C),\qquad
L_R=\frac1B\sum_{i=1}^B\operatorname{softplus}(\ell_i^R).
$$

当前损失为：

$$
\boxed{
L_{\mathrm{BCE}}=\frac12L_E+\frac14L_C+\frac14L_R
}.
$$

尽管负样本数量是正样本的两倍，正负两类的总权重仍相同。这与直接对 $3B$ 个样本统一求均值不同；直接求均值会得到三组各占 $1/3$。

### 7.4 理想化的最优判别器

暂时忽略批次标准差、梯度惩罚、有限网络容量和动态归一化，把判别器视为逐样本函数。设：

$$
p_P=\tfrac12p_C+\tfrac12p_R.
$$

对于固定的参考和策略分布，目标为：

$$
L(D)=-\frac12\int
\left[p_E(\mathbf x)\log D(\mathbf x)
+p_P(\mathbf x)\log(1-D(\mathbf x))\right]d\mathbf x.
$$

对每个 $\mathbf x$ 的 $D$ 求导并令其为零：

$$
-\frac{p_E}{2D}+\frac{p_P}{2(1-D)}=0,
$$

得到：

$$
\boxed{
D^*(\mathbf x)=\frac{p_E(\mathbf x)}{p_E(\mathbf x)+p_P(\mathbf x)}
}.
$$

这个表达式适用于 $p_E+p_P>0$ 的位置；若两者都为零，分类目标不约束该位置的输出。只有一类密度为零时，应通过边界最优值或极限理解结果。

若 $p_E=p_P$，在它们的支持集上理想预测为 $1/2$，BCE 为 $\log2\approx0.693$。

这是帮助理解目标的理想模型，不是当前含批次特征和正则化的神经网络严格满足的公式。实际出现 $D\approx0.5$，也可能是网络尚未学会区分，不能单独据此证明分布已经匹配。

## 8. R1 梯度惩罚：约束分数在参考附近的变化

### 8.1 输入梯度的含义

对一个小扰动 $\Delta\hat{\mathbf x}$，一阶近似给出：

$$
f_\phi(\hat{\mathbf x}+\Delta\hat{\mathbf x})
\approx f_\phi(\hat{\mathbf x})
+\nabla_{\hat{\mathbf x}}f_\phi(\hat{\mathbf x})^\top
\Delta\hat{\mathbf x}.
$$

由柯西–施瓦茨不等式：

$$
|\Delta f|
\lesssim
\Vert \nabla_{\hat{\mathbf x}}f_\phi\Vert _2
\Vert \Delta\hat{\mathbf x}\Vert _2.
$$

若输入梯度很大，微小的状态扰动就可能使分数显著变化。训练中的数值噪声、采样差异和策略变化可能因此引发不稳定的奖励。

R1 在参考样本处惩罚输入梯度的平方范数，使判别器在这些位置附近更平滑。这是局部、采样意义上的约束，不是整个状态空间上的稳定性证明。

### 8.2 当前公式

先把参考样本归一化，再将其从原有计算图中分离：

$$
\hat{\mathbf x}_i^E
=\operatorname{stopgrad}
\left(\frac{\mathbf x_i^E-\boldsymbol\mu}
{\sqrt{\mathbf v}+\epsilon_n}\right).
$$

然后对这个新变量开启求导。当前惩罚为：

$$
\boxed{
L_{\mathrm{R1}}
=\frac{\lambda_{\mathrm{gp}}}{2B}
\sum_{i=1}^B
\left\Vert
\nabla_{\hat{\mathbf x}_i^E}
f_\phi(\hat{\mathbf x}_i^E;\mathcal E)
\right\Vert _2^2
},
\qquad\lambda_{\mathrm{gp}}=10.
$$

也就是“梯度平方范数的批次均值乘以 5”。

注意：

- 求导对象是 **logit**，不是概率 $D=\sigma(f)$。
- 求导自变量是**归一化后的输入**，不是原始关节角度或物理状态。
- 只使用参考样本，不在正负样本之间做插值。
- 惩罚鼓励梯度趋向 0，不是让梯度范数趋向 1。

当前代码只有在选用 Wasserstein 分支时才使用另外的插值梯度惩罚，不能把那个公式当作当前 BCE 任务的公式。

### 8.3 为什么要 detach 批次标准差

如果不 detach，某个样本通过 $s_{\mathcal B}$ 影响其他样本的输出，则：

$$
\frac{\partial\sum_j f_j}{\partial\hat{\mathbf x}_i}
$$

会包含其他样本经由批次统计传回来的项。

R1 路径把隐藏特征标准差作为常数，使样本 $i$ 的输入梯度只通过它自己的主干特征计算。这也使代码中 `scores.sum()` 的输入梯度与逐样本梯度的含义相符。

这里的 detach 只用于梯度惩罚路径。正常 BCE 前向中的标准差仍参与反向传播。

### 8.4 为什么需要二阶自动微分

R1 首先计算：

$$
\mathbf g_i=\nabla_{\hat{\mathbf x}_i}f_\phi.
$$

为了更新判别器，还要计算：

$$
\nabla_\phi\Vert \mathbf g_i\Vert ^2
=2\left(\frac{\partial\mathbf g_i}{\partial\phi}\right)^\top\mathbf g_i.
$$

因此需要保留“求输入梯度”这一步的计算图，代码使用 `create_graph=True`。这里需要的是输入与参数之间的混合导数，不要求显式构造一个完整 Hessian 矩阵。

这也是 AMP 更新比普通 PPO 增加计算和显存开销的原因之一。

### 8.5 线性模型例子

对教学模型：

$$
f(\hat{\mathbf x})=\mathbf w^\top\hat{\mathbf x}+b,
$$

有：

$$
\nabla_{\hat{\mathbf x}}f=\mathbf w,\qquad
L_{\mathrm{R1}}=\frac{\lambda_{\mathrm{gp}}}{2}\Vert \mathbf w\Vert ^2.
$$

当 $\lambda_{\mathrm{gp}}=10$、$\mathbf w=(0.3,0.4)$ 时：

$$
L_{\mathrm{R1}}=5(0.3^2+0.4^2)=1.25.
$$

在线性特例中，R1 看起来像权重 L2 正则；对于多层网络，它约束的是输入到输出的局部敏感性，并不等同于直接约束所有权重。

## 9. 从判别概率到风格奖励

### 9.1 奖励公式与数值稳定性

rollout 期间，策略执行动作后，环境给出新观测。程序先更新 AMP 历史，再用当前判别器计算这一环境步的风格奖励。

当前公式是：

$$
\boxed{
r_{\mathrm{style}}
=c\,\operatorname{softplus}(\ell)
=-c\log(1-D)
},
\qquad c=1.
$$

其中：

$$
-\log(1-\sigma(\ell))
=-\log\frac1{1+e^\ell}
=\log(1+e^\ell).
$$

数值实现采用 `softplus`，避免先得到一个接近 1 的 sigmoid，再对 $1-D$ 取对数导致精度损失。

| $D$ | logit $\ell$ | $r_{\mathrm{style}}$ |
|---:|---:|---:|
| 0.01 | −4.595 | 0.01005 |
| 0.10 | −2.197 | 0.10536 |
| 0.50 | 0 | 0.69315 |
| 0.90 | 2.197 | 2.30259 |
| 0.99 | 4.595 | 4.60517 |

高分表示更容易被判为参考动作。低分表示风格奖励较小，不等于受到一个负的风格惩罚。

### 9.2 当前奖励没有显式上限

配置中虽然有 `reward_clamp_epsilon`，但当前 BCE 的 `predict_reward` 分支不使用它进行概率裁剪。数学上：

$$
\ell\rightarrow+\infty
\quad\Rightarrow\quad
r_{\mathrm{style}}\sim\ell.
$$

不能把奖励假定为始终处于 $[0,1]$。文档中的“权重 0.3”也不表示 AMP 在数值或梯度上永远恰好占 30%。

### 9.3 与任务奖励混合

设 $R_{\mathrm{task},t}$ 是按单位时间表示的任务奖励率。当前环境先计算每步任务奖励：

$$
r_{\mathrm{task},t}=\Delta t\,R_{\mathrm{task},t}.
$$

AMP runner 再混合：

$$
\boxed{
r_t=(1-\alpha)r_{\mathrm{task},t}
+\alpha\Delta t\,r_{\mathrm{style},t}
}.
$$

当前 $\alpha=0.3,\Delta t=0.02$：

$$
r_t=0.7r_{\mathrm{task},t}+0.006r_{\mathrm{style},t}.
$$

若用奖励率统一表示：

$$
r_t=\Delta t\left(0.7R_{\mathrm{task},t}
+0.3r_{\mathrm{style},t}\right).
$$

**例 9-1：** 假设某一步任务奖励率为 4，判别概率为 0.9，则：

$$
r_{\mathrm{task}}=0.02\times4=0.08,
$$

$$
r=0.7\times0.08+0.006\times2.302585
\approx0.069816.
$$

这里不能再对 0.08 乘一次 0.02。风格奖励相对于任务奖励的影响必须使用一致的时间尺度比较。

### 9.4 风格奖励如何影响速度范围

将第 7.4 节的理想判别器代入奖励，得到：

$$
r_{\mathrm{style}}^*(\mathbf x)
=\log\left(1+\frac{p_E(\mathbf x)}{p_P(\mathbf x)}\right).
$$

这说明，在理想化模型中，参考中常见、策略中相对少见的运动能得到较大风格奖励；参考支持很弱而策略大量生成的运动，奖励可能较小。

当前标准 K1 的判别器直接观察躯干线速度，因此扩大速度课程时，可能发生“速度目标要求的动作在参考里很少”的冲突。它不是硬性速度禁令，策略仍可通过增加任务回报来补偿风格奖励损失。

真实网络还依赖关节运动、历史关系和批次上下文。仅比较一个速度维度的最大值，不能完整判断判别结果。

## 10. 策略怎样使用奖励：PPO 与判别器的梯度分工

### 10.1 风格奖励不是可微控制器

计算 rollout 奖励时，代码使用 `torch.no_grad()`。因此：

$$
r_{\mathrm{style}}
\longrightarrow\text{一个保存下来的数值},
$$

而不是从奖励直接沿判别器、历史状态、物理仿真反传到策略参数。

策略通过强化学习的回报和优势估计获得更新。判别器通过带标签的分类样本获得更新。

### 10.2 回报、TD 误差与 GAE

critic $V_\psi(s_t)$ 估计未来折扣回报。忽略记号上的部分可观测性，设非终止标记为 $m_t=1-d_t$，则：

$$
\delta_t=r_t+\gamma m_tV_\psi(s_{t+1})-V_\psi(s_t).
$$

GAE 递推为：

$$
\hat A_t=\delta_t+\gamma\lambda_{\mathrm{GAE}}m_t\hat A_{t+1}.
$$

当前：

$$
\gamma=0.99,\qquad\lambda_{\mathrm{GAE}}=0.95.
$$

价值目标是：

$$
\hat G_t=\hat A_t+V_{\mathrm{old}}(s_t).
$$

这是核心递推；时间上限截断等情况还涉及环境传来的 `time_outs` 及 bootstrap 处理，不能把所有重置都理解成物理失败。

### 10.3 PPO 裁剪目标

令：

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\mathrm{old}}(a_t\mid o_t)}.
$$

当前策略损失为：

$$
L_{\mathrm{policy}}
=-\mathbb E_t\left[
\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t
\right)\right],
$$

其中 $\varepsilon=0.2$。裁剪限制的是单次更新对已采集动作概率的改变，不是直接裁剪机器人速度。

价值预测也使用裁剪。设：

$$
\tilde V_t=V_{\mathrm{old},t}
+\operatorname{clip}(V_{\psi,t}-V_{\mathrm{old},t},-\varepsilon,\varepsilon),
$$

则当前价值损失为：

$$
L_{\mathrm{value}}
=\mathbb E_t
\left[
\max\left(
(V_{\psi,t}-\hat G_t)^2,
(\tilde V_t-\hat G_t)^2
\right)\right].
$$

当前没有在这个价值损失前额外乘 $1/2$。

PPO 部分组合为：

$$
L_{\mathrm{PPO}}
=L_{\mathrm{policy}}+1.0L_{\mathrm{value}}
-0.01\mathcal H(\pi_\theta).
$$

当前 DA 任务启用对称数据增强，但没有启用 mirror loss，因此这里没有额外的镜像损失项。

### 10.4 联合 backward 不等于共享优化目标

程序对以下总损失调用一次 backward：

$$
L_{\mathrm{total}}
=L_{\mathrm{PPO}}+L_{\mathrm{BCE}}+L_{\mathrm{R1}}.
$$

但是保存下来的奖励、动作历史与监督样本不保留到物理仿真和旧策略的梯度路径，网络参数又相互独立，因此：

$$
\nabla_\theta L_{\mathrm{total}}
=\nabla_\theta L_{\mathrm{PPO}},
\qquad
\nabla_\phi L_{\mathrm{total}}
=\nabla_\phi(L_{\mathrm{BCE}}+L_{\mathrm{R1}}).
$$

风格奖励对策略的影响已经进入 $r_t$、$\hat A_t$ 和 $\hat G_t$。它不是在 actor 的损失里再增加一个可直接反传的判别器项。

同理，$\alpha=0.3$ 不会把判别器分类损失乘以 0.3；该系数控制的是策略收到的奖励混合。

## 11. 参数更新与双卡同步

### 11.1 判别器仍由 Adam 更新

即使任务名称带 Muon，判别器也使用 Adam。当前参数组为：

| 参数 | 优化器 | 权重衰减 |
|---|---|---:|
| actor/critic 二维矩阵 | Muon | 由 Muon 配置决定，当前为 0 |
| actor/critic 其他参数 | Adam | 0 |
| 判别器主干，包括偏置 | Adam | $10^{-3}$ |
| 判别器输出头，包括偏置 | Adam | $10^{-1}$ |

当前 Adam 使用耦合的 L2 权重衰减。对某组判别参数：

$$
g_k=\nabla_\phi(L_{\mathrm{BCE}}+L_{\mathrm{R1}})
+\lambda_{\mathrm{wd}}\phi_k.
$$

Adam 的基本更新为：

$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\qquad
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,
$$

$$
\hat m_k=\frac{m_k}{1-\beta_1^k},\qquad
\hat v_k=\frac{v_k}{1-\beta_2^k},
$$

$$
\phi_{k+1}
=\phi_k-\eta_k\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon_A}.
$$

这里的 $v_k$ 是梯度二阶矩，与输入归一化方差是不同对象。当前使用 Adam 默认的 $\beta_1=0.9,\beta_2=0.999,\epsilon_A=10^{-8}$。

权重衰减由优化器加入，日志中的 BCE loss 和 R1 loss 不包含这部分数值。

### 11.2 学习率与裁剪

判别器各参数组当前学习率缩放因子均为 1，与 PPO 自适应学习率保持一致。初始学习率为 $10^{-3}$。

设策略的新旧分布平均 KL 为 $k$，目标 KL 为 $k_0=0.01$，当前调整规则为：

$$
\eta'=
\begin{cases}
\max(10^{-5},\eta/1.5), & k>2k_0,\\
\min(10^{-2},1.5\eta), & 0<k<k_0/2,\\
\eta, & \text{其他情况}.
\end{cases}
$$

这意味着判别器的学习率由策略 KL 间接调节，并没有独立的判别器学习率调度器。

当前 `max_grad_norm=1.0` 只用于 actor 和 critic，判别器梯度没有应用这一裁剪。

### 11.3 同步训练的数学含义

设有 $W$ 张卡，第 $r$ 张卡根据本地样本得到损失梯度 $g^{(r)}$。更新前平均：

$$
\bar g=\frac1W\sum_{r=0}^{W-1}g^{(r)}.
$$

各卡使用相同参数、相同优化器状态和相同 $\bar g$ 执行更新，才能继续保持模型一致。

当前实现同步 actor、critic 和 AMP 判别器的梯度；开始学习时广播模型状态，判别器广播包括归一化缓冲区。正常续训还会从 checkpoint 恢复优化器状态。

如果只同步 actor/critic 而遗漏判别器，每张卡就可能形成不同的风格奖励函数。这正是 AMP 不能仅依靠普通 PPO 的梯度同步方法的原因。

### 11.4 同步与不共享的内容

| 内容 | 当前处理 |
|---|---|
| actor、critic、判别器参数 | 起始同步，梯度平均后同步更新 |
| 运行归一化统计 | 使用跨卡合并的统计 |
| 学习率 | 聚合 KL 后由 rank 0 决定并广播 |
| 环境状态和随机种子 | 每卡独立 |
| 当前 rollout 和 AMP 回放池 | 每卡独立 |
| 参考采样 | 每卡独立采样 |
| 隐藏特征批次标准差 | 每卡本地计算 |
| 日志和 checkpoint 写入 | rank 0 执行 |

因此双卡训练不是要求每张卡处理相同动作，而是让它们用不同样本共同估计更新方向。

由于网络含本地批次标准差，双卡梯度平均严格对应“两个本地批次目标的平均”；它不保证等同于把两卡全部样本拼成一个大批次、重新计算全局隐藏标准差后得到的梯度。

### 11.5 环境数量和课程步数

命令：

~~~bash
uv run train Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 \
  --gpu-ids '[0, 1]' \
  --env.scene.num-envs 4096
~~~

表示每卡 4096 个环境，共 8192 个环境。每轮每卡仍推进 24 个控制步；增加 GPU 不会把配置中按环境步计数的课程切换迭代提前一半。

显存和 GPU 利用率还受到缓存、图形程序和其他进程影响，不能只凭整卡利用率判断梯度同步是否正确。

## 12. 一轮训练的完整时间顺序

理解顺序很重要：本轮 rollout 的奖励由采集时的判别器给出；随后判别器虽然更新多次，但代码不会回头重算已经保存在该 rollout 中的风格奖励。

~~~text
初始化
  构建环境、actor、critic、判别器、参考加载器和回放池
  用当前 AMP 观测填充 10 帧历史
  双卡模式下同步模型状态

每轮训练
  采集 24 个控制步
    actor 根据策略观测输出动作
    环境推进，得到任务奖励、下一观测和 done
    更新 AMP 历史；done 环境重新填充历史
    当前判别器计算风格奖励（不求梯度）
    按 0.7 / 0.3 和时间步缩放混合奖励
    保存 PPO transition 和当前 AMP 历史

  critic 估值，计算 GAE、回报目标和优势归一化
  消费当前 AMP 历史，刷新回放池

  执行 5 × 4 = 20 次更新
    抽取 PPO minibatch；按配置执行对称数据增强
    抽取 B 个参考、B 个当前策略、B 个回放历史
    用已有运行统计归一化 AMP 输入
    对 3B 混合样本计算分类 logits
    对 B 个参考样本重新前向，计算 R1
    计算 BCE；更新 AMP 运行归一化统计
    对 PPO + BCE + R1 执行 backward
    双卡模式下平均 actor、critic、判别器梯度
    裁剪 actor 和 critic 梯度
    优化器更新

  从 rollout 更新 actor / critic 的观测归一化统计
  清空 PPO rollout storage
  rank 0 记录日志；到保存间隔时写 checkpoint
~~~

这是逻辑层面的伪代码。PPO 损失计算、自适应 KL 调整与 AMP 损失在代码中交织进行；无关的数据准备细节省略。

### 12.1 两条不同的梯度路径

~~~mermaid
flowchart TD
    R["已保存的混合奖励"] --> G["GAE 和回报目标"]
    G --> LP["PPO 损失"]
    LP --> AC["actor / critic 梯度"]
    X["已保存或加载的 AMP 历史"] --> F["判别器前向"]
    F --> LD["BCE + R1"]
    LD --> DG["判别器梯度"]
    AC --> AR["跨卡梯度平均"]
    DG --> AR
    AR --> OPT["优化器更新"]
~~~

两条路径在程序上可以用一次 backward 处理，但没有把参考分类误差直接反向传播到物理仿真。

### 12.2 为什么回放与 PPO 可以同时出现

PPO 的策略更新使用当前 rollout 的动作、旧概率和优势。AMP 判别器则使用当前与回放运动做监督分类。

因此“AMP 使用回放”不意味着“PPO 直接使用旧回放动作做策略更新”。不同模块对数据的要求不同。

## 13. 一组可以手算和运行验证的例子

### 13.1 一个小批次

设 $B=2$，网络输出的 logits 为：

$$
\ell^E=(2,1),\qquad
\ell^C=(-1,0),\qquad
\ell^R=(-2,0.5).
$$

对应损失：

$$
L_E=\tfrac12[\operatorname{softplus}(-2)+\operatorname{softplus}(-1)]
\approx0.220095,
$$

$$
L_C=\tfrac12[\operatorname{softplus}(-1)+\operatorname{softplus}(0)]
\approx0.503204,
$$

$$
L_R=\tfrac12[\operatorname{softplus}(-2)+\operatorname{softplus}(0.5)]
\approx0.550502.
$$

当前实现得到：

$$
L_{\mathrm{BCE}}
=0.5L_E+0.25L_C+0.25L_R
\approx0.373474.
$$

若错误地对六个样本统一平均，则得到：

$$
(L_E+L_C+L_R)/3\approx0.424601.
$$

两个答案不同，原因不是数值误差，而是定义了不同的类别权重。

### 13.2 可运行的 PyTorch 算例

在仓库的 uv 环境中，将下面代码保存为临时 Python 文件后执行 `uv run python 文件名.py`。它只运行 CPU 上的小型数学算例，不启动仿真或训练。

本例验证 BCE 恒等式、logit 梯度、R1 的线性特例、奖励混合与方差合并；它不是完整训练脚本。

~~~python
import torch
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)

# 1. 三组样本分别求均值，保证正负类别总权重各为一半。
expert = torch.tensor([2.0, 1.0])
current = torch.tensor([-1.0, 0.0])
replay = torch.tensor([-2.0, 0.5])
policy = torch.cat((current, replay))

loss_e = F.binary_cross_entropy_with_logits(expert, torch.ones_like(expert))
loss_p = F.binary_cross_entropy_with_logits(policy, torch.zeros_like(policy))
loss = 0.5 * (loss_e + loss_p)
equivalent = (
    0.5 * F.softplus(-expert).mean()
    + 0.25 * F.softplus(current).mean()
    + 0.25 * F.softplus(replay).mean()
)
torch.testing.assert_close(loss, equivalent)
torch.testing.assert_close(loss, torch.tensor(0.37347415755295477))

# 2. 对未平均 BCE，验证 dL/dlogit = sigmoid(logit) - label。
logits = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
labels = torch.tensor([0.0, 1.0, 1.0])
per_sample = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
gradient = torch.autograd.grad(per_sample.sum(), logits)[0]
torch.testing.assert_close(gradient, logits.detach().sigmoid() - labels)

# 3. 验证 softplus(logit) = -log(1 - D)。
# 这里只使用适中的 logits；真实代码用 softplus 避免极端概率的精度问题。
score = torch.tensor([-2.0, 0.0, 2.0])
probability = score.sigmoid()
reward = F.softplus(score)
torch.testing.assert_close(reward, -torch.log1p(-probability))

# 4. 线性判别器的 R1，并验证它对权重的二阶求导结果。
weight = torch.tensor([0.3, 0.4], requires_grad=True)
normalized_input = torch.tensor([[0.0, 1.0], [2.0, -1.0]], requires_grad=True)
linear_score = normalized_input @ weight
input_gradient = torch.autograd.grad(
    linear_score.sum(), normalized_input, create_graph=True
)[0]
r1 = 0.5 * 10.0 * input_gradient.square().sum(dim=1).mean()
torch.testing.assert_close(r1, torch.tensor(1.25))
r1_weight_gradient = torch.autograd.grad(r1, weight)[0]
torch.testing.assert_close(r1_weight_gradient, 10.0 * weight.detach())

# 5. 两个局部方差为零的批次，合并后方差不为零。
left = torch.tensor([0.0, 0.0])
right = torch.tensor([2.0, 2.0])
n_left, n_right = left.numel(), right.numel()
mean = (n_left * left.mean() + n_right * right.mean()) / (n_left + n_right)
variance = (
    n_left * (left.var(unbiased=False) + (left.mean() - mean).square())
    + n_right * (right.var(unbiased=False) + (right.mean() - mean).square())
) / (n_left + n_right)
torch.testing.assert_close(mean, torch.tensor(1.0))
torch.testing.assert_close(variance, torch.cat((left, right)).var(unbiased=False))

# 6. 任务奖励率与风格奖励使用同一时间尺度。
dt, alpha = 0.02, 0.3
task_reward_rate = torch.tensor(4.0)
style_probability = torch.tensor(0.9)
style_reward = -torch.log1p(-style_probability)
mixed_reward = (1 - alpha) * dt * task_reward_rate + alpha * dt * style_reward
torch.testing.assert_close(mixed_reward, torch.tensor(0.06981551055796428))

print(f"BCE = {loss.item():.6f}")
print(f"R1 = {r1.item():.6f}")
print(f"mixed reward = {mixed_reward.item():.6f}")
print("All numerical checks passed.")
~~~

预期输出：

~~~text
BCE = 0.373474
R1 = 1.250000
mixed reward = 0.069816
All numerical checks passed.
~~~

## 14. 如何读懂判别器日志

### 14.1 各个指标测量什么

下面使用代码中的指标名称。具体 TensorBoard 标签可能由日志器再添加前缀。

| 指标 | 数学含义 | 不能直接推出的结论 |
|---|---|---|
| `policy_pred` | 负样本预测概率的批次均值 | 高就一定跟踪更准确 |
| `expert_pred` | 正样本预测概率的批次均值 | 高就一定能物理复现参考 |
| `accuracy_policy` | 负样本分类正确的比例 | 高就一定有利于策略学习 |
| `accuracy_expert` | 正样本分类正确的比例 | 高就说明策略自然 |
| `Discriminator/loss` | $L_{\mathrm{BCE}}$ | 越低越接近理想运动策略 |
| `Discriminator/grad_penalty` | 已乘系数后的 $L_{\mathrm{R1}}$ | 越低越好且不影响分类能力 |
| `Episode_Metrics/Amp/style_reward` | 本次结束的 episodes 中，累计加权风格奖励的均值 | 可以脱离 episode 长度直接比较 |

当前 BCE 指标通过 `round(sigmoid(logit))` 判别类别，近似等价于以 0.5 为阈值；概率恰好为 0.5 时，PyTorch 的 round 会得到 0。

`policy_pred` 来自用于分类的当前策略与回放混合负样本，不是只对当前 rollout 的动作评分。

分类 logits 是在该 minibatch 更新前得到的；损失与指标随后按多次更新累积。双卡下这些额外判别指标没有全部做跨卡均值归约，rank 0 记录的主要是其本地样本统计。模型梯度同步与日志指标聚合是两件事。

### 14.2 三种常见现象

**现象 A：参考概率接近 1，策略概率接近 0。**

这说明当前分类任务容易。可能是策略动作确实与参考相差很大，也可能是特征格式、坐标系或默认姿态不一致。应先核对数据管线，再判断是否需要调整判别器强度。

**现象 B：两类概率都在 0.5 附近。**

可能是两类分布更接近，也可能是判别器欠拟合、正则过强或训练尚未起作用。应结合分类损失变化、运动视频和任务误差判断。

**现象 C：风格奖励增加，但速度跟踪变差。**

可能是策略更接近参考风格，却偏离命令目标；也可能只是 episode 变长，累计奖励上升。应同时记录按步平均的风格奖励、episode 长度、速度分桶误差和跌倒率。

这些解释都是待验证的假设。仅凭单条曲线无法确定原因。

### 14.3 常见理解错误

| 错误理解 | 当前实现中的事实 |
|---|---|
| 判别器逐帧对齐某一段参考 | 它做分布分类，没有给每个策略时刻指定对应参考帧 |
| 判别器输出是动作质量的绝对标尺 | 它依赖当前正负样本分布、参数和归一化状态 |
| 负样本有两倍数量，因此损失权重也有两倍 | 正负两类分别求均值后各占一半 |
| 判别器知道当前命令速度 | 当前 AMP 特征中没有命令 |
| 风格高分说明动作一定物理可行 | 风格得分不构成动力学可行性证明 |
| actor 直接通过判别器做普通监督梯度下降 | actor 使用混合奖励构造 PPO 优势 |
| AMP 权重 0.3，也会把 BCE 损失乘 0.3 | 该权重只用于策略的奖励混合 |
| Muon 任务里判别器也用 Muon | 判别器仍使用 Adam |
| 双卡必须拥有相同回放数据 | 回放各卡独立，梯度同步 |
| `max_grad_norm=1` 限制全部网络 | 当前只裁剪 actor、critic |
| `reward_clamp_epsilon` 保证 BCE 奖励有上限 | 当前 BCE 奖励分支没有用它裁剪 |

## 15. 练习题与参考答案

### 15.1 练习题

1. **输入维度。** 将标准 K1 历史帧数改为 15，输入维度与首尾跨度是多少？若同时加入三维躯干角速度呢？
2. **BCE 梯度。** 正样本 logit 为 0。计算单样本 BCE 及其对 logit 的导数。如果它是负样本，导数如何变化？
3. **类别权重。** 参考组平均损失为 0.2，当前策略组为 0.6，回放组为 1.0。计算本仓库的 BCE，再计算三组统一平均的结果。
4. **采样概率。** 两个参考变体分别有 100 帧和 1000 帧，权重相同。各变体被抽中的概率是多少？各自某个指定帧的概率是多少？
5. **奖励单位。** 判别概率为 0.5，任务每步奖励为 0.1，$\alpha=0.3,\Delta t=0.02$。计算混合奖励。
6. **局部梯度。** 线性判别器 $f=2\hat x_1-\hat x_2+b$，$\lambda_{\mathrm{gp}}=10$。求 R1。偏置会直接改变这个 R1 吗？
7. **双卡统计。** 卡 0 的两个样本均为 1，卡 1 的三个样本均为 2。求合并均值和总体方差。
8. **反向传播。** 为什么可以写一个总损失调用 backward，却说 BCE 不直接训练 actor？
9. **实验设计。** 若怀疑判别器阻碍高速侧移，怎样区分“风格冲突”与“物理或任务奖励瓶颈”？
10. **批次上下文。** 相同历史在 rollout 打分与分类 minibatch 中得到不同 logit，是否一定是程序错误？

### 15.2 参考答案

1. 不加角速度时为 $15\times30=450$ 维，首尾跨度 $14\times0.02=0.28$ 秒。加角速度后单帧 33 维，共 495 维。修改时需保持参考与策略两侧布局一致。
2. $D=0.5$，正样本 BCE 为 $\log2\approx0.693147$，导数为 $-0.5$。负样本导数为 $+0.5$。
3. 当前权重下 $0.5\times0.2+0.25\times0.6+0.25\times1=0.5$。统一平均为 $0.6$，两者类别权重不同。
4. 各变体概率均为 $1/2$。指定帧概率分别为 $1/200$ 和 $1/2000$。
5. $r=0.7\times0.1+0.006\log2\approx0.074159$。题目已给每步任务奖励，不应再将它乘以 $\Delta t$。
6. $L_{\mathrm{R1}}=5(2^2+(-1)^2)=25$。输入梯度与偏置无关，因此偏置不直接改变该惩罚。
7. 均值为 $8/5=1.6$，方差为 $[2(1-1.6)^2+3(2-1.6)^2]/5=0.24$。简单平均两卡均值会错误地得到 1.5。
8. 总损失包含独立计算图分支。BCE 的历史是保存下来的数据，没有连接到当前 actor 参数；actor 从 PPO 分支获得梯度。
9. 固定动力学、任务奖励、命令采样和预算，比较 AMP 权重或参考覆盖。按速度分桶评估误差、成功率、风格奖励、滑动与力矩饱和，并使用多个随机种子。降低风格权重有效，支持风格冲突假设；各版本均失败时还需检查动力学、奖励与探索，不足以直接宣称达到硬件极限。
10. 不一定。网络使用批次标准差；批次内容、运行归一化统计或模型参数变化均可能改变输出。严格比较时应固定这些条件。

## 16. 从公式定位到源码

下表使用仓库内相对链接。行号可能随版本变化，建议按函数名定位。

| 本文内容 | 文件与主要入口 |
|---|---|
| AMP 观测、风格权重 | [_amp_wrapper.py](../src/booster_mjlab/tasks/velocity/config/_amp_wrapper.py)，`with_amp_obs_group` |
| 标准 K1 启用线速度 | [k1/__init__.py](../src/booster_mjlab/tasks/velocity/config/k1/__init__.py)，`_with_amp_reset_cfg` |
| K1-Parallel 的注册差异 | [k1_parallel/__init__.py](../src/booster_mjlab/tasks/velocity/config/k1_parallel/__init__.py) |
| BCE、层宽、PPO 配置 | [k1/rl_cfg.py](../src/booster_mjlab/tasks/velocity/config/k1/rl_cfg.py)，`booster_k1_amp_ppo_runner_cfg` |
| 判别器配置默认值 | [amp/config.py](../src/booster_mjlab/amp/config.py)，`AmpDiscriminatorCfg` |
| 历史、回放配置与训练循环 | [amp_on_policy_runner.py](../src/booster_mjlab/amp/runners/amp_on_policy_runner.py) |
| 参考历史和采样权重 | [motion_loader.py](../src/booster_mjlab/motion/motion_loader.py)，`MotionLoader` |
| 重采样与速度计算 | [motion_data.py](../src/booster_mjlab/motion/motion_data.py)，`MotionFile.prepare` |
| 参考镜像与变速 | [motion_augmentations.py](../src/booster_mjlab/motion/motion_augmentations.py) |
| 历史、回放与分类样本 | [amp.py](../src/booster_mjlab/amp/modules/amp.py)，`AMP`、`AmpReplayBuffer` |
| 网络、R1 与风格奖励 | [discriminator.py](../src/booster_mjlab/amp/modules/discriminator.py)，`Discriminator` |
| BCE 类别平均 | [losses.py](../src/booster_mjlab/amp/modules/losses.py)，`BCELoss.forward` |
| 联合更新与双卡同步 | [amp_ppo.py](../src/booster_mjlab/amp/algorithms/amp_ppo.py)，`AmpPPO.update` |
| 每步奖励混合 | [amp_support.py](../src/booster_mjlab/amp/runners/amp_support.py)，`_amp_collect_step` |
| Muon 与 Adam 分组 | [muon.py](../src/booster_mjlab/rl/muon.py)，`HybridMuonOptimizer` |
| 双进程回归检查 | [test_amp_distributed.py](../tests/test_amp_distributed.py) |

输入归一化与基础 PPO 位于安装依赖 `rsl_rl` 中。可在当前环境定位其源码：

~~~bash
uv run python -c 'import inspect; from rsl_rl.modules import EmpiricalNormalization; print(inspect.getfile(EmpiricalNormalization))'
uv run python -c 'import inspect; from rsl_rl.algorithms.ppo import PPO; print(inspect.getfile(PPO))'
~~~

阅读代码时建议沿两条调用链：

~~~text
奖励链：
AmpOnPolicyRunner.learn
  → AmpRunner._amp_collect_step
  → AmpPPO.process_amp_step → AMP.process_step
  → AmpPPO.predict_style_reward → AMP.predict_style_reward
  → Discriminator.predict_reward
  → 混合任务奖励与风格奖励

判别器学习链：
AmpPPO.update
  → AMP.generators
  → AMP.compute_discriminator_batch
  → Discriminator.forward
  → Discriminator.compute_loss
      → Discriminator.compute_grad_penalty
      → BCELoss.forward
  → Discriminator.update_normalization
  → total_loss.backward
  → AmpPPO.reduce_parameters（双卡）
  → optimizer.step
~~~

## 附录：关键公式速查

| 计算 | 公式 |
|---|---|
| 单帧特征 | $\mathbf o=[\mathbf q-\mathbf q_0;\dot{\mathbf q}-\dot{\mathbf q}_0;\mathbf v^b;\mathbf g^b]$ |
| 历史拼接 | $\mathbf x_t=[\mathbf o_{t-9};\ldots;\mathbf o_t]$ |
| 输入归一化 | $\hat{\mathbf x}=(\mathbf x-\boldsymbol\mu)/(\sqrt{\mathbf v}+\epsilon_n)$ |
| 判别概率 | $D=\sigma(f_\phi)$ |
| 单样本 BCE 梯度 | $\partial L/\partial\ell=D-y$ |
| 三组分类损失 | $L_{\mathrm{BCE}}=\frac12L_E+\frac14L_C+\frac14L_R$ |
| R1 | $L_{\mathrm{R1}}=5\mathbb E_E[\Vert \nabla_{\hat{\mathbf x}}f_\phi\Vert _2^2]$ |
| 风格奖励 | $r_{\mathrm{style}}=\operatorname{softplus}(f_\phi)=-\log(1-D)$ |
| 混合奖励 | $r_t=0.7r_{\mathrm{task},t}+0.006r_{\mathrm{style},t}$ |
| 联合损失 | $L_{\mathrm{total}}=L_{\mathrm{PPO}}+L_{\mathrm{BCE}}+L_{\mathrm{R1}}$ |
| 多卡梯度 | $\bar g=W^{-1}\sum_r g^{(r)}$ |

解释训练行为时，需要同时核对判别器看到了什么、正负样本怎样加权、奖励通过哪条路径影响策略。这三点是把公式与实验联系起来的基础。
