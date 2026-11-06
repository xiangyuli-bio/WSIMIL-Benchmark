# SCL-WC：完整配方接入前的源码核验

目前仍为待接入方法，不计入已适配数量。固定来源为
[Xiyue-Wang/SCL-WC](https://github.com/Xiyue-Wang/SCL-WC)，提交
`d8d0ff44e887cadeeaff199cd771af9ee23e5855`。
本文的观察仅针对这个公开版本；不能推断论文实验实际使用了相同训练路径。

## 原始方法包含的机制

[NeurIPS 2022 正文](https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf)
第 3.3 节同时定义类别特定注意力聚合 CDA、正负区域分离 PNM、跨切片对比 WSCL。
PNM 的 Eq.5 使用一个类别无关的可学习权重头生成正、负区域特征。
WSCL 的 Eq.7 则让当前可微的 anchor 实例与其他训练切片的代表实例进行比较。
因此，只有注意力分类和实例伪标签损失的训练不能代表这套完整方法。

[补充材料](https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Supplemental-Conference.pdf)
第 5 节列出的下游配方为单 bag、Adam、lr=1e-3、weight decay=1e-5、cosine、
temperature=.07、k=8、分类/实例损失权重 .8/.2、PNM/WSCL 权重 .2/.01。
公开 example YAML 中 lr=3e-4、k 默认值=5 等配置与此不同，应分别命名和登记。

## 固定源码中的四项问题

1. `training_methods/SCL.py::train_loop` 接收 epoch，但调用模型时没有传入。
   模型的 epoch 默认值为 0，而对比损失仅在 `epoch>4` 时启用。
2. 即使显式传入 epoch=5，`CLAM_SB.inst_eval` 的对比计算也只读取三个 memory 的历史张量。
   `Memory._dequeue_and_enqueue` 使用 `no_grad` 更新 buffer；当前网络特征并未作为可微 anchor
   进入该对比损失。因此对比项的非零数值不表示网络得到了对比训练。
3. `SCL` 实例化了 `att_head`，但其 forward 从不调用它，也没有将 PNM 损失加入总损失。
   所有 PNM 头参数均没有梯度。仅修复 epoch 参数不能解决这个缺失。
4. memory 随机初始化、没有有效长度和来源切片标识；公开逻辑可能采样未写入的随机槽位。
   类别 0/1 的硬编码还不足以表达论文中的多种阳性类别分开存储。

CPU 独立探针直接从上述固定 Git 对象加载作者定义，使用原生 768 维输入、60 个实例、
分别检查 label=0/1 与 epoch=0/5，共四种情况。诊断将每个队列容量设为 256，保留原始 128 项采样，
随机种子为 2026；它不是原生完整训练验收。两种标签在 epoch=0 时对比项为 0；
epoch=5 时分别为 11.534370422363281 和 11.480781555175781，均为
`requires_grad=False`，PNM 头带梯度参数数均为 0。
这些数值是合成数据的软件诊断，不能用于推断真实数据上的效果。

可用固定作者源码缓存重做该诊断：

```bash
python scripts/audit_sclwc.py --output /path/to/new_sclwc_diagnostic.json
```

本轮实际输出保存在 [诊断记录](validation/sclwc_diagnostic.json)。脚本只检查原始行为，
不会向方法目录加入一个缺少 PNM/WSCL 的执行配置。

## 接入时必须解决的定义

完整适配必须同时提供 CDA、PNM 和有当前网络梯度的 WSCL，以及训练专用、可恢复、
排除当前切片的有效 memory。分类头、负类语义、多阳性类别、袋大小、队列容量和采样均要登记。
验证和测试期间不更新 memory，也不能利用其标签生成训练状态。

正文 Eq.4/6 的印刷符号与相邻的交叉熵说明并不一致；PNM 分类头的共享方式亦不能从
公开 forward 得到验证。在取得更完整实现之前，若依据公式与文字重建，应将符号解释、
头共享方式、当前 anchor/历史 key 配对及 memory 初始化明确登记为重建决策，
并与未经修正的源码对照分开。不能把一个通用 SupCon 加注意力分类器直接命名为原始完整方法。

准入需要逐项验证 PNM、WSCL 对当前网络的非零梯度，独立的 Eq.5/7 数值对照，
跨切片键值隔离、多类别 memory、恢复训练的状态一致性，以及完整原生维度的训练运行。
现阶段没有将诊断探针或仅 CDA 的路径纳入执行配置。

## 0.58 完整候选

已在隔离候选中实现完整 CDA/PNM/可反传 WSCL 及完整训练、恢复流程；正式准入仍待验收。
原始源码对照与完整论文参考分开，详见 [SCLWC.md](SCLWC.md)。本文原始诊断保留，不改写为作者完整实现。
