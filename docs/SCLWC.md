# SCL-WC：完整论文参考实现和固定源码对照

这是待验收的 0.58 候选，尚未加入正式版本的方法数。两个执行 ID 为：

| ID | 范围 |
|---|---|
| `sclwc_paper_reference` | 完整 CDA、PNM、可反传 WSCL 和下游训练流程；论文与公开源码缺项之间的重建决策逐项声明 |
| `sclwc_literal_source_control` | 固定作者网络及实际 `train_loop` 的对照，保留缺失 PNM、未传 epoch、无有效 WSCL 梯度的原始行为；不能单独称为完整 SCL-WC |

固定来源为 [Xiyue-Wang/SCL-WC](https://github.com/Xiyue-Wang/SCL-WC)，commit
`d8d0ff44e887cadeeaff199cd771af9ee23e5855`。
[NeurIPS 2022 论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf)
第 3.3 节 Eq.1–7 定义下游网络与三类目标；
[补充材料](https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Supplemental-Conference.pdf)
第 5 节定义主要超参数。
来源使用私有命名空间从已核验 Git 对象读取，没有将作者实现复制到发行包。
README 的 GPLv3 与学术用途条款、缺少根 LICENSE 的事实仍保留在来源锁中。

## 数据包与调用

使用标准冻结 patch 特征、真实患者映射和固定划分。完整注册的 FM 通道均进入原生
`D→512` 投影，随后为 256 维 gated attention；不截断到原始 768 维。
训练、选模和预测都以单张 slide 为单位。若需要患者级汇总，应作为后续明确的评价条件。

PNM 的背景目标必须明确登记。对于真实正常组织参考，填写：

```yaml
training_unit: slide
evaluation_unit: slide
sclwc:
  negative_class: normal  # 必须是 task.classes 内的真实正常组织类名
  negative_class_role: normal_tissue
```

不能仅因为 BRCA 的 IDC/ILC 或 COAD 的某两个癌症亚型恰好编码为 0/1，就把第 0 类
自动解释为正常组织。补充材料第 8 节也报告了 MSI 与 NSCLC 亚型任务，却没有公开这些任务
的 PNM 背景标签映射。因此也支持明确命名的参考终点条件；它保留全部 CDA/PNM/WSCL，
但不能声称我们知道未公开的原始映射：

```yaml
sclwc:
  negative_class: reference_endpoint_name  # 必须出现在 classes 中
  negative_class_role: reference_endpoint_class
  background_target_provenance: Explicit endpoint reference for PNM; not benign tissue
```

这一条件必须写明目标解释，不自动将分子阴性或某一癌症亚型重命名为正常组织。
论文参考至少需要每类两张不同 TRAIN 切片，供真正的跨切片正样本使用；每 bag 至少 2k 个
实际 patch，以保留 top/bottom 两组。没有伪造或复制 patch 来凑输入数量。

```bash
milplus setup --method sclwc
milplus run --data /path/to/task_bundle --method SCL-WC \
  --config configs/sclwc.yaml --output /path/to/new_run
# TEST 只有显式评估才读取；训练只使用 TRAIN/VAL。
milplus evaluate --run /path/to/new_run

# 合成软件样例，默认 768 维；不表示存在临床 WSI。
milplus demo --sclwc --patients 30 --classes 3 --output /path/to/new_synthetic_data
```

命令名称以 `milplus --help` 为准；配置可显式传入不同超参数，实际请求、来源与输入哈希会保存。

## 保留的完整机制

1. **CDA**：作者 class-specific gated attention 和各类别独立线性 bag head；bag CE 权重 .8。
   对真实标签分支取 top-k 与 bottom-k，用该分支二类实例头做 CE，权重 .2。
2. **PNM**：使用作者 `Att_Head` 的 class-agnostic sigmoid gate。Eq.5 为
   `mean(gate*h)` 与 `mean((1-gate)*h)`，两者之和等于整包特征均值。共享 C 类线性头，
   前景目标为 slide 标签、背景目标为明确登记的参考类及其角色；两项 CE 之和权重 .2。
3. **WSCL**：Eq.7 的当前 top-k anchor 保留计算图；历史 key detach。L2 归一化、温度 .07，
   对同类别历史正 key 取均值，对当前 k 个 anchor 求和，分母包含正、负、硬负 key。
   参考类 bag 的 top-k 进入 hard-negative、bottom-k 进入 easy-negative；每个非参考类别拥有
   单独的 positive bank。不同非参考类别相互作为负样本，不折叠成一个“阳性”标签。
4. **完整状态**：模型、有效队列内容与身份、optimizer、cosine、选模/停止计数、epoch 和
   Python/NumPy/Torch/CUDA RNG 都进入恢复检查；TRAIN 以外或标签不符的 memory 状态拒绝加载。
   推理不修改 memory，不需要 VAL/TEST 标签来生成训练状态。

## 论文未完全公开的实现选择

这些选择属于有明确名字的论文参考重建，不能描述为获得了作者未公开的原始完整 trainer：

- Eq.4/6 的印刷减号与相邻 CE 文字存在歧义。采用规范 CE；PNM 两项 CE 的数值及 gate
  梯度另与作者 `utils/loss.py::loss_fg/loss_bg` 的实际函数体对照。CPU 只移除 `loss_bg`
  未使用默认参数的 CUDA 分配，调用始终提供明确 C 类 one-hot 目标。
- PNM 共享一个 C 类头，所有 slide 都计算这两项；参考类 slide 的两个目标均为所登记参考类。
  这是明确的头共享与作用域选择，不冒充公开 forward 已包含这一分支。
- 沿用源码 5 epoch warmup 和三种 key 容量 2048/16384/2048；按 k 个 key 的完整 bag 做 FIFO。
  论文参考使用所有有效历史 key，冷启动为空，正负配对不足时显式记为 inactive zero。
  不把 Gaussian 初始化当作训练切片，不采样尚未写入的槽位。
- 当前 anchor 与历史 key 的身份均有记录，排除所有相同 slide_id 的历史条目。允许同一患者的
  不同 TRAIN 切片作跨切片样本，与患者拆分一起记录；这不是独立患者数的增加。
- 当前网络的 anchor 保留梯度，历史 key 不反传；参数更新后将当前 forward 得到的 detached
  特征写入 bank。这一更新时点和历史特征的滞后性质明确保留。
- SSL Swin/MoCo 预训练属于编码器层。本配置提供完整下游聚合和训练，接受明确权重版本的
  冻结 FM 特征；不声称重跑了论文 48×V100 的编码器预训练。

## 完整训练配方

补充材料的 Adam lr=1e-3、weight decay=1e-5、k=8、单 bag、温度 .07、四个损失权重
原样保留。最大 100 epoch、cosine T_max=50/eta_min=1e-6、patience=20、epoch>30
的最早停止条件来自固定公开 example，因为补充材料没有完整说明这些外层参数。
Adam betas=(.9,.999)，不裁剪梯度、不采样缩短整包。

使用与作者 `BagDataset.get_balance_weight` 相同的 N/count 权重、replacement、
恰好 N_train 次抽样，`DataLoader(batch_size=None)`。示例 YAML 的 batch_size=32 和
train_set_size=400 并不改变作者 dataloader 的实际 bag 数。适配使用 num_workers=0，
将完整抽样与模型随机状态纳入恢复；不声称复刻 16 个 worker 的进程调度。

每 epoch 后在 VAL 上计算 slide macro AUROC，严格改善才保存，保留首次并列；先判断
`bad>=patience and epoch>30`，未停止时才推进 cosine。没有 TEST 调参或选模。
完整恢复包含 optimizer/scheduler/RNG，超出作者仅保存网络权重的基础状态文件。

源码对照使用 example 的 lr=3e-4、k=5、原生 Gaussian memories，保留小 bag 时 k=3 的
源码行为；仅支持显式参考类为索引 0 的二分类，超过原始 16384 patch bank 的 TRAIN bag 拒绝，
避免用静默修复改变对照。完整 D 输入只替换首个投影维度，保留 512/256 原生结构；
原始 epoch 函数仅将自动 CUDA 选择改为调用者明确设备。

## 验证状态

尚待全库、全部配置和独立安装包原生预算验收。专项测试覆盖实际原始 CDA 输出与梯度、
PNM 原始 helper 与独立 CE、NumPy Eq.7、真实 TRAIN bank 的填充/隔离/多类/FIFO、
实际作者 train_loop 的全部参数/optimizer/队列/RNG 对齐、完整流程逐位恢复与选模重载。
所有当前运行均为 CPU 合成输入；没有自然 BRCA/COAD、GPU 或论文数值复现结论。

## 候选修正记录

R1 曾把正常组织类误设为所有任务的强制条件。补充材料第8节的 MSI/NSCLC 证据表明
这一约束过窄，因此 R1 已撤回，没有正式纳入或复用其未完成的全库通过声明。R2 明确支持
参考终点类及必须给出的背景目标来源，新增完整训练和角色检查，重新冻结代码并验证。
