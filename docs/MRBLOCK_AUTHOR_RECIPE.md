# MR-Block 原始配方与统一入口

此候选私有加载作者仓库 `BearCleverProud/MR-Block` 的固定提交
`a71fb28262fdb9aa61265c86c3978080c07ac966`。作者 MIT LICENSE 已读取。
新增训练与恢复实现正在验证，不能解释为真实 WSI 实验或论文指标复现。

## 覆盖范围

20 个显式配置覆盖原始五个 MR 模型及五个对应基线：

| MR 模型 | 作者对应基线 |
|---|---|
| mr-abmil | abmil |
| mr-transmil | transmil |
| mr-dgrmil | dgrmil |
| mr-rrtmil | rrtmil |
| mr-cate | cate |

每个模型都保留原始 `--early_stopping` 开启和关闭两个配置。
这 20 套属于同一方法家族的实验条件，不增加独立方法数量。

```bash
milplus setup --method MR-Block
milplus run --data /data/my_task --method MR-Block --output /data/my_run
```

展示名 `MR-Block`、`MR-AB-MIL` 和 `MR-Block / MR-AB-MIL` 默认选择
`mrblock_author_200_mr_abmil_earlystop_locked_test`。
原有技术 ID `mrblock` 仍然对应已有受控适配，不改写它的定义。
可用 `milplus methods --runnable --json` 查看执行配置，`milplus methods --all --json`
查看全部 164 个接入条目。

## 保留的原始训练设置

- 最高 200 轮，batch size 1，无 patch 上限，不缩小原始内部宽度。
- AdamW，lr 5e-4，weight decay 1e-5，dropout .25，small model size。
- 原始 MR rank 64，冻结基础权重，保留可训练 bias 与原始初始化。
- LinearLR(start_factor=.01, end_factor=1, total_iters=200)。
  每个 optimizer 更新后执行一次，因此首个更新用 5e-6，200 次更新完成预热。
- 默认 RandomSampler；显式 `config.model.weighted_sample: true` 对应作者
  `--weighted_sample`，按原始类别反比权重有放回采样。
- 每轮 TRAIN 后 VAL；开启早停时按 VAL CE 选模，patience 20，epoch > 50
  才停止。相同 loss 也保存并重置计数，与原代码一致。
- 关闭早停时保留最后权重。不能用“选最优验证 AUC”替代这个条件。
- 原始 MR-DGRMIL 的预测类别中心更新、MR-CATE 的随机重参数化和 CE-only
  trainer 均保留。不会把原 trainer 丢弃的 IB 辅助项自行加入损失。

## 合并状态必须随 checkpoint 保存

原始 ManifoldResidualBlock 在 eval 时向基础矩阵加入 delta，在 train 时撤销；
`merged` 布尔值不在普通 state_dict 中。实际原始模型诊断证明，只把已经合并的
权重加载进新模型，再调用 eval，会重复合并，导致预测不同。

本实现同时保存每个模块的 training 模式和 merged 标志，加载权重后直接恢复
这些标志，再按原始模型流程切换模式。因此同进程选中权重和新进程重载可以
保持一致。模型、AdamW、scheduler、早停、原始采样顺序、全局 RNG、更新游标
一并保存，支持批次中、最后训练批次后及 epoch 后恢复。

原始残差活动路径包含 GELU，而作者的合并 delta 是线性矩阵乘积。这一原始
train/eval 差异按原样保留；本适配没有静默修复为另一种模型。

## MR-CATE 概念资产

CATE 原始内部投影为 512 维，本实现要求与真实概念对齐的 512D patch 特征，
并使用作者规定的概念文件名。任务数据包可增加：

```yaml
mrblock_concepts:
  path: assets/nsclc_concepts.pt
  task: nsclc
```

task 支持 `camelyon`、`nsclc`、`rcc`，分别映射为 `brca_concepts.pt`、
`nsclc_concepts.pt`、`rcc_concepts.pt`。概念文件必须实际存在，包含有限的
二维 512D 概念矩阵。文件哈希、路径、形状和 task 写入运行请求，恢复和
重新评价均核对。形状检查不能证明真实语义对齐，调用方仍须提供原始来源。
合成软件测试中的随机概念矩阵有显式 synthetic 标记，不可用于声称原始
概念资产验证或临床结果。

CATE 的原始 reparameterise 在 eval 时依然调用随机数；恢复选中 checkpoint
的 RNG，才能保持独立进程的重复评价一致。这里不静默改成取均值。

## 数据、运行环境与边界

- 患者互斥划分由数据包锁定；few-shot 必须由真实 split 清单提供，不能从
  文件夹或配置名里的 `4_shot` 字样推定已经满足。
- 原始 train_fold 构建并在训练结束时评价 TEST；本实现将 TEST 评价改为
  完整训练后的独立命令。TRAIN 与 VAL 需覆盖全部声明类别，以满足原始指标。
- CPU/CUDA 支持使用同一来源定义；workers 固定 0，便于记录与精确恢复。
- timm0.9.12 与 nystrom-attention0.0.14 是本基准明确选择的运行版本，
  不能称为作者指定版本。其他互斥方法使用各自环境。
- 当前验证输入是合成数据；完整方法准入还需全部原始预算、源码与独立 wheel、
  全库回归、真实资产和真实参考实验。
