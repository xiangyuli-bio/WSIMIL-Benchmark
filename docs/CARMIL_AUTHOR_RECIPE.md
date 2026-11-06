# CAR-MIL 原作者完整流程与明确修正

固定来源为 `ImaneCR/CAR-MIL`，提交
`26fdee4dc05e6b358395e28fb2073da3b29b6cc3`。
模型、损失和验证定义从已检查的 Git 对象加载，第三方源码不复制到本包。
固定树没有根目录 LICENSE；来源与许可审查分别记录，不据此推断授权范围。

展示名称 `CAR-MIL`、`CAR_MIL`、`CARMIL` 选择
`carmil_author_100_attmil_l1_locked_test_repaired`。
原有技术 ID `carmil` 保留原 MIL_BASELINE 受控移植，结果不能改称原作者配方。

## 八套完整配置

每种骨干分别保留 L1、COS 两种原始损失定义，共八套：

| 骨干 ID | 原始模型 | 配置前缀 |
| --- | --- | --- |
| attmil | 非门控 DAttention_CF | `carmil_author_100_attmil_` |
| gattmil | AttentionGated_CF | `carmil_author_100_gattmil_` |
| transmil | TransMIL_CF，原始 Nyström 实现 | `carmil_author_100_transmil_` |
| dsmil | MILNet_CF、BClassifier_CF | `carmil_author_100_dsmil_` |

前缀后接 `l1_locked_test_repaired` 或 `cos_locked_test_repaired`。
默认 main 为 attmil/L1；README 示例采用 gattmil/COS，并显式传入可选的
baseline checkpoint。两者分别登记，不把示例误解成默认配置。

## 保留的训练条件

- 原始输入投影和512隐藏维度，注意力维度128；Transformer 保留原始
  三个 TransLayer、8 heads、256 landmarks、6次伪逆迭代。
- 完整100轮、batch1、完整 bags、Adam lr2e-4、wd1e-5。
- CosineAnnealingLR T_max100、eta_min0，每个完整训练 epoch 后更新。
- CE、effect CE 与原始 L1/COS attention 项，effect/attention 系数各0.2。
- DSMIL 保留0.5 bag loss +0.5 instance loss，其余骨干为原始 bag loss。
- 不增设 patch cap、梯度裁剪、AMP、默认初始化 checkpoint 或隐藏维度缩小。
- 原始 camelyon16 早停器 patience30、stop_epoch130；其停止条件不能在
  本配置的100轮预算内触发。保留原始计数与状态。
- TRAIN 使用有放回关闭的 RandomSampler；私有 generator 初始 seed 为
  7784414403328510413，同时供 DataLoader base seed 与 sampler 使用。
- 保留原始模型初始化前 seed，以及构建模型/优化器后的训练 reseed。
  统一 CLI 显式 seed 替代作者默认2021，选择值写入请求。

## 训练前选模

原始 one_fold 在每轮训练前调用 val_loop，采用 VAL AUROC 严格改善选模，
初始 opt_auc=0；第一轮评价的是尚未训练的模型。
100次训练分别对应 epoch0到99的训练前 selector。
最后一次训练后不额外添加 selector，不用 CE 替换 AUC。
如果严格选择器始终没有产生权重，运行明确失败，不偷偷选择最后模型。

最佳权重、Adam、scheduler、早停器、RNG 和私有 generator 一起保存，
最佳快照不会随后续训练变化。恢复覆盖训练前 selector、实际 minibatch、
末批 scheduler 前及完整 epoch 边界。
RandomSampler 在最后一个 batch 后还会消耗一次空切片 permutation；
恢复流程保留该次随机消耗，不提前改变私有随机状态。

## 原始实现的边界

TransMIL_CF 的原始 A_ori 将两条一维 attention 用 cat 拼成2N向量，
原 trainer 的 `A_all[0]`/`A_all[1]` 因而取到标量。这里明确保留该行为；
没有用 stack 悄悄修正它，也不把其 attention 结果解释成正常的两行图。
若后续增加形状修正版，应单独命名和评价。

原验证函数按二分类编写，因此这些配置要求二分类、单切片 bag、TRAIN/VAL
各有两个类别。DSMIL 原始 squeeze 路径要求每包至少两个实例。
其他输入条件应进入另行命名的适配版本。

## 明确修正与协议变化

1. 原始 EarlyStopping 的 np.Inf 在 NumPy2 下报错，改为 np.inf，
   保留原报错证据。
2. 原 train_loop 无条件调用 CUDA 同步和计时事件，CPU 会报错；
   CPU 路径仅省去计时，不更改模型、损失、梯度或 scheduler。
3. 固定患者不重叠的 TRAIN/VAL/TEST 替换内置 CV；
   原默认 val_ratio0 将 TEST 当作 VAL。这里明确锁定 TEST，
   训练只构建 TRAIN 和 VAL 数据集。
4. num_workers0 替代原始2，用于精确记录和恢复加载顺序。
5. 原仓库没有固定 timm 版本；本基准明确选择 timm0.9.12，
   这是运行时选择，不是原论文环境证据。

原始失败与修正后的软件证据分别保存。并未据此断言原论文结果错误。

## 可选原始初始化

可在 model.initial_checkpoint 中传入实际 baseline checkpoint 文件，
保留原始 `checkpoint["model"]`、`strict=False` 加载方式。
文件路径、SHA256、缺失键和额外键写入运行记录。恢复与评价拒绝
已改变的初始化文件；不自行生成或替代用户期望的原始权重。

```yaml
model:
  initial_checkpoint: /path/to/actual/baseline.pt
```

```bash
python -m pip install '.[carmil]'
milplus setup --method CAR-MIL
milplus run --data /path/to/prepared_task --method CAR-MIL \
  --output /path/to/new_run --device cpu
milplus evaluate --run /path/to/new_run --subset test --device cpu
```

统一评价从最佳快照重新载入模型，导出 factual branch 的原生概率。
作者阈值优化指标用于原选择器对照；统一评价不在 TEST 上优化分类阈值。

## 验证范围

机制检查以完整原始模型文件、原始 CustomLoss 和 train_loop 为独立参考，
覆盖四骨干×两损失×1024/512维、长bag输出、梯度、Adam、scheduler与RNG。
完整轨迹以原始 C16Dataset/HDF5、train_loop、val_loop 和私有 generator
为参考，保留100轮和训练前选择顺序。参考 train_loop 仅省去 CUDA 计时；
参考 EarlyStopping 仅修正 np.Inf。

软件 fixture 都是明确标记的合成特征，含人为类别信号。
具体通过数量由保留日志和版本证书证明；收集数量不等于已执行数量。
这些检查不能当作真实 WSI、GPU验证或原论文指标复现。

## 真实 BRCA UNI × IDC/ILC 专项完成项

2026-11-05T21:17Z，`carmil_repair2` 的原训练进程自然完成全部100轮、
74200个TRAIN minibatch和100次原训练前selector，未因调度交接重启或削减预算。
其训练状态明确 `test_evaluated:false`；随后独立评价进程612945/start833746968
重新加载锁定选模结果，完成45位既有TEST患者的评价，未用TEST选择阈值或模型。

21:18:38Z，独立患者身份/标签/概率和指标复算通过，当前全部完成项审计为57通过、0错误。
本项TEST AUROC为0.9913793103448275，accuracy44/45，balanced accuracy0.96875，
macro-F1为0.9753963914707491。这里只是当前固定划分45人的单次结果，不是论文分数复现或显著性结论。
预测SHA256 `7c33189954505042a4565ffc8c314629c2cae9bfd43ff0d42d0d057597dc24c3`，
指标文件SHA256 `b02be38871c0e7f41aeff8011485b87cf395294dbf9f1616191c57d5b9b2c79e`。
证据在BRCA专项共享目录 `results/carmil/`、`completed_predictions_audit.json`，
及 `progress_muse_uni1024_admitted_20261106.json` 的后续实际完成记录。
