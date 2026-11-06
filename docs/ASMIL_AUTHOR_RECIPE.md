# ASMIL 原作者训练流程与明确修正

本候选增加四套完整训练配置，原有技术 ID `asmil` 的受控移植配置保持不变。
展示名 `ASMIL` / `AS-MIL` 选择
`asmil_author_50_literal_nan_teacher_locked_test`。

| 配置 | 轮数 | 教师更新 |
| --- | ---: | --- |
| `asmil_author_50_literal_nan_teacher_locked_test` | 50 | 固定原代码的 NaN-only 条件 |
| `asmil_author_50_finite_ema_repaired_locked_test` | 50 | 单独命名的有限值 EMA 修正 |
| `asmil_author_100_literal_nan_teacher_locked_test` | 100 | 固定原代码的 NaN-only 条件 |
| `asmil_author_100_finite_ema_repaired_locked_test` | 100 | 单独命名的有限值 EMA 修正 |

来源为 `Linfeng-Ye/ASMIL`，提交
`5d062271e1626b27a955c309d10add902311865c`。
50 轮来自默认 `config/camelyon_config.yml`；100 轮来自显式
`config/camelyon_medical_ssl_config.yml`。Lung shell 示例引用了不同脚本名；
本实现没有把这些示例解释为当前主程序的完整论文配方。

`asmil_source.py` 从经检查的 Git 对象中执行原始
`Classifier_1fc`、`residual_block`、`DimReduction`、`sigmax`、
`MutiHeadAttention`、`ACMIL_MYMHA` 和 `ACMIL_MYMHA_EMA` 定义，
没有改写其函数体。跳过的导入仅供未调用的架构或外部日志使用。
第三方代码仍在单独获取的原始仓库中，不随本包重新分发。

## 保留的原始训练条件

- 在线模型和教师依次独立初始化，原程序没有复制在线权重到教师。
- 教师在训练阶段保持 train 模式，其 dropout 消耗的 RNG 也需要恢复。
- 输入维度映射为 384→128、512→256、768→384、1024→512、1536→768；
  attention 保持 8 heads、dropout0.1、LayerNorm eps1e-6。
- batch1，完整 slide bags，训练打乱顺序、drop_last=True。没有 patch 数上限。
- CE 加教师 sigmoid-normalized attention 蒸馏；原代码中的两个辅助项为零。
- AdamW 初始化 lr0.001、wd1e-5。每个实际 minibatch 使用作者函数把学习率
  改为 lr1e-4 起点的半周期 cosine；warmup0、min_lr0、clip_grad_norm5。
- 完整 50 或 100 轮，没有早停。`--stop-after` 只是保留检查点的中断。
- 原始 `n_token=1`、`n_drop=0`、`temp=1` 和 lr 可按原 CLI 的含义显式选择；
  不允许缩小内部维度、训练预算或偷偷增加 bag 采样。

## 教师规则的两个版本

原始训练循环仅当 `torch.isnan(msd[k]).any()` 时更新教师，并且发生于
当前 minibatch 的教师前向之后、在线 optimizer.step 之前。因此有限参数下，
教师保留独立随机初始化；不能把原始代码自动解释为正常 EMA。

修正版只把该条件改为 `torch.isfinite(msd[k]).all()`。它仍保留独立初始化、
train 模式和原始更新时机。若需要初始化复制或 optimizer 之后更新，应另行命名，
不能把它们归入这里的修正版。

## 选模与评估边界

实际原代码调用 `torchmetrics.F1Score(task="multiclass")`，
其参考运行时 1.3.0.post0 默认 average 为 **micro**；
AUROC 默认 average 为 **macro**。使用验证集两者之和严格改善选模，
初始最佳得分为0。这里没有用 macro-F1 替代原始选择器。

原始仓库没有固定 torchmetrics 版本。统一库明确采用
`torchmetrics==1.3.0.post0`，并登记为运行时参考选择，而非原论文环境证据。
依赖通过 `pip install '.[asmil]'` 准备。

以下变更明确记录为基准适配：

1. 固定且患者不重叠的任务清单替代源程序内置 slide 划分与 HDF5 路径。
2. 显式设置统一 seed；原 main 中 `set_seed` 被注释，不能宣称重现原始随机序列。
3. 锁定测试集。源程序每轮访问 TEST，本配置仅访问 TRAIN 和 VAL。
   多 token 模型在 eval 也调用 randperm，因此隔离 TEST 会改变后续 RNG；
   参考轨迹也按明确的 locked-test 协议执行。
4. num_workers0，用于完整、可恢复的同进程加载；源配置为8。
5. 保存在线模型、教师、optimizer、RNG、shuffle 顺序、当前游标和历史，
   支持实际 minibatch 及选模边界恢复，最佳权重为不可变快照。

评估只有在完整训练结束后才允许执行，并重新绑定全部特征字节、患者划分、
源提交、软件版本及配置，再从最佳权重和其 RNG 恢复。统一患者指标与
原作者 slide 指标分别输出。

## 使用

```bash
milplus setup --method asmil
milplus run --data /path/to/prepared_task \
  --method ASMIL --output /path/to/new_run
milplus evaluate --run /path/to/new_run --subset test
```

不同输入 FM 必须使用上面原始支持的维度；本配置不自动投影或裁剪特征。
更换到支持维度只表明输入接口兼容，不证明该 FM 的原论文结果已复现。

## 验证范围

测试将完整原始模型文件与原始 train_one_epoch/evaluate 定义作为独立参考，
检查五种原始维度、长 bag、多 token、梯度、教师、AdamW、RNG，以及
完整 50／100 轮轨迹、最佳快照、minibatch/末批/epoch 恢复和新进程评估。
有限值 EMA 参考仅修改原始教师条件一处，原始条件参考不改训练定义。

所有软件 fixture 都是合成特征。运行这些检查不等于真实 WSI 实验或
原论文数值复现。版本级完整检查和原始资产审查仍单独验收。
