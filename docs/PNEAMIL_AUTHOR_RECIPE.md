# PNEA-MIL：完整原作者训练配方候选

本实现使用 Chen-Jxiang/PNEA-MIL 的固定提交
`ef204ff5179ce7ea4bb7d04744f1553bee2f039c`，从 Git blob 加载原始
`PNEA_MIL`、`train_one_epoch` 和 `evaluate_loss` 定义。模型与原始损失没有文本替换。
它独立于既有 MIL_BASELINE 移植配置 `pneamil`，仍处于验证阶段。

## 运行入口

先在联网 CPU 环境准备原作者源码，再提供已经登记患者、标签、划分及特征的任务包：

```bash
milplus setup --method PNEA-MIL --plan
milplus setup --method PNEA-MIL
milplus check --data /path/to/task --method PNEA-MIL --deep
milplus run --data /path/to/task --method PNEA-MIL --output /path/to/new-run
milplus evaluate --run /path/to/new-run
```

完整显示名称 `PNEA-MIL` 选择
`pneamil_author_val_selector_repaired`。显式技术配置 `pneamil`
继续调用原有受控移植版本，不将其重命名为原作者配方。

两条新增配置：

| 配置 | 选模人群 | 明确修正 |
| --- | --- | --- |
| `pneamil_author_train_selector_snapshot_repaired` | 原作者实际传入的 TRAIN loader，含重新抽样 | 固定患者不重叠划分、独立保存所选权重 |
| `pneamil_author_val_selector_repaired` | 完整 VAL bags，不做 patch 抽样 | 上述修正，另将原始 TRAIN 选模替换为 VAL 选模 |

原脚本将 TRAIN-loader 评价结果命名为 `validation_loss`，两者不等价。
原始 `best_state` 直接保留 `model.state_dict()` 引用，后续训练可能改变所选状态；
这里在选模时立即深拷贝并原子序列化，两个配置都在名称与运行记录中披露修正。

## 原始训练条件

- 原始默认输入为 1024 维 UNI 特征；按原构造器支持的 `D_in` 接受其他 FM 维度。
- 每类别 64 条 evidence axes；类别数来自已固定的任务标签。
- 原模型 `return_loss`：交叉熵加 `5 × mean(ReLU(W))`。
- 原模型创建的 Adam：学习率 `1e-4`，weight decay 为 0；保留权重与 bias 的参数分组。
- batch size 为 8，末尾不足 8 个 slide 的批次保留。
- 每次访问 TRAIN slide，通过原始 `torch.randint` 有放回抽取 1024 个 patch。
- 最多 1000 轮，patience 为 10；只有相对既有最佳损失至少改善 `0.0005` 才更新所选模型。
- 不添加 scheduler、梯度裁剪、额外预训练或通用 trainer 的参数。

原作者脚本种子默认 0；统一 CLI 种子默认 2026，原始模型接收请求中实际记录的种子。
需要原种子时显式传入 `--seed 0`。任务划分来自数据包，不重新执行原脚本的随机 slide 划分。
这些差异不作为未披露的论文复现。

目前只支持 slide 作为训练单位的分类任务。多个切片属于同一患者时必须位于同一分区；
评价按任务包已声明的 slide 或 patient 规则执行。VAL、TEST 以全部原始 patch 推理，
不会套用 TRAIN 的 1024 个 patch 采样上限。完整输入检查会读取各分区特征以验证维度、
有限值和内容哈希，这不代表这些样本参与梯度更新或选模。

## 中断与恢复

每个实际完成的 TRAIN minibatch 和 epoch 选模边界保存：
模型、Adam 状态、Python/NumPy/Torch RNG、实际 RandomSampler 顺序、
已消费 slide 数量、加权损失累积量、patch 抽样轨迹和选模历史。
恢复现有 epoch 时抵消重建 DataLoader iterator 额外消耗的随机种子，
保留原始训练后续抽样。

```bash
milplus run --data /path/to/task --method PNEA-MIL \
  --output /path/to/new-run --stop-after 1
milplus run --data /path/to/task --method PNEA-MIL \
  --output /path/to/new-run --resume
```

`--stop-after` 是在实际轮次边界中断，不改变 1000 轮总预算。
更细的批次边界使用 `model.stop_after_train_batches` 运行配置，
它只控制本次中断，不进入不可变的算法请求。`epochs` 只有显式等于 1000 时才被接受；
改变抽样、优化器、evidence axes 或删减轮次的配置会被拒绝。

恢复及评价绑定输入清单、完整特征字节哈希、实际软件内容、作者提交和配方。
存活的 PID/starttime 锁禁止重复启动同一输出。评价只接受已完成的训练，
在新建原始模型中加载独立保存的最佳权重，再导出逐切片预测及患者感知指标。

## 验证边界与许可

新增测试比较独立执行的原始完整文件与适配器：原始 1024 维及不同 FM 宽度、
完整 64 axes、1024 patch、梯度和 Adam 更新；另检查原始 DataLoader/RandomSampler
的实际随机数轨迹。两条配置的完整预算训练与原作者函数逐轮比较，
包括 TRAIN/VAL 选模差异、所选快照、批次中断恢复和新进程加载。
实际执行结果另记入 `CURRENT_VALIDATION.json`，测试代码存在本身不证明测试通过。

冻结父版本 0.121.0 的 4597 项源码和 70 项独立安装包测试属于父版本，
不能计为 0.122.0 的通过结果。当前没有真实 WSI 实验或论文指标复现，
也没有完成全部 164 条方法的正式准入。

原作者根目录提供 GPL version 3 许可文本。新增原作者路径通过固定外部源码加载，
不会将它描述为既有 MIL_BASELINE 代码的 MIT 许可。来源文档存在和模型运行通过
都不构成对整个方法库的统一许可结论。
