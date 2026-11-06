# MIL_BASELINE_PULS

这是在 0.131.0 基础上继续完善的 0.132.0 代码候选仓库，本轮补齐 HIPT 混合扫描倍率的逐切片物理步长契约，不改变模型。164 条方法记录均已有代码及运行入口，共 415 套配置；此前已新增 DyKo 8 套、DSPAMIL 2 套和 FEATHER 9 套。DSPAMIL 仅支持 DSPA-MIL 2025 显式参考实现，**DSPA-U-MIL 仍未实现**，合并的目录条目不代表其所有变体完成。注册不等于完整训练、真实数据验证或论文复现；各方法仍需自己的输入、原始源码、权重和独立依赖。

本目录是独立的 Git 工作目录，不覆盖原有的 0.62.0 主仓库。它通过 `legacy` remote 指向原主仓库；当前开发快照并非从该仓库的 Git 历史直接导出，因此不能声称两者有完整连续的提交历史。`CURRENT_VALIDATION.json` 和 `DEVELOPMENT_STATUS.json` 是快照携带的历史材料，不代表本目录已完成整库验收。

本轮代码接入与检查记录见 [USABLE_CODE_STATUS.json](USABLE_CODE_STATUS.json)，与历史全预算验证记录分开保存。

在本机，`.venv` 已安装本仓库并复用原项目环境中的基础依赖，入口为 `.venv/bin/milplus`；因此该环境依赖原环境路径，复制仓库后应重新安装依赖。迁移到其他机器时请新建独立 Python 3.10 或更高版本环境，再执行 `python -m pip install -e .`。各方法可能需要各自的可选依赖，不能假设全部依赖彼此兼容。

## 本机快速开始

在本仓库目录执行：

```bash
.venv/bin/milplus coverage --json
.venv/bin/milplus inspect --method ABMIL
.venv/bin/milplus demo --output /path/to/new-demo
.venv/bin/milplus check --data /path/to/new-demo --method abmil --deep
.venv/bin/milplus run --data /path/to/new-demo --method abmil --epochs 2 --device cpu --output /path/to/new-run
.venv/bin/milplus evaluate --run /path/to/new-run --device cpu
```

以上演示使用合成数据，输出目录必须是不存在的新路径。真实数据请按下方数据包协议准备，不要把演示结果当作临床或论文结果。

## 首次准备源码

在联网CPU环境中按方法准备固定版本源码；准备计划可先只读查看：

```bash
milplus setup --method ProtoMIL --plan
milplus setup --method ProtoMIL
```

该入口会一并准备ProtoMIL的原始SAE依赖、KMP-MIL的CONCH源码等固定依赖。MeanMIL和MaxMIL使用包内实现。运行环境与实际数据按方法文档准备，见[源码准备](docs/SOURCE_PREPARATION.md)。

## 运行

在独立Python环境安装，不把所有方法的互斥依赖合装到一个环境：

```bash
python -m pip install -e .
milplus methods --json
milplus inspect --method ABMIL
milplus check --data /path/to/task --method abmil --deep
milplus run --data /path/to/task --method abmil
```

`--data`指向已准备好的任务数据包，其中包括标签、患者与切片映射、固定划分、FM特征索引和默认配置；原始WSI文件夹本身不能确定这些信息。依赖原图、多尺度、图、组学、文本或特殊训练阶段的方法，必须提供其完整输入和独立环境，见各方法文档。

```bash
milplus run --data /path/to/task --method clam_sb --fm uni_v1 \
  --split fold0 --seed 2026 --device cuda --config configs/clam.yaml \
  --output /path/to/new-run
milplus evaluate --run /path/to/new-run --device cuda
```

训练按各方法明确登记的配方更新与选模，锁定测试通过明确的评价入口执行。原作者配方使用TRAIN选模的情形会单独命名并披露；PNEA-MIL默认显示名称选择VAL修正版。新运行拒绝覆盖原结果目录。演示数据始终标记为合成数据。

## 数据和方法

- [数据包协议](docs/DATA.md)
- [BRCA、COAD、TRIDENT及H集群流程](docs/H_CLUSTER.md)
- [方法与适配差异](docs/METHODS.md)
- [完整方法登记](src/milplus/resources/methods.json)
- [当前验证状态](CURRENT_VALIDATION.json)
- [PNEA-MIL完整配方与修正](docs/PNEAMIL_AUTHOR_RECIPE.md)
- [AttriMIL完整配方与修正](docs/ATTRIMIL_AUTHOR_RECIPE.md)
- [CAR-MIL完整配方与修正](docs/CARMIL_AUTHOR_RECIPE.md)
- [DSPA参考与缺项](docs/DSPAMIL.md)
- [最后三个方法的准备与统一运行入口](docs/LAST_THREE_METHODS.md)
- [真实BRCA第一轮：UNI × IDC/ILC](docs/BRCA_UNI_IDC_ILC_20261105.md)
- [ProbSA实际CUDA稀疏乘法对齐修复与证据](docs/PROBSA_CUDA_ALIGNMENT.md)
- [DyKo原始输入与TITAN要求](docs/dyko_native_candidate.md)
- [FEATHER原始权重与三种协议](docs/FEATHER_NATIVE_CANDIDATE.md)
- [许可证和第三方声明](THIRD_PARTY_NOTICES.md)

BRCA当前候选任务为IDC／ILC；COAD现有形态标签不自动等同于MSI标签。临床标签、患者身份、划分与WSI的MPP／QC均须核验后再进入正式实验。

完整冻结首页及历次实现记录保存在[历史快照](docs/frozen_0_117_root_metadata/README.md)。当前候选保留完整源码准备流程，并修正11个方法完整名称的既有配置选择，见[方法名称说明](docs/CATALOG_NAME_ROUTING.md)。父候选的QPMIL数值设置与CPU数学初始化、原失败证据和默认稳定／开发入口均保留。完整方法验证仍待完成。

QPMIL的原始CPU逐值失败、实际梯度算子复现及修复范围见[数值修复说明](docs/QPMIL_CPU_RECOVERY_FIX.md)。0.118.0 的新增4项回归与原完整协议合计30项曾在源码和独立安装包通过；本候选需重新验证。当前其它验收状态见[本轮记录](docs/validation_history/20261029_CPU_contexts_and_unresolved_sparse.md)。

CPU数学库初始化候选及当前验证边界见[说明](docs/CPU_VECTOR_MATH_INIT.md)。
