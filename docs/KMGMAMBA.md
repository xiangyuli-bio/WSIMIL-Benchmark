# KMG-Mamba

候选实现已写入，正式准入仍待专项、全库、逐配置及完整预算安装包检查。真实 WSI 训练和论文指标复现尚未执行。

来源是 [作者仓库](https://github.com/dmhdmhdmh/KMG-Mamba)，固定提交 `4b5075bb2d3eddf46f9bc4f46175d250b134e27c`。通过已验证的本地 Git 源码加载原类；本项目不打包未声明整体许可证的作者实现。

## 两个执行配置

| 配置 | 支持范围 |
|---|---|
| `kmgmamba_source`，别名 `KMG-Mamba` | 源码支持的二分类，原始两个注意力分支和原型分类头 |
| `kmgmamba_multiclass_repaired` | 三类及以上；明确将原代码固定为2的注意力分支、原型头及 SVM 类别数改为实际类别数 |

两个配置都使用原始96维投影、9邻居特征图、最高度原型及8头原型交互、4状态跨尺度扫描、128维注意力和切片分类器。没有以简化聚合器替代。输入 FM 维度可不同，必须与登记的完整特征一致。

目标为 `0.5*bagCE + 0.5*(coarsePrototypeSVM + finePrototypeSVM)`。默认 Adam 学习率0.0002、weight decay0.00001，逐切片完整双尺度 bag，随机不放回采样；最多100轮，验证 AUROC 的最新并列最优权重，耐心20且零起始轮号大于50才早停。原代码虽然创建 Plateau scheduler，却没有调用 step，本路线保留恒定学习率。模型、全部 Adam 状态、采样顺序、选模历史和随机状态均参与断点恢复。

## 数据

在标准任务包之外，提供同一张 WSI 的两个真实特征尺度。粗尺度 patch 的物理边长是细尺度的两倍，倍率关系相反。每个尺度都需要全维特征与同序的 level-0 左上角坐标；需要写明权重、预处理及尺度来源。原模型支持不等长序列，不要求伪造一一对应的父子映射。

索引 CSV 使用 `slide_id,path`；每行 HDF5 包含 schema 指定的两组特征和坐标。主任务特征必须与粗尺度逐行一致。登记器检查全部切片、患者、划分、标签、输入哈希、维度、数值与坐标边界。使用 `milplus prepare-kmgmamba --help` 查看登记参数，然后在选定 encoder 下填写 `kmgmamba: {manifest: 路径}`。

```bash
milplus fetch-source --help
milplus prepare-kmgmamba --data /path/to/task --index kmgmamba_index.csv \
  --schema dual_scale_schema.json --output kmgmamba/manifest.json
milplus check --data /path/to/task --method KMG-Mamba --deep
milplus run --data /path/to/task --method KMG-Mamba --output /path/to/new_run
milplus evaluate --run /path/to/new_run
```

`milplus demo --kmgmamba` 只生成明确标记的合成特征，用于软件验证。三类演示至少27名患者，不能作为正式癌症数据实验。

## 扫描后端与验证边界

默认 `scan_backend: torch` 使用保留跨块状态的精确选择性扫描递推，可在 CPU 或 CUDA 上运行，不是替换为通用 Mamba。`scan_backend: cuda` 是显式可选的已安装扩展入口，缺少 CUDA 或算子时直接失败；该优化扩展尚未完成本方法的数值与完整训练验收。后端和依赖身份写入记录。

专项检查对照作者完整网络、两尺度梯度、实际训练循环及所有 Adam 状态；独立参考使用作者 selective_scan_ref 与固定 timm DropPath。另检查最新并列选模、精确续训、输入篡改、CLI 和多分类修正。完整安装包验证要求1024/1536维、二/三分类四组原生100轮上限及源早停规则，软件检查的短训练不替代这一步。详见 [源码审计](KMGMAMBA_AUDIT.md)。
