# KMG-Mamba 固定源码审计

审计对象：`dmhdmhdmh/KMG-Mamba@4b5075bb2d3eddf46f9bc4f46175d250b134e27c`，仅描述该提交的实际可执行路径。

| 文件或环节 | 观察与适配 |
|---|---|
| `models/model_MG_Mamba.py` | 使用完整共享投影、WiKG、原型交互、CrossMamba、注意力池化及三个目标；仅替换硬编码 `.cuda()` 为输入设备，返回额外诊断值 |
| `models/datten.py` | 注意力分支固定2；原型分类器及 SVM 也固定2。原配置明确限二分类，多分类修正单独命名 |
| `models/vmamba.py` | 保留4状态的跨尺度参数交换和实际扫描调用；默认精确张量递推，独立对照源码 `selective_scan_ref` |
| DropPath | 固定 timm0.4.12 的 uniform-plus-floor 随机实现，避免新版 Bernoulli 实现改变训练随机序列；保留外部参考文件及 Apache-2.0 许可证 |
| 未调用参数 | 原代码创建但不调用 depthwiseConv2d、MLP/norm2 和 CrossMamba.norm；保留参数和初始化，不人为添加梯度路径 |
| `utils/core_utils.py`、`main.py` | 实际默认100轮、Adam2e-4/1e-5、随机不放回切片采样；创建但不调用 Plateau scheduler；验证 AUROC，最新并列最优，patience20/stop_epoch50 |
| 序列重复 | 默认阈值0；非零阈值时保留作者规则，同时验证重复后两尺度都满足9邻居要求。不会静默截断完整 bag |
| 数据对齐 | 作者活跃路径无需逐点父子矩阵；适配器保留两个实测尺度、不等长序列和原始行序 |
| 许可证 | 固定仓库未声明整体许可；作者代码保留在用户本地固定源码目录，不随 PULS 分发 |

初步专项检查中，三类合成任务误设24名患者，被既有最少每类9人规则正确拒绝。已改为30名，保留原失败记录并重新执行检查，没有放宽数据校验。

此前全库检查停在 `test_full_matrix_retains_unimplemented_and_incompatible`：旧测试使用 ZoomMIL 未接入时的 family 名查询已解析的执行 profile。仅在本新候选修正为按 family 查询，同时核验完整164条范围、所有未接入条目的阻断原因和 ZoomMIL 真实多尺度输入要求；旧冻结树和失败证据保持不变。

这些源码与软件验证结论不等同于历史论文结果错误，也不等同于真实 WSI 数值复现。
