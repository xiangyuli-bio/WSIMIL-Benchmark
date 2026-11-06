# FC-MIL 原始机制与显式 benchmark 配方

原始来源为 `7FFDW/FCMIL`，固定提交
`315ac88808d7517429ca0b55073e75dce03ddfd9`。实际读取的许可证为
Apache-2.0。运行时私有加载固定 Git 对象，不分发作者模型源码。

本次接入保留原始 FCMIL、FAA、DAB 和因果损失函数体。输入为单张切片的
1024 维冻结特征，内部宽度为 512；完整 patch 顺序用于 FFT，不加隐式
patch 数量上限。FFT 四分频要求至少 4 个 patch。干预依据为频率分支返回的
mean-query FAA attention。

原始训练目标为：

```text
CE + alpha_cr * (-lam * KL_drop + (1-lam) * KL_replace)
```

原始预测概率作为 detached soft target。保留原始置零、替换、top-k 和
额外前向过程；alpha 为 0 的对照也执行原始 trainer 的 counterfactual 调用。

## 运行入口与命名

```bash
milplus setup --method FC-MIL
milplus run --data /path/to/task --method FC-MIL --output /path/to/new-run
```

两套新增配置：

- `fcmil_original_benchmark100_cr_auc_repaired`：完整原始机制，alpha_cr=0.1；
  展示名 `FC-MIL` 选择此配置。
- `fcmil_original_benchmark100_ce_control_auc_repaired`：alpha_cr=0 的明确消融。

技术 ID `fcmil` 保留已有 MIL_BASELINE 受控移植的含义，不能将其当作新增
原始机制配置的同义词。

## 作者源码能够证明的设置

保留原始 Adam、weight decay=1e-5、ReduceLROnPlateau factor=0.1、
patience=5；调度发生于验证和选模之后，监测 VAL CE。
保留原始 strict AUC improvement、best_auc 初始值 0。
源码 fallback 值为 topk_ratio=0.03、lam=0.5、alpha_cr=0.1。

## 缺失配置与修复

固定树没有训练配置文件，因此 **100 轮、学习率 2e-4、batch size=1、
workers=0 和默认 seed=2026 是本库显式选择的 benchmark 设置**。
这些设置不代表已经恢复原论文或作者默认训练配方。两套配置均完成全部
100 轮，不使用缩短预算代替验证。

原始源码中实际观察到的失败与对应处理：

- `BaseDataset` 在初始化 logger 之前调用会使用 logger 的解析函数；
  独立参考子类只提前初始化日志字段，保留原始解析、读取与标签映射。
- manager 向不接受 `writer` 的 `_run_epoch` 传入该关键字；
  参考调用只接收并忽略日志参数，执行原始 epoch 函数体。
- 原始 epoch 不返回 AUC，strict selector 因而无法保存最佳权重；
  从实际 VAL logits 与标签计算 macro AUROC。
- 原始共享 `./checkpoints`、`./logs` 清理过程替换为用户指定的独立输出目录。
- 调用者设备、患者不重叠数据包、锁定 TEST、保存与恢复状态属于本库适配。
- 入口固定 deterministic algorithms=True、float32 matmul=highest、
  CUDA TF32=False、cuDNN TF32=True、cuDNN deterministic=True／benchmark=False。
  这是明确的 benchmark 数值策略，避免继承调用者或其它方法的进程全局状态；
  不是声称作者原始 trainer 设置过这些全局选项。

首次失败日志保存在开发工作目录中，后续成功不能抹去原始失败。

## 验证范围

使用明确标记的合成数据，比较原始 manager 的完整 100 轮 `train()` 与本库
适配器的参数、Adam、scheduler、最佳选择、每步 loss/logits、bag 顺序与 RNG。
还需验证三个恢复边界、公开 CLI、重复最佳权重评价、独立安装包和父配置保留。

当前结果以根目录 `CURRENT_VALIDATION.json` 和工作目录审计证据为准。
合成运行及精确逐值比较不构成真实 WSI、GPU 或论文指标复现。
