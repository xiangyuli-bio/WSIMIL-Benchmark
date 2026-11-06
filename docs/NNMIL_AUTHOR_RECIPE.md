# nnMIL 原始分类流程与 TRAIN-only 规划修复

固定原始提交：`e92e8a0cf72dc1beb2a5915e27ca128af45fcb40`。
原始代码私有加载，不随本库重新分发。当前固定树未观察到根 LICENSE，
不能以论文开放获取许可替代代码许可。

此版本增加四套分类配置，分别按原始 AUC、BACC、F1、Kappa 选择权重。
展示名 `nnMIL` 默认使用 BACC；显式技术 ID `nnmil` 保持先前受控实现的含义。

```bash
milplus setup --method nnMIL
milplus run --data /data/my_task --method nnMIL --output /data/my_run
milplus run --data /data/my_task \
  --method nnmil_author_100_auc_train_planned_locked_test --output /data/my_auc_run
```

## 原始机制

- 原始 SimpleMIL、每次训练的 randperm、门控注意力与 dropout 保持原定义。
  输入维度不大于隐藏维度时仍消耗 randperm，不能省略这次随机调用。
- 验证、测试使用完整 bag 和原始重叠通道分块集成。方差、标准差使用原始
  ddof=1，只有一个块时辅助不确定性可为 NaN；最终概率仍须有限。
- 隐藏维度 max(256,D//4)，原始自适应 batch 16–48，最高100轮，
  patience10，严格改善才更新权重；相同指标增加早停计数。
- AUC 使用原始低方差比例采样器；BACC/F1 使用原始平衡采样器；
  Kappa 使用原始带独立 generator 的随机采样。
- AdamW 分为 gain/bias 不衰减和其他参数衰减两组；CPU/CUDA bfloat16，
  CUDA 使用原始 GradScaler。原始 cosine/warmup 在每次 optimizer 更新后调用，
  第一次更新使用完整初始学习率，随后才应用 step0 的预热学习率。
- 保留原始 workers4。输入适配维持索引42+idx的固定 patch 选择、补零和
  padding mask；VAL/TEST不裁剪。无坐标时仅补原始 loader 未用于预测的坐标字段，
  不表示获得了真实空间信息。

## 规划修复与锁定 TEST

原始规划器会分析所有切片形状与标签比例。本适配只向原始配置生成函数提供
TRAIN 切片统计和 TRAIN 标签比例。VAL、TEST包长度不能改变隐藏层、batch或采样计划。
这是与原流程的明确差异，配置名中的 train_planned 和协议记录共同标识。

序数任务以分类训练和显式 Kappa 指标适配，不增加额外序数损失。
生存、回归属于不同原始 trainer；本轮分类验证不能替代它们的独立验收。

## 恢复与运行版本

checkpoint保存模型、AdamW、scaler、原始早停、全局RNG、原始采样器池和游标、
独立generator、epoch起始输入状态、更新游标及完整训练轨迹。批次中恢复时，
从epoch起始状态重放确定性数据迭代，恢复workers4的预取与采样状态；
不会重放模型前向、反向或optimizer更新。原始内存best浅拷贝行为被保留，
真正选中权重以不可变序列化文件为准。

本基准的已运行原始流程使用 torch2.5.1、numpy2.2.6；本候选绑定同一运行版本。
作者 environment.yml列出torch2.3.0、numpy1.26，这一差异已披露，不能称为
作者原始环境复现。设备、数值后端、软件、特征和源码身份均与恢复请求绑定。

当前仅进行合成特征软件验证。完整原始轨迹、三种恢复边界、公开CLI、独立安装包、
父版本测试完整性及全库回归分别验收；不代表真实WSI或论文指标复现。

## 本轮实际验收结果

源码与独立安装包各758项专项检查通过；实际同名测试集合一致。
四种配置均执行原始100轮上限和patience10，AUC/BACC/F1/Kappa实际早停于
16/21/17/11轮，对应32/42/34/22次更新。逐次采样特征、通道选择、logits、
学习率、AdamW、早停和RNG与未改写的原始ClassificationTrainer.train()逐项一致。
三种中断恢复边界、公开CLI、选中权重重复评价及源码/安装包预测一致性通过。
284份父版本测试/fixture输入有哈希核对，12份目录扩展测试保留原文件；
4966项全库测试仅完成收集及父4931项身份保留核对，尚未全部执行。

验证过程保留了失败证据：初始运行版本假设错误、任务fixture缺少task_id、
临时目录过长导致AF_UNIX套接字创建失败，以及数值环境记录发生在原始初始化前。
分别修正为实际已运行依赖版本、完整任务定义、短临时传输目录，以及在原始
deterministic初始化后记录请求；没有减少训练轮数、worker数、宽度或比较断言。
既有MR源码精确对照在首次直接调用时出现微小数值差异；使用公开训练入口的
同一CPU数值初始化后通过。初始化条件显式保存，未宣称已经证明底层数值根因。

本轮只验收nnMIL分类路径及其明确标记的TRAIN-only规划修复。原始生存、回归、
真实WSI和论文指标仍需要各自证据；当前整个164条目集合尚未完成正式准入。
