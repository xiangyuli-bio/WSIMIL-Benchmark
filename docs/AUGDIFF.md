# AugDiff：完整训练流程与公开代码缺口

这是隔离的 0.68 开发候选，已实现完整 DAE 与 MIL 流程及六个显式命名配置，
**尚未通过全部正式准入检查，不能作为已完成原论文复现的方法**。
正式主库已验收 0.61：116/164 条方法或家族、217 个执行配置。
本开发候选已写入123条实现、233个配置，其中7条等待全部正式验收。

## 固定来源

- 原方法：[szc19990412/AugDiff](https://github.com/szc19990412/AugDiff)，
  `abfdfe25c6373019d71bf3323cf59be799203563`，MIT。
- 扩散依赖：[CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)，
  `a506df5756472e2ebaf9078affdde2c4f1502cd4`，MIT。
  AugDiff 作者只链接此项目，未锁定历史依赖提交；本固定版本不代表还原了作者历史环境。
- 已阅读完整 [arXiv:2303.06371](https://arxiv.org/abs/2303.06371) 的13页正文及附录。
  arXiv 与后续期刊材料的数据集数量不应混用。

## 已实现并检查的部分

1. 私有加载真实 LDM UNet、DDPM、LatentDiffusion、IdentityFirstStage、LitEma
   及作者 DDIM，保留 YAML 的256基础宽度、1/2/4通道倍率、每层2个残差块、
   全部注意力、1000步噪声调度和 epsilon MSE。完整512维重建网络有122,055,680参数。
2. 原始 AMIL、LossAttn、DSMIL 三个分类器及分类损失。
   DSMIL 保留先平均 bag/instance logits 再做一次 CE 的实际源码行为。
   LossAttn 中未使用的可学习线性层仍保留，不将无梯度误判成缺失训练机制。
3. 作者 RAdam 和 Lookahead 的实际更新。保留第6步才初始化慢权重的源行为，
   修复现代 Optimizer hook 初始化及以 Python 对象ID保存慢权重造成的恢复问题。
   17次连续更新、跨第6/12步慢权重更新及中途新建模型恢复均与原数值步骤一致。
4. 原图及六种实际图像增强视图的特征输入：affine、blur、color、elastic、HED、rotation。
   绑定相同FM、完整维度、切片归属及文件内容哈希，不用特征噪声伪造图像增强。
   原作者无条件DAE将视图特征直接拼成训练池，不要求视图逐行配对。
   默认 `view_pairing_policy: independent_augmentation_pools`，各视图可有不同的patch数量和顺序；
   每个视图的全部行均进入池，不裁剪到最短视图，也不伪造跨视图坐标对应。
   需要 `view_identity_provenance` 说明各视图的提取来源；有序patch哈希可选，提供时必须合法。
   可选 `paired_patch_views` 要求相同行数、记录及所有视图相同的有序patch哈希，
   同时填写 `row_correspondence_provenance`。这是额外数据审计约束，不是原方法必需条件。
   两种模式均要求原图视图确实为监督TRAIN的注册特征，且保持患者内部DAE划分及无监督VAL/TEST泄漏。
5. 完整 DAE 预训练控制器：七视图平坦patch池、AdamW、原自适应EMA、随机时间/噪声、
   原先raw再EMA的两次独立验证、EMA验证损失选模、模型/优化器/AMP/RNG/采样顺序检查点。
   MIL 按源加载器使用选中检查点的 **raw权重**，不是EMA推理权重。
6. 完整 MIL：原三个分类器各配两种时间步条件；TRAIN随机顺序、TRAIN与VAL各7次随机增强，
   batch1、Lookahead/RAdam、梯度累积2、自动GPU半精度、验证CE选模、200轮上限及耐心10。
   奇数末尾单bag仍除以配置的累积数2，保留原Lightning1.2.3的尾批行为。
   七次VAL损失用于选模；通用指标先将同一真实切片的七次概率取均值，不将增强副本当新病例。
   TEST只使用干净特征，不加载或调用DAE。DAE冻结且排除于分类器优化器和分类器检查点。
7. 完整恢复：分类器、原RAdam状态与Lookahead慢权重、AMP scaler、患者/切片输入约束、
   顺序、梯度累积窗口、验证随机流与所有RNG。选中模型的VAL重放不消耗调用方随机流。

## 明确的重建与修复

公开 Feature 数据返回 `[B,1,D]`，原LDM图像 `get_input` 将它转成 `[B,1,1,D]`，
与 YAML 的512输入通道不符。作者采样器又使用 `[N,D,1]`，而 YAML 没有修改默认二维UNet。
公开材料不足以确定作者实际使用的特征UNet实现，所引用的预训练检查点也未提供。

当前明确命名为 `singleton_1d_reconstruction`：使用全部原生宽度和层数的原UNet，
选择一维卷积，使每个patch为一个D通道特征点；在长度1时保持上采样长度为1，
以便完整skip连接。Identity编码、epsilon目标的所有非batch维均值、设备移动均显式适配。
**这是缺失特征布局的公开重建条件，不是已证明等同于作者未公开模型或历史权重。**
不能将它写成“原始架构和原论文结果已复现”。

发布的 DDIM 在第一次注入噪声后重复使用同一个 `ts`。
保留字面参考入口，并单独提供每步更新时间的修复版。
修复版 T20/nu0.4 的实际网络调用为351、301、251、201、151、101、51、1。
eta0仍保留源代码的随机数消耗；前向扩散从原始特征开始，不是随机生成新bag。

原LDM激活检查点未保存前向autocast上下文，在现代PyTorch半精度反向中可能产生输入/偏置类型冲突。
适配器记录并在重计算时恢复原设备、精度和autocast状态，继续使用原检查点算法。
完整122M网络已完成CPU BF16反向检查；原生1024通道注意力的检查点/直接求导结果及所有梯度逐位一致。
这两项不替代独立CUDA FP16检查。VAL CE使用浮点log-softmax，避免先转半精度概率再取log的下溢。

公开MIL仓库缺少实际导入的 `datasets/data_interface.py`，无法证明原DataLoader的完整行为。
本适配明确使用TRAIN随机采样、VAL顺序采样、workers0；模型构造参数中的 `data_shuffle=False`
不能被当作原DataLoader顺序已知的证据。

原扩散数据按每个视图的无序glob做80/20划分，可能跨患者及监督边界。
本受控条件要求在**监督TRAIN患者内部**显式分组DAE训练与验证。
这是披露的数据协议修改，不能声称原训练划分逐字相同。

论文和YAML没有给出扩散预训练轮数。控制器要求显式传入轮数；
私有路径出现 `epoch=000368.ckpt` 仅是作者示例，不能据此推断所有任务统一369轮。
原 `base_lr=5e-8` 按实际patch batch、设备数、梯度累积缩放；当前控制器为单设备。
默认batch1200；较小batch的软件检查不代表完成此原生批次预算。

## 使用流程

先准备符合公共数据包规范的分类任务、七种真实图像增强的同FM特征及 `augdiff_views.json`。
不能只提供干净特征或用随机扰动代替原图增强。

```bash
milplus setup --method AugDiff

# DAE_EPOCHS 为研究协议显式选择的预算，公开材料没有统一轮数。
milplus pretrain-augdiff --data /data/task --views-manifest augdiff_views.json \
  --epochs "$DAE_EPOCHS" --batch-size 1200 --precision fp16 --device cuda \
  --output /runs/augdiff_dae

milplus prepare-augdiff --data /data/task --views-manifest augdiff_views.json \
  --pretrain-run /runs/augdiff_dae --output /data/task/augdiff_model.json
```

将所用 `task.yaml` 编码器项登记为 `augdiff: {manifest: augdiff_model.json}`，再使用统一入口。
原始视图、预训练协议和选中检查点是资产追溯的一部分，应保留于可访问的持久目录。
导出工具拒绝未训练、未完成、输入变化、来源不符或未执行真实更新的DAE检查点。

六个配置为 `augdiff_{amil,lossattn,dsmil}_{released_time_reconstruction,schedule_repaired_reconstruction}`。
`--method AugDiff` 默认选择AMIL修复时间步条件；例如：

```bash
milplus run --data /data/task --method AugDiff --output /runs/augdiff_amil
milplus run --data /data/task --method augdiff_dsmil_released_time_reconstruction \
  --output /runs/augdiff_dsmil_control
```

首次资产准备后，日常训练由数据路径、方法名称及结果目录确定。
`milplus demo --augdiff` 仅生成明确标记的合成软件数据，不能充当真实WSI或图像/FM验证。

## 当前验证边界

R3的42项专项检查已通过：两种完整输入维度下三个原始分类器的输出、损失与所有梯度，
单patch秩修复、原RAdam/Lookahead更新与恢复、完整宽度DAE的真实反向/AdamW/EMA、
训练后DDIM调用、七视图完整读取、患者泄漏、输入不可变性、完整MIL阶段及目录重建保持等。
R4扩展独立视图池，改变了资产代码约束，必须重新准备并验证当前版本的DAE资产；
旧R1/R2/R3运行保留作为历史证据，不挪用其收据充当R4训练。

另已在完整122,055,680参数DAE上执行2轮、每轮1次实际更新的软件预算检查；
使用七种合成标记视图，连续训练与中断后新建模型恢复的raw+EMA全部张量、
AdamW、scaler、历史、顺序及最终RNG逐位一致。
这些视图只用于软件验证，不声称真实WSI、实际图像增强或FM提取。

R2六个MIL配置均已实际训练并重载，含17张TRAIN切片、每轮119次抽样、60个累积窗口及奇数尾批。
连续两轮与中断后新建模型恢复的全部分类器/优化器/慢权重、顺序、验证流及RNG逐位一致。
这些属于软件机制证据；R3增加AMP上下文修复，R4扩展视图池，需要各自新版本检查。

待完成：R4全部专项、当前版本CLI训练与恢复、独立安装包CUDA半精度与完整预算检查、
全库与所有233配置验收、最终安装包重载及有序合入。当前规格状态为 `adapted`，
执行配置为 `implemented_unvalidated`；真实WSI、历史检查点和论文指标均未复现。
