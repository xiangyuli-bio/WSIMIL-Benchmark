# RankMix：两阶段训练与显式源码对照

作者来源：[willpower057/RankMix](https://github.com/willpower057/RankMix)，
固定提交 `a6d51301ec878540f4dd185544c265fad5ef1dd2`，MIT。
[CVPR 2023 论文入口](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_RankMix_Data_Augmentation_for_Weakly_Supervised_Learning_of_Classifying_Whole_CVPR_2023_paper.html)。
当前实现依据锁定源码的 `dsmil.py`、`frmil.py`、`train_mil.py`、
`train_mil_rankmix.py`、`dataset.py`、`dataset_frmil.py` 和启动脚本逐项核对。
模型定义从单独下载的 Git 对象加载；源码和作者权重没有打包进 PULS。

## 四个执行配置，一个方法家族

| ID | 骨干 | 实例选择 |
|---|---|---|
| `rankmix_dsmil` | 作者仓库中的 DSMIL | 修正选中 patch ID |
| `rankmix_dsmil_source` | 同一 DSMIL | 保留作者 argsort 位置索引行为 |
| `rankmix_frmil` | 作者仓库中的 FRMIL | 修正 patch ID，并排除 padding 候选 |
| `rankmix_frmil_source` | 同一 FRMIL | 保留作者位置索引及 padding 行为 |

`--method RankMix` 指向 `rankmix_dsmil`；其规格和运行协议明确标记修正。
这四个配置只覆盖作者脚本的**单正类 logit 二分类路线**。不宣称完整多类别、
CutMix、图像训练、固定打分器或蒸馏路线。FRMIL 的作者实例评分为 sigmoid 概率，
不能再次作为 logits 传给 BCEWithLogits。推理只用 bag sigmoid，输出 `[1-p,p]`。

## 数据与完整训练过程

输入为标准任务包的单张切片冻结特征 `[N,D]`、二分类标签、患者划分和 FM 登记。
同一患者的多张切片可在评价时汇总，但不把不同切片直接拼成一个 FRMIL 伪空间网格。
FRMIL 要求 D 能被注意力 head 数整除。其序列方格来自作者模型，不冒充真实组织坐标。

1. 监督初始化阶段训练同一骨干；DSMIL 使用 bag/最大实例两项 BCE；FRMIL 使用
   bag BCE、最大实例概率 BCE、原始阳性/阴性两包 feature-magnitude 三项目标。
2. 在独立验证集选初始化权重；进入 RankMix 阶段时载入它、清空 Adam 状态并重启 cosine。
   预训练阶段不会成为最终 RankMix checkpoint。默认 warmup 20 轮；完整示例配置为
   200 轮监督初始化加 200 轮 RankMix。科研训练预算需另行固定。
3. DSMIL 每轮自然频率遍历训练包，另一路按类别频率倒数有放回采样；
   lambda 为 `Beta(alpha,1)`。FRMIL 每个 episode 每类取一个包、每轮 N_train 个 episode；
   lambda 为 `Beta(alpha,alpha)`。默认 alpha=.1。
4. 打分网络处于 eval/no_grad，选共同数量的实例、按原始顺序排列后混合特征与软标签。
   训练网络的实例评分和 bag 分类参数继续联合更新；没有冻结打分器。
5. DSMIL 只训练混合包的双 BCE。FRMIL 同时前向原始两包和混合包；分类损失只取混合包，
   magnitude 仍来自原始阳性/阴性包。三项均参与梯度。
6. 保留 .2 特征 dropout、Adam betas=(.5,.9)、weight_decay=.005、lr=.0002、
   阶段内 cosine 最低 lr=5e-6。默认不裁剪梯度；可显式提供正 `grad_clip`。
   FRMIL 按最大**训练**包补零，模型和 magnitude 损失仍看到作者式 padding；
   修正版只在 RankMix 选择候选时排除它。

FRMIL margin=10.17 来自作者 Camelyon 启动脚本，依赖特征尺度；它不是所有 FM 的已验证最优值。
`max_instances` 是 PULS 的显式资源预算适配，默认不截断。作者原始 dataset 硬编码 D=512，
且重复定义的读取函数使用已移除的 `np.float`；PULS 使用通用安全特征读取，不执行这段读取代码。

## 固定源码中观察到的差异

- `mixup_by_rank` 先得到 top-k 原始下标，随后用 `torch.sort(indices)[1]` 取排序位置。
  例如 top-k 是 `[3,1]`，实际用于读取的变成 `[1,0]`，并不是 patch 3 和 1。
  修正版取 `.values`，保留选中集合及原始顺序。源码配置保留原行为便于审计。
- FRMIL 对含补零位置的完整序列评分，补零可能进入候选。修正版只在真实长度内选实例；
  其余网络和损失中的 padding 行为保留，不宣称实现了全网络 padding mask。
- `rank2` 对阴性包升序、阳性包降序；`rank` 均降序。`keep_percent` 仅控制共同实例数比例，
  向下取整为空时明确拒绝。没有继承原代码中对 `top_k` 参数的遮蔽问题。
- 作者脚本将训练集再用作 val、每轮读取测试集、用当前测试集拟合分类阈值。
  PULS 使用固定患者划分和独立验证选模；测试集只有显式 evaluate 时才读取；
  二分类默认阈值 .5。这是公开的评估协议变化。
- 作者脚本要求单独运行监督预训练并手填 checkpoint 目录；PULS 将两阶段纳入同一记录。
  初始化权重、阶段切换、两个阶段的 scheduler、Adam、RNG 都进入 checkpoint。
  不把先随机打分再混合的单阶段训练冒充该两阶段配方。

这些是指定提交的观察，不据此推断原论文的历史实验是否使用了相同代码。

## 使用

```bash
milplus setup --method RankMix
milplus demo --output /tmp/rankmix_demo
milplus run --data /tmp/rankmix_demo --method rankmix_dsmil \
  --config configs/rankmix_demo.yaml --output runs/rankmix_demo
milplus evaluate --run runs/rankmix_demo
```

正式包使用 `configs/rankmix.yaml`，并根据锁定的训练预算和特征尺度确定超参数。
切换源码对照时使用新输出目录和 `*_source` 方法名。

## 验证范围

`tests/test_rankmix.py` 包含作者完整单步训练（两骨干、两选择行为、两排名方向）
的目标、梯度和 Adam 更新逐位对照；索引反例；padding 排除；bag-only 预测；
warmup 目标；最佳初始化恢复；调度重启；四配置在阶段切换前后中断的精确恢复；
train-only padding 和非法输入检查。全套证据见 `docs/VALIDATION.md`。
这些属于 CPU 合成软件验证，真实 WSI、GPU 和原论文数值复现尚未完成。
