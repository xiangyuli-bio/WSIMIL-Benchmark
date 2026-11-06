# AdvMIL：区域投影与对抗生存训练

原始来源：[liupei101/AdvMIL](https://github.com/liupei101/AdvMIL)，固定提交
`6853786624bbab36c2a08b35dcf20c7bdff1a105`，MIT。运行时从已校验的 Git 对象加载作者定义，
不会执行作者主程序或启动 W&B。没有将其他 MIL 换名为 AdvMIL。

## 四个执行配置

| ID | 生成器骨干 | 判别器目标 |
|---|---|---|
| `advmil_esat` | 作者 ESAT：4×4 区域池化、Transformer、门控池化 | 显式修正后的 real/fake BCE |
| `advmil_abmil` | 作者门控 ABMIL | 显式修正后的 real/fake BCE |
| `advmil_esat_source` | 作者 ESAT | 公开源码的字面 `bce` 表达式 |
| `advmil_abmil_source` | 作者门控 ABMIL | 公开源码的字面 `bce` 表达式 |

四者属于一个方法家族。`--method advmil` 解析为 `advmil_esat`；实际 profile 始终写入请求和配方。
本次覆盖连续生存的 ESAT/ABMIL 路线；Graph、DeepAttnMISL 骨干和离散生存路线没有计为完成。

作者 `loss/utils.py::real_fake_loss` 的 fake 项为 `mean(log(sigmoid(fake)+1e-8))-1`，
不是标准的 `-mean(log(1-sigmoid(fake)))`。源码配置保留该行为，修正版采用等价稳定的 softplus BCE。
两者的生成器仍采用作者的 `-mean(D(fake))`，不能解释为更换了完整 GAN 目标。
这是固定源码观察，不能据此推断原论文实验结果。

## 数据包

使用既有 survival `slides.csv`，每位患者有正的 `time`、二值 `event`（1=事件，0=删失），
`time_unit: months` 或 `days`。同一患者多张片的结局必须一致。
即使选 ABMIL，判别器仍依赖真实区域，因此需要：

```yaml
encoders:
  uni:
    dimension: 1024
    index: features.csv
    regions:
      coordinate_frame: level0_pixels
      patch_stride: 256
      preprocessing: "本次切块和区域组织版本；填写实际来源"
```

`features.csv` 每行提供 `slide_id,path,coords_path,region_ids_path`；HDF5 内有 coords 时可省略 coords_path。
region_ids 是与特征行严格对应的非负整数 NPY `[N]`，可选 `region_ids_sha256` 校验。
每组必须恰好 16 个不同坐标，组成间距与 patch_stride 一致的完整 4×4 网格。
加载器按区域 ID、y、x 排序；全片重复坐标、缺块、错序对齐和错误间距均拒绝。
任意 TRIDENT patch bag 不能自动视为这种区域结构。不能补零或伪造坐标来通过检查。

患者多片在每片分组验证后拼接，不跨片组成区域。`max_instances` 只能采样完整区域，
实际预算向下取为 16 的整数倍；小于 16 的设置被拒绝。预算未必保留患者所有片的区域，
采用预算时须将其作为明确实验条件。

## 保留的训练流程

- D 更新：G.eval、D.train；真实对只来自可见标签且 event=1 的病例，所有病例的生成时间 detach 后组成假对。
- G 更新：G.train、D.eval；重新采噪声，判别分数对 G 反传；可见病例使用作者删失 L1 重建目标，另加 GAN 和全部 G 参数 L1。
- 默认一轮 D、一轮 G，每 16 位患者更新。两个独立 Adam；G 的 bias/一维参数不衰减，D 不做 weight decay。
- 默认 G 隐藏宽度 384，D 宽度 128；作者噪声 `[0,1]`、一层中间 MLP、uniform、sigmoid 输出；支持明确选择 gaussian。
- 默认 lr=8e-5、G weight_decay=5e-4、GAN 权重 .004、L1=1e-5、300 epochs。
- G 使用 validation 重建损失的 ReduceLROnPlateau（factor=.5、patience=10）；默认第 6 个 epoch 起选模，early stopping patience=30。
- 配方/优化器/随机状态/时间缩放/标签可见清单/半监督循环折/调度状态完整保存，支持 epoch 边界精确续训。

## 数据与执行修正

作者数据表的 ratio 缩放使用整个 CSV 的最大时间。本实现只用**可见训练患者**最大时间，
隐藏训练结局和验证/测试结局不参与拟合；验证/测试目标不截断。sigmoid 输出仍意味着预测时间受训练最大值限制。

作者循环会丢弃最后不足整批的病例；这里保留尾批，避免小数据集整轮零更新。
`square2sequence` 的 transpose 后 view 改为保持元素顺序的 reshape，兼容非连续张量。
验证选模遵循公共引擎的严格改善规则，没有采用作者 1e-6 最小改善阈值。

验证默认 1 次采样，测试默认先单采样再独立采 30 次并取 `torch.median`（偶数时取较小中位数）。
为可重复评价，CPU 噪声使用患者 ID 派生的隔离随机状态；不推进训练随机流。
输出患者负预测时间作为 risk，并保存预测时间、单次预测、采样标准差和重建损失。
公共指标是患者 Harrell C-index，未实现作者全部 IPCW/Brier 等统计协议，标准差不能视为校准置信区间。

## 显式标签隐藏实验

默认完全监督。可在 task.yaml 增加：

```yaml
adversarial_training:
  mode: UD+LD
  visible_training_patients: [patient_A, patient_B]
  unlabeled_folds: 5
  label_provenance: "预先固定的训练结局可见性清单版本"
```

使用 global_patient_id，只接受训练患者，不自动抽样或引入验证/测试患者。
UD+LD 每 epoch 使用全部可见病例及一个循环的隐藏病例折；UD 每轮使用全部隐藏病例，LD 只取可见病例。
unlabeled_folds 只影响 UD+LD 的采样；UD/LD 不根据折数删减病例。
隐藏患者结局不会进入重建、真实对或缩放；仍提供图像构造假对。
此协议对应作者“已知结局再隐藏”的实验，数据表仍要求完整结局，**尚不支持自然缺失结局队列**。
其他方法遇到 adversarial_training 字段会拒绝执行，以免默默使用本应隐藏的标签。

## 使用

```bash
milplus setup --method advmil
milplus check --data /path/to/region_survival_task --method advmil_esat --deep
milplus run --data /path/to/region_survival_task --method advmil_esat \
  --config configs/advmil_esat.yaml --output runs/advmil_esat
milplus evaluate --run runs/advmil_esat
```

软件证据见 `validation/adversarial_survival.json`。测试包括作者原训练方法的多次 D/G Adam 更新、
loss/梯度/参数对照、两阶段参数归属、区域几何、标签隔离、尾批、四种监督模式精确续训。
这些是 CPU 合成验证；GPU、真实 WSI 和论文数值复现尚未执行。
