# MoMIL：三路序列、Mamba2 与完整配方

来源：[作者仓库](https://github.com/YuqiZhang-Buaa/MoMIL)，固定提交 `73f0814d56701ff0012054d7c11cbe9637c6cdb3`。[出版社论文条目](https://www.sciencedirect.com/science/article/pii/S0262885626000247)（Image and Vision Computing167,105918,2026，DOI10.1016/j.imavis.2026.105918）直接链接该仓库。未发现根模型许可证；内含Mamba组件保留Apache-2.0等原始署名。本包只从外部固定Git对象加载指定定义，不重新分发或重新许可作者源码。当前未完成论文指标复现。

## 两个明确命名的配置

| ID | 任务 | 来源边界 |
|---|---|---|
| `momil` | 切片分类，包括多分类 | 原作者网络和CE训练；构造器名称作必要修复 |
| `momil_survival_extension` | 患者离散生存 | 原网络生存头与同仓库通用生存trainer的显式组合扩展 |

它们属于同一候选家族。MoMIL原始构造器调用 `super(SpeMIL,self)`，而该类未定义；将其修成 `super(MoMIL,self)` 后才能运行。测试同时验证原固定源码会在此失败，及最小修复后的初始化和数值图。没有用其他同名模型替换。

网络保留 D→512、ReLU、两层/路独立Mamba2（state64、conv4、expand2、head64）、层内残差、三路token拼接、LayerNorm、128维Sigmoid注意力和分类头。保留作者反复递归初始化Linear的实际行为及RNG消耗，不换成通用Mamba初始化。

## 三路变换与填充

实例序列先补到ceil(sqrt(N))平方，补的是包头已有特征，补充token一直保留到最终三路注意力。三路分别为：原序；字面源码的 `flip([-1])` **特征通道反转**；方格token转置。第二路没有被擅自换成实例倒序，融合也不先裁掉补充token。这里是序列组织，不声称构造了真实物理空间。

源网络使用已登记特征文件内的patch顺序。分类一切片一包；患者生存按slide_id排序后拼接其切片，每片内部顺序不变。不同切片没有伪造共同坐标框架。拒绝全局max_instances截断，避免悄悄改变序列。

## 计算后端

默认 `model.backend: torch` 读取本仓库捆绑的Mamba2.2.4定义，保留原模型的融合路径计算图，仅把conv/SSD/RMSNorm/输出投影内核明确实现为PyTorch FP32。完整状态可微跨块传递；分块梯度检查点不会截断历史梯度。MoMIL使用默认full-SSM、ngroups=1配置，不声称提供任意Mamba2配置、流式生成或张量并行接口。

`model.backend: cuda` 明确要求GPU、由固定 `mamba_latest/` 构建的mamba-ssm2.2.4，以及本适配选择的causal-conv1d1.5.0准入环境。作者源码仅给causal-conv1d>=1.2.0，没有指定1.5.0；这里的精确版本是待验证环境选择，不冒充作者历史版本。记录Torch/CUDA、包版本、Mamba Python文件和causal二进制哈希，恢复拒绝后端身份改变；版本字符串不能代替环境构建来源证据。缺少依赖或GPU时失败，不自动回退到普通Mamba。

源CUDA与本次CPU环境不同，CPU递推耗时不能用于宣称原融合CUDA吞吐。当前没有执行融合CUDA对照、GPU训练或真实WSI实验。

GPU环境构建完成后可执行：

```bash
python scripts/validate_momil_cuda.py --output /path/to/new_cuda_check.json
```

它在多个包长和作者默认宽度下比较融合/参考完整网络的输出与参数梯度；无GPU时写入not_run并返回非零。当前CPU环境的实际准入结果为not_run。

## 分类配方与模型选择

默认遵循源CLI：200epochs、Adam lr1e-4、weight_decay1e-5、dropout=.25、无默认梯度裁剪或scheduler，每切片一次CE更新，默认无weighted sampler、无early stopping。**不启用early stopping时保留最后一轮**，与原作者入口一致，不偷偷换成验证最优。

`model.early_stopping: true` 启用验证选模，默认log_loss、patience20，零基epoch>50后允许停止。可显式选择类别逆频率有放回采样。统一数据包决定患者/切片评价单位；作者分类入口原为切片评价。可用任务协议的验证侧指标覆盖默认选择，测试标签不参与选择。

作者BRACS启动脚本另设lr2e-4/dropout0并启用early stopping，但还传了当前main不接受的mambamil_layer参数及未定义变量；不能把那份shell当成已跑通的当前入口。提供独立 `configs/momil_classification_early.yaml` 记录其主要数值设置；默认配置仍是源CLI行为。

## 生存扩展边界与修复

原MoMIL类有sigmoid/cumprod生存输出，但发布的生存模型工厂没有MoMIL分支；脚本又传momil，而parser仅列mo2mil等值。这条扩展明确组合原头和同仓库通用NLL trainer，不宣称原MoMIL生存入口已可直接运行。

默认四区间、alpha0，Adam/lr/wd同源CLI，dropout=.25；按(时间区间,删失状态)联合类别逆频率有放回采样，32患者累积。按作者k-fold方案采用验证C-index与patience20，零基epoch>40后允许停止。训练时间区间仅拟合唯一训练患者的观察事件时间，修复原全队列qcut；最后短窗口按实际患者数平均并更新，避免原代码把残余梯度留到下一epoch。完整窗口的目标和更新保持源行为。

可选l1_weight默认0，保留源逐患者附加L1的计算位置：每患者loss/gc加一次L1，不能无说明改成每优化器窗口加一次。只接入图像分支全部参数L1，不假装存在组学模块。对齐测试将c-index汇报回调隔离为无副作用占位，实际训练更新仍执行原函数；这不等于已验证作者外部生存统计库。

## 使用

```bash
milplus setup --method momil
milplus check --data /path/to/task --method MoMIL --deep
milplus run --data /path/to/task --method momil --config configs/momil_classification.yaml --output runs/momil_cls
milplus run --data /path/to/patient_task --method momil_survival_extension --config configs/momil_survival_extension.yaml --output runs/momil_surv
milplus evaluate --run runs/momil_cls
```

无需伪造坐标或额外组学。原始WSI需先用TRIDENT提特征，再建立患者、标签、划分、FM版本和patch顺序清单。

测试独立读取**此固定2.2.4版本**的selective_scan_ref、SSD/RMSNorm参考；对照跨块状态、完整投影内核、全模型输出与全部参数梯度；在相同数值内核下运行原分类trainer及通用生存trainer，逐位比较Adam更新；另核验三路输入、保留填充、原生512隐藏宽度/state64、初始化、采样、最后轮选模、训练专用分箱、短窗口修复和精确续训。当前软件身份对应的实际结果见 [验证目录](validation/)。合成CPU验证不证明论文数值复现或真实队列结论。
