# ReCo-MIL：上下文原型、稀有实例与双尺度训练

来源：[作者维护仓库](https://github.com/zsszc/ReCo-MIL)，固定提交 `27cefbc8d4a9a897b61ceb78b77626bc21d16685`。固定 README 仅给投稿信息；本轮另从 [Springer 论文条目](https://link.springer.com/chapter/10.1007/978-3-032-31666-0_19) 和 [ICPR2026 官方会程](https://icpr2026.org/sessions/poster3/) 核对了论文身份，出版社摘要直接链接同一代码库。本次适配针对这一公开源码版本；尚未取得完整论文正文，未宣称全文机制核查或论文指标复现完成。未发现根许可证；指定网络、门控和正则定义从外部固定 Git 对象读取，不在本包中重新分发或重新许可。

## 四个运行配置

| ID | 输入 | 原型压缩 |
|---|---|---|
| `recomil_single` | 单尺度切片特征 | MLP |
| `recomil_graph` | 单尺度切片特征 | 图压缩 |
| `recomil_dual` | 同切片高低尺度独立特征包 | 两个MLP分支 |
| `recomil_dual_graph` | 同切片高低尺度独立特征包 | 两个图压缩分支 |

家族名 `ReCo-MIL` 默认指向单尺度MLP配置。图压缩建立的是**原型之间的相似度图**，不需要物理patch坐标，也不替换成空间patch图。原网络包含patch/prototype投影、三层上下文增强、原型数逐层减半、patch与末层原型联合注意力聚合、分类头；默认输入维度来自FM登记、隐藏512、K128、三增强、dropout=.25、soft top-3路由和L2相似度。支持源码中的dot/cosine、hard straight-through路由等显式配置，实际设置保存在请求和协议中。

双尺度使用两个独立原型分支，按作者默认 `sigmoid(alpha)` 混合两个logits；alpha初始为.5，对应的实际高倍权重约.622，不能解释为精确各半。`model.low` 可以覆盖低倍分支的隐藏宽度、K、相似度、图压缩、门控、温度、冻结轮数和聚类设置；增强层数仍为共享构造参数。

## 任务与双尺度数据

目前接入作者公开的**二分类配方**，按每切片一包训练。患者评价时按任务协议融合概率；原始作者脚本按切片评价。支持任务数据包决定的评价单位，默认选模仍使用 `(AUROC+accuracy)/2`，仅使用验证集。当前不把此入口扩展为生存或任意多分类复现。

单尺度直接读取encoder.index。双尺度复用通用dual_scale接口，基础index是低倍，high_index为同一slide_id对应的高倍独立包，实例数可不同，不要求伪造逐patch配对或quadtree：

```yaml
encoders:
  uni_example:
    dimension: 1024
    index: features_low.csv
    weights: your-low-encoder-version
    preprocessing: your-low-extraction-protocol
    dual_scale:
      high_index: features_high.csv
      high_dimension: 1536
      low_field_of_view_um: 512
      high_field_of_view_um: 128
      provenance: your-actual-high-and-low-extraction-and-encoder-versions
```

high_dimension默认为基础特征维度；若两路FM维度不同，必须明确登记。FOV必须是实际物理组织视野，高低文件不可复用为同一文件，slide集合必须一致。ViLa仍要求两路与图文嵌入维度一致，不因接口扩展而放宽自身约束。

## 原型准备与完整训练配方

首次训练自动在**真正训练split**上按slide_id排序收集特征，运行作者所用MiniBatchKMeans：K128、batch_size4096、max_iter200、seed2025。原脚本可能对后来划作验证的train_list样本也聚类；本库仅使用最终训练患者。记录拟合患者、切片、特征内容哈希、各片采样数、初始原型与频率；初始原型、频率和训练中原型全部保存在checkpoint中，无需另备隐式外部原型文件。

算法保留原作者“采样后堆叠，再fit MiniBatchKMeans”的计算定义，不替换成流式partial_fit。大队列可能占用大量CPU内存；可显式设置sample_ratio或cluster_max_per_slide，并在协议中披露采样预算。样本少于K时失败，禁止静默随机初始化。构造网络时暂存零原型，随后严格填入训练原型；这与作者加载有效原型文件的Torch RNG消耗一致，不经过其随机fallback。

默认训练：Adam lr=1e-4、weight_decay=5e-4，50epochs、patience10、gradclip1、label smoothing=.03，一包一次更新。前12轮线性warmup后cosine；tau从1.2线性降至.7，第35轮到达；原型前12轮冻结，第13轮解冻。所有调度按作者一基epoch计算，断点保留并恢复这些状态。

稀有实例门控按离最近原型的距离选取 `max(1,int(N*.015))` 最稀有实例，包含阈值并列项，将输入放大1.2倍。可在原始或投影空间决定门控。正则默认原型Gram离对角损失、系数.02。可选类别权重只从训练slide标签计算。原型频率权重 `(freq+1)^(-.3)` 及均值归一化默认关闭，与作者默认freq_path未提供一致；`use_frequencies: true` 才启用，使用训练聚类频率和原压缩权重传播。

## 保留的源码行为与明确修复

- 作者evaluate没有传入稀有实例门控或频率权重；本适配保留这一训练/推理差异，没有将训练增强自动用于推理。
- similarity_scale通过`.item()`参与温度除数，实际没有梯度；保留并验证该惰性参数，没有把它标成有效可学习温度。
- `ortho_projected: true` 对应作者no_grad的投影原型辅助函数，正则不产生梯度；默认false的原始原型正则按正常梯度训练。此选项不宣称修复了源码机制。
- 作者主脚本没有创建单尺度DataLoader；本库用统一的显式切片数据装配补齐。标签来自任务登记，禁止根据文件名推断癌种或阳性。
- 作者原型加载对异常静默随机初始化；本库使用可追溯的训练专用聚类并严格拒绝错误输入。
- 不启动作者的在线W&B上报。保留本地日志、配置、权重和预测。

## 使用与验证

```bash
milplus setup --method recomil
milplus check --data /path/to/task --method ReCo-MIL --deep
milplus run --data /path/to/task --method recomil_single --config configs/recomil_classification.yaml --output runs/reco_single
milplus run --data /path/to/dual_task --method recomil_dual_graph --config configs/recomil_classification.yaml --output runs/reco_dual
milplus evaluate --run runs/reco_dual
```

真实WSI仍需先经TRIDENT提取特征、登记FM与物理尺度并冻结患者划分。四配置属于一个来源家族，不计为四篇独立论文。

软件验证分别对照：独立固定源码网络输出/联合损失/全部活跃参数梯度；原生1024/1536输入、512隐藏宽度、128原型；源码MiniBatchKMeans采样及中心/频率；直接抽取原始训练循环，在epoch1/2/12/13/14/35/50比较Adam参数和状态；跨原型解冻边界的精确续训。另检查不对称双尺度、训练专用拟合、无推理门控、惰性参数、投影正则、输入拒绝和CLI。实际通过记录见 [验证目录](validation/)。

当前验证环境为FP32 CPU，未启用作者可选CUDA AMP。GPU、真实WSI实验与论文指标复现未运行。
