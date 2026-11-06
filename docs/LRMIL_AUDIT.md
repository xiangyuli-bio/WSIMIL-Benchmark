# LRMIL 接入审计（进行中）

完整三段训练调度、保存恢复和统一入口已写入，首次原生编码器端到端软件训练通过。**LRMIL 已计入129条已写入条目，正式验收仍待当前完整检查及完整预算安装包验证。**

来源是 [作者仓库](https://github.com/hvcl/LRMIL)，固定提交 `66ea1c5a447ed2a830bb1af6739ab9261e61000d`，整体 Apache-2.0。研究描述见 [论文](https://arxiv.org/abs/2606.06864)。所需 CONCH 权重保留其原始使用条款。

| 环节 | 固定源码观察与已落实的处理 |
|---|---|
| 完整模型 | 使用作者 AttentionMIL、MILClassifier、StudentMIL、StudentBackbone 和全部六个 loss/helper。原生视觉编码器为768维、12层、12头、86,250,240参数，实际加载全部150个 CONCH 视觉张量 |
| 原始入口 | `main.py` 导入不存在的 `lrmil.trainer`；实际文件在 `utils/`、`models/`、`dataset/`。目前通过固定 Git 对象私有加载真实定义，未修改作者仓库 |
| 动态图像布局 | 实际 CONCH 的 timm 动态 patch embedding 返回 NHWC，原 StudentBackbone 按 NCHW 展平会失败。已显式修正 token 排列，并用未经修改的作者类加独立 NCHW 适配器完成全输出、全部参数梯度逐位对照 |
| 位置编码 | 原代码直接截取448px预训练位置网格的前257个 token。保留为 source_slice；真正的二维插值是单独指定的 spatial_interpolation，不能混称同一条件 |
| 教师元数据 | MILClassifier 返回 logits、bag、attention；save_teacher_meta 把 bag 和 attention 错位。已以实际作者函数构造反例，并单独修正输出对应关系 |
| Patch 蒸馏 | 保留源码 layer0/1/10、实际4×4 token 区域平均、逐层 MSE 平均，以及额外的注意力加权特征 MSE；不得静默删掉第二项目标 |
| 学生蒸馏 | 保留 CE、bag KL、soft attention KL、top/bottom 实例分类四项目标，以及实际源码的 attention sum 门槛、温度平方和 batchmean 缩放 |
| 输入与尺度 | 独立登记真实 RGB256、20x/5x MPP、全部高倍率中间层、每个尺度坐标和权重来源；40x扫描坐标通过实际 MPP 换算为源码20x像素单位。不得用对比特征512维自动投影冒充原生中间层 |
| 数据身份 | 校验全部 slide、患者、标签、划分、文件哈希、三层特征、RGB边界和实测匹配。允许实际组织导致的局部缺失网格，保留每个真实匹配，不补造高倍率 patch |
| 训练依赖 | 源码 Patch KD 需要已训练教师的注意力，因此完整自动流程需要明确安排教师训练、patch KD、学生训练；不能只接最终学生分类头 |
| 数值与任务边界 | 论文和源码存在层编号、额外 patch 目标、hard 监督定义等差异，须分条件记录。论文所称 survival 按 alive/dead 分类实施；当前不宣称是删失时间生存模型 |
| 未给定的配置 | 命令行训练参数多数默认 None，完整配置和部分数据生成步骤未公开。后续执行配方必须明确登记补充选择，不能冒充作者原始超参数 |

独立依赖使用 timm0.9.12，满足固定 CONCH 源码的 `timm>=0.9.8` 条件；LRMIL 未公开历史依赖锁，因此这里只声明所采用的确定版本。依赖安装在私有覆盖目录，不修改正在验证的0.74环境。

三段实现保留三份 Adam、真正选中的教师/编码器、各阶段重置和真实采样记录。教师循环和 patch 循环已分别与作者完整训练函数的权重、全部 Adam 状态和 RNG 逐位一致；跨阶段续训、多分类、逐配置及完整预算安装包检查继续验收。尚未进行真实 WSI 训练或论文数值复现。
