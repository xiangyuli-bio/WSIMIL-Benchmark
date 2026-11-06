# 后续状态空间方法源码审查，2026-10-21

下面保留两项的源码审查记录。MamMIL 已在 0.29 接入；GMMamba 仍待适配。源码已获取并核对原始论文入口；下面是固定版本的源码观察，不能推断历史论文结果错误，也不代表已执行论文复现。

## MamMIL

- [作者仓库](https://github.com/Vison307/MamMIL)，固定提交 `64736e1a06a23e6814dfee664ff02b27c72dfedb`；[论文](https://arxiv.org/abs/2403.05160)。根许可证 GPL-3.0，后续继续采用外部来源加载并登记具体组件署名。
- 当前主配置调用 `models/MamMIL2.py`。虽然名称包含 Mamba，实际 `DirectMamba` 位于 `mamba/mamba_ssm/modules/direct_mamba.py`，使用 **Mamba2 SSD** 的 chunk-scan、gated RMSNorm 和分方向投影。不能直接套用已验证的 MambaMIL/MSMMIL 一阶扫描内核。
- 图构建在 `h5toPyG.ipynb`：坐标 HNSW 邻接、特征余弦距离边权、最小生成森林、DFS 前序/后序与各分量随机根 BFS 序。另建 latent 特征图。需固定 HNSW 参数、节点/边顺序、随机根、图文件哈希和 FM 版本。
- 主网络以 `range(layer_n)` 创建扫描层，以 `range(1,layer_n)` 创建 GENConv/DeepGCN 层，forward 使用 `zip(self.layers,self.ppeg)`。发布默认 layer_n=2 时，只执行第一组扫描层。需分别登记字面源码与经论文机制核验的修正路线，并对全部参数梯度做检查。
- 分类和生存配置均为 embed_dim=512、RAdam lr=1e-4、weight_decay=.05、250 epochs、patience=20；分类按验证 loss，生存按验证 C-index。需保持具体作者优化器和参数分组，不能改成通用 CE/Adam。
- 原生生存模型从 kwargs 读取 label/c，返回独立 NLL。数据层按全队列拟合时间分箱；后续必须建立训练专用分箱、患者图组织和只用于损失的标签边界。

0.29 已完成上述接入和 CPU 软件检查，包含三个明确命名的配置；当前使用与差异见 [MamMIL](MAMMIL.md)，实际通过记录见 [验证目录](validation/)。GPU、真实 WSI 和论文指标复现仍未运行。

## GMMamba

- [作者仓库](https://github.com/titizheng/GMMamba)，固定提交 `70576f1aaeef0d7041d6860799ade1ac905b85aa`；[ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_GMMamba_Group_Masking_Mamba_for_Whole_Slide_Image_Classification_ICCV_2025_paper.pdf)。未发现根许可证。
- 发布仓库有 IMM、CSS、Transformer 分类头和完整 CE/AdamW 训练循环，但 `models/IMM.py` 引用 `mamba.mamba_ssm.BiMamba` 等类，仓库没有该 `mamba/` 目录或锁定依赖来源。不能假定其他仓库同名类就是原始后端。
- 坐标 KMeans（默认10组）在 `datasets/pre_input_data.py`。索引使用 squeeze，单实例分组会丢失实例维度；IMM 至少屏蔽一个实例，单实例组可能成为空序列。需要明确小组契约与修正，而不是静默丢包。
- IMM 声明 `_fc1` 输入投影，但发布 forward 未调用它；后续 LayerNorm/BiMamba 固定512维。默认 in_chans=512 可以满足该路径；替换成其他维度 FM 时须有单独命名、验证的投影适配。
- `mask_attention` 仅经排序和布尔索引选择实例；在所读 CE 训练路径中未发现额外可微 selector 目标。该选择路径的参数梯度需要单独验证和披露，不能只凭模块名称称其已学习。
- CSS 使用跨组最大特征、相似度加权与 MultiheadAttention，之后进入带 class token 的分类头。默认 num_layers=1；多层时 class token 的返回维度还需核查。
- 实际训练使用 AdamW lr=1e-4、weight_decay=1e-5，未使用 parser 的 lr 值；100 epochs、patience20、epoch>50 才停止，epoch>5 后按 val_F1+val_AUC 选择。测试在验证改善后另行评价，不参与该比较。

下一步：确认精确 BiMamba 来源；补齐分组/缓存身份；验证 IMM/CSS/分类头的梯度与原始更新；将必要维度、小组和多层修正独立登记，再接入统一训练与恢复。

### 2026-10-22 执行审查与隔离开发

实际加载固定 Git 对象中的 IMM、CSS、分类头，并以明确标识的 MambaMIL BiMamba 参考绑定执行三步原始 CE/AdamW 更新；这不证明缺失的原始依赖具有相同身份。实际反向证实 `_fc1` 和 `mask_attention` 在发布路径中没有梯度；两次 BiMamba、组内池化、CSS 和分类头存在有效梯度。非512输入、单实例组和多层分类头的失败边界也已实际触发。

完整配方候选暂保存在隔离开发目录 `/tmp/milplus_gmmamba_v53`，主库不新增 GMMamba 已完成计数。候选以 `gmmamba_mambamil_source` / `gmmamba_mambamil_projected` 明确命名依赖参考重建及输入修正，包含原始验证选模/早停、真实坐标分组、全部模块训练和断点恢复；已执行的独立来源审查见 `validation/gmmamba_source_audit.json`。其正式准入及完整回归仍待完成，不能以模型同名或局部检查替代原始依赖与整体训练证据。

[作者仓库 issue1](https://github.com/titizheng/GMMamba/issues/1) 同样询问缺失的 BiMamba；本次读取未找到能恢复该依赖身份的作者说明。没有向作者发送消息。
