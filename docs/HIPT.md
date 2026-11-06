# HIPT：原生预训练层级分类

接入官方 [mahmoodlab/HIPT](https://github.com/mahmoodlab/HIPT)，固定提交 `780fafaed2e5b112bc1ed6e78852af1fe6714342`。使用外部 Git 对象中的完整 `HIPT_LGP_FC`、ViT4K、门控注意力和作者优化器定义。作者源码、预训练权重均不随本库分发；许可证是 Apache-2.0 **附 Commons Clause**，不能按普通 Apache-2.0 理解。

| 配置 | 原生局部 ViT | 训练范围 |
|---|---|---|
| `hipt_lgp_frozen_source`（`HIPT` 默认） | 加载实际公开教师并冻结 | 全局 Transformer、池化和分类头 |
| `hipt_lgp_finetune_source` | 加载同一教师并微调 | 局部与全局网络 |

当前实现是原生层级 **分类**。固定作者树仅提供生存结果、划分和评估材料，没有对应完整 Python 生存训练器；本接入未用通用 Cox 头替代。自监督预训练、RGB 特征提取也不是这两条下游训练配置的功能。

## 输入与权重

每张切片是 `[R,256,384]`：R 个物理区域，每个区域完整包含 16×16 个 patch。384 维 token 必须来自配套 HIPT ViT256 DINO 编码器；另一个 FM 即使也是 384 维，也不能冒充同一预训练空间。其他 FM 的层级迁移需要另行命名和验证。

主特征使用普通任务数据包的 `.h5`，包含 `features` `[R*256,384]` 和真实 `coords` `[R*256,2]`。附加 H5 包含 `region_ids` `[R*256]`（对应主特征的每一行）和 `region_origins` `[R,2]`（每个区域左上角）。所有实际坐标必须组成声明步长的完整 16×16 网格；输入行可乱序，读取器按真实 y/x 坐标恢复作者序列。缺格、重复格、伪网格、跨区域错配均拒绝。任务采用单张 WSI 的 slide 分类训练，患者拆分仍由统一协议控制。

`encoders.<FM>` 必须包含：

```yaml
dimension: 384
embedding_space: HIPT_ViT256_DINO
weights_parent_sha256: 6960cd5a8657dc8bb214671aa0c6dbd3f5b698e84386884955836487ddc89e24
spatial:
  unit: level0_pixels
  origin: top_left
  # 其余空间字段按实际数据填写，见 DATA.md。
```

该 SHA 是作者 `HIPT_4K/Checkpoints/vit256_small_dino.pth` 的原始父文件身份。用户须实际准备对应编码器的特征及预处理来源；声明字段自身不证明已经完成真实提取。模型初始化使用作者 `vit4k_xs_dino.pth` 的教师骨干，原始完整文件 SHA256 为 `2b0bd9e9a602a35f2bb3f76da39d2b53a91f23fc3f115dc59a63267d95ad2b7b`。安全导出后仅保留 78 个骨干张量，稳定张量映射摘要为 `c4a5c7b076768d6180f940070df225e43edc3ab70c85f5c1d227041d1b72793b`。

```bash
milplus setup --method HIPT
# 从作者固定版本的 Checkpoints 路径单独取得真实公开权重后：
milplus prepare-hipt-weights --checkpoint /path/to/vit4k_xs_dino.pth \
  --output /path/to/hipt_teacher.pt
```

加载始终使用 `weights_only=True`。只有完整原始 SHA 匹配后才在独立解释器中启用旧 NumPy 数值标量的有限安全解码，返回纯张量；调用进程已有的反序列化白名单及其顺序不变，也不启用任意 pickle 执行。这样避开 PyTorch 2.5 临时 `safe_globals` 上下文移除已有同名项的问题。便携教师仍需逐张量摘要匹配，不接受随机权重或仅改元数据的替代物。源码获取采用稀疏规则，不自动拉取作者仓库数 GB 的 LFS 数据。

## 准备真实物理金字塔

创建 `pyramids.csv`，严格只有 `slide_id,path` 两列，每张任务切片恰好对应一个附加 H5。相对路径基于该 CSV 所在目录。创建 `hipt_schema.json`：

```json
{
  "schema_version": 1,
  "kind": "hipt_complete_16x16_pyramid",
  "source_commit": "780fafaed2e5b112bc1ed6e78852af1fe6714342",
  "synthetic": false,
  "grid_shape": [16, 16],
  "coordinate_frame": "level0_pixels",
  "patch_stride": 512,
  "patch_encoder": "HIPT_ViT256_DINO",
  "patch_encoder_weights_sha256": "6960cd5a8657dc8bb214671aa0c6dbd3f5b698e84386884955836487ddc89e24",
  "patching_provenance": "替换为实际切块配置、倍率/MPP和坐标来源",
  "patch_encoder_provenance": "替换为实际编码器版本与权重记录",
  "preprocessing_provenance": "替换为实际缩放、裁剪、归一化记录"
}
```

这里的 512 是示例 level0 步长，必须替换为实际值。原生输入 patch 是提取倍率下的 256×256 像素；level0 步长随倍率变化。预训练位置表含 197 个 token，来源于 14×14 预训练裁剪；作者 ViT 会将位置编码插值到实际 16×16 区域，这不表示下游应只提供 196 个 patch。

0.132.0另支持混合扫描倍率的显式schema2：删除上述`patch_stride`字段，改为
`"schema_version": 2`与`"patch_stride_by_slide": {"WSI_40X": 512, "WSI_20X": 256}`。
示例ID/数值须换成实际记录；映射必须恰好覆盖任务全部slide ID，不得缺项、多项，
或同时保留全局stride。level0步长须为正整数。manifest外层与内层均登记版本2，
旧schema1及其单一真实步长保持兼容。
它只按逐张真实坐标恢复同一个完整16×16序列，不推断倍率、重采样特征、改写坐标或改变模型。

```bash
milplus prepare-hipt --data /path/to/task --fm hipt_vit256 \
  --index /path/to/pyramids.csv --schema /path/to/hipt_schema.json \
  --weights /path/to/hipt_teacher.pt --output /path/to/task/hipt.json
```

随后在 `task.yaml` 的相应 encoder 中登记 `hipt: {manifest: hipt.json}`。准备工具绑定源特征、区域文件、几何、编码器声明和权重 SHA；训练、评估和续训重新检查对应身份，改变内容不能沿用旧记录。

```bash
milplus check --data /path/to/task --method HIPT --deep
milplus run --data /path/to/task --method hipt_lgp_frozen_source \
  --config configs/hipt.yaml --output runs/hipt_native
milplus evaluate --run runs/hipt_native
```

## 保留的完整训练流程

作者分类入口默认 **20 轮**，Adam `lr=2e-4`、`weight_decay=1e-5`；batch size 1。训练集 N/类别数量的逆频率替换抽样，保留 DataLoader 随机种子消耗顺序。作者循环对 CE 除以 32，**每个 bag 都执行 Adam 更新**，不是积累 32 个 bag 再更新。没有调度器或梯度裁剪；需要完整区域，不允许统一接口随机截断 patch。

默认作者入口未开启 early stopping，最终轮被选中。对应 `best.pt` 表示该协议的已选最终权重，不宣称验证指标最优。配置 `model.early_stopping: true` 才按验证集逐 slide CE、最后并列最优选模；默认 patience 20，保留零起算 epoch>50 后才停止的作者边界。测试集始终在选模后使用；这比原入口每轮查看 TEST 更严格，已作为评测边界说明。

核心网络完整保留六层 192 维局部 ViT、两层 192 维全局 Transformer、门控池化和分类器。加载真实预训练骨干后严格检查全部键；原作者的相对路径、`strict=False`、无预训练随机分支没有成为静默回退。模型、Adam、抽样、Dropout 和全部 RNG 支持续训；全局患者评测输出不改变作者逐 slide 验证损失。

实例导出保留真实坐标、区域来源、每头局部 CLS 注意力、全局区域注意力及恢复原行顺序的层级乘积。局部 CLS 自身注意力质量没有重新归一化消除；乘积不是完整 Transformer attention rollout，也不解释为因果证据。

## 验证与边界

`tests/test_hipt.py` 包括实际公开预训练骨干、独立作者加载器与前向、两条作者平衡训练循环的 108 次完整更新逐位对照、三分类梯度、完整续训、选模边界、物理输入拒绝和 CLI。专项初次运行 28 项通过；正式全库状态以 `validation/summary.json` 为准。

`scripts/validate_hipt.py --output 新目录` 执行两条原生 20 轮配置；默认权重位置可通过 `MILPLUS_HIPT_WEIGHTS` 指定。另有明确标记的合成演示：

```bash
milplus demo --hipt --backbone-weights /path/to/hipt_teacher.pt --output /tmp/hipt_demo
```

合成数据使用高斯 patch token 和明确物理网格，只检验软件链路；真实教师权重不会使合成 token 变成真实 DINO 特征。GPU、真实 WSI、真实特征提取、原文队列数值复现仍需分别运行。当前分类接入不等于论文全部自监督与生存实验已复现。

## BRCA 原始 ViT256 教师资产实际补齐（2026-11-06）

固定官方Git LFS对象中的ViT256原文件为704238867字节，SHA即上文`6960cd5a...`。
计算节点直接联网超时的失败保留；本地用系统curl下载完整原文件后，
以42个有界块传入现有GPU作业，每块实际目标SHA及最终完整SHA均通过，不新增作业。
传输初版在stdin EOF时丢失远端完成观测，且部分块不完整；只修正传输握手，
保持stdin开放直至远端定长读取/写入确认，已有正确块不重传，不以退出码代替实际字节核验。

22:18Z，专用CPU子进程在完整父文件SHA核对后，以`weights_only=True`和有限旧数值元数据
白名单读取，未启用任意pickle。完整150个骨干张量严格匹配作者ViT-small，
稳定张量映射SHA为`317e897991c8d53d17f0dc26a692da3459f32aea98a9dabbacded1df423f6a70`。
作者原始`get_vit256`函数也原样执行，仅将其不安全load调用绑定到已安全解码的同一父文件；
该函数报告零missing keys，额外8项仅为原DINO预训练head，实际推理骨干无缺失。

真实首个BRCA袋的首/中/末三张256RGB，原作者`ToTensor/Normalize(.5,.5)`与直接张量处理逐位一致。
严格加载、原作者factory和便携权重重新加载的三路384D前向逐位相同；没有resize/crop或GPU计算。
便携文件86711760字节，SHA为`336a19938028fbecbdae6c297d8605853512c83c01cf38369f3954bf8d1e7e65`，
另有共享目录私有资产备份；不是把权重打包再分发进本库。
原有ViT4K完整父文件和78张量便携教师也重新核对通过。

凭据位于共享BRCA目录的`hipt_vit256_local_transport_verified_r1.json`、
`hipt_vit256_native_teacher_real_rgb_verified_r1.json`及`hipt_vit4k_existing_teacher_reverified_r1.json`。
这仅补齐并核验真实教师资产：三个缓存patch来自原UNI的level1读取，
既不是完整16×16物理金字塔，也不是原论文倍率或全部856张特征提取。
HIPT仍blocked于原WSI访问和真实区域准备，不能将UNI1024转换/凑256行冒充HIPT384原生层级输入。

22:43Z后续访问进展：原856张的GDC官方UUID/名称/大小/患者/项目/open元数据已全部对应，
本地仅下载一张原TRAIN WSI并将全部169个原RGB patch与此前GPFS交付像素SHA完整对齐。
当前本地读取路线无需新增CPU作业即可继续探索真实区域准备；但尚无完整16×16金字塔或HIPT特征。
不能把这一后续来源进展误称为上文已完成了全856处理。凭据见
`gdc_original_wsi_local_access_route_r1.json`及BRCA研究文档最新章节。

## 真实20×物理输入与完整原生特征 pilot（2026-11-06）

23:21:50Z，以同一张GDC原TRAIN切片完整执行作者组织/区域路径，得到4个真实4096×4096区域，
共1024张256×256patch。源WSI为40×，实际level0读取8192后按原PIL整区resize到4096，
真实level0 patch步长512；不是把原169行UNI特征reshape、复制或取整成网格。
四个区域均在真实WSI边界内，1024坐标唯一且逐区域完整覆盖16×16。

使用HIPT固定Git对象中的`WholeSlideImage`、区域生成/QC及完整`HIPT_4K.forward_asset_dict`。
组织参数明确取固定CLAM提交`53e2409d...`的默认seg_and_patch字面值：
sthresh8/mthresh7/close4，a_t100/a_h16/max_holes8，four_pt与原use_padding=True；
这些是显式的固定默认输入条件，不冒称取得作者历史region列表或所有历史预处理配置。
真实读取位置独立记录为level0坐标，不错误地把源generator保存的custom_downsample坐标当level0。

两个真实预训练教师严格加载，冻结FP32推理；每区保持作者完整batch256，实际hook记录四次
`[256,3,256,256]`，没有缩小模型、区域或batch。
原作者独立FP读取器与直接原始区域读取的全部像素SHA及归一化张量逐位相同。
输出完整384Dpatch与原192Dregion CLS；原`aligned_inputs`在真实1024行上通过。
独立复核重新检查源代码SHA、全部输出SHA、dtype/finite、所有物理坐标/区域顺序及无越界，
并将1685955字节结果保存到共享`hipt_native_real_physical_pilot_evidence_r7/`，未重跑推理。

执行在已有H200作业的CPU子进程，CUDA仅在该子进程隐藏；原13个训练未收到信号。
记录自身峰值RSS2920132KiB，先核对至少16GiB实际主机余量，并设置额外地址空间保护。
本地早期读取/转换多次被137终止，原因未证明；分块I/O尝试也未通过，均保留，未冒充等价输入。
最终成功路线使用现有节点的原样one-shot区域读取，而非采用那些未验证的分块缓存。
初始节点缺openslide的导入失败及随后NumPy整数日志序列化失败均保留。
后者仅将日志字段改为已保存的Python整数，3项控制验证传入原生reader的参数对象/数值不变。
OpenCV/OpenSlide均在独立依赖前缀，原活动代码、runtime2及既有依赖未改。

本输入组为`native_HIPT_ViT256_DINO20x_CLAM_default`，不是原UNI主组，也不是20epoch临床训练。
教师来自TCGA预训练，不能据此宣称与当前队列的预训练患者重叠已经排除。
全856真实输入、完整公共task契约及MIL训练仍待完成；不能把一个真实pilot计作HIPT完成。
凭据：`hipt_original_wsi_physical_pilot_verified_r7.json`及
`hipt_native_real_physical_pilot_evidence_r7/independent_audit.json`。

## 完整856原队列的连续输入准备（2026-11-06）

23:51:35Z，通用逐切片worker完成第二张原TRAIN切片TCGA-AC-A62V：
官方原文件57528662字节，MD5为`f65cbedaaa080e60a7da6a4c7b29d20f`，
本地及现有节点SHA256均为`9c4387e3d4ff1b04287aa1a35369f66cc434bd231c53cf586bcffc3900388131`。
完整305行旧坐标RGB重新读取后与原GPFS像素摘要一致；实际40×、MPP0.252。
原组织/QC流程得到7个完整20×区域、1792行384D及7行192D特征。
23:51:58Z独立复核全部输出SHA、全部坐标、原生批次、区域顺序及边界通过。
连同R7，启动完整流时累计2张、11区域、2816行，不能当作全队列或训练完成。

新worker只泛化WSI身份、输出路径和实际20/40×倍率绑定，保留原4096区域、
组织筛选、完整batch256及原教师前向。21项保真控制在现有节点通过；
20×分支的几何控制不等于已有20×扫描切片的完整实际提取证据。
逐张记录MPP和真实level0步长；未知倍率、无组织区域、重复坐标、源像素不同均报错，
不会猜尺度、凑格子、删除患者或降低模型/batch。

23:53Z，既有节点CPU连续准备流程已启动，计划覆盖严格相同856患者，
通常按原文件大小/slide ID排序，仅为有界IO顺序；若已有worker存活则先接管其观测。
不新增rjob、不成为第二个MIL训练调度器，也不改现有训练源码。
每张先核官方MD5/完整传输SHA，再核全部原RGB像素，完整提取后独立审核所有输出。
仅对本流程新建的私有原图spool，在核验通过后再次校验SHA并删除；首张R7原图、
既有原图/真实RGB/UNI、所有特征和失败证据均不动。已退役副本可按相同官方UUID重取。
首轮实际删除本地与节点各57528662字节，并保留逐路径摘要收据。
5项队列/完整人口/安全目标检查含12个拒绝删除目标的子例，未执行宽目录清理。

全队列真实文件、实际任务manifest、MIL训练及当前发行包验收仍分别待完成。
混合扫描倍率须按逐切片真实步长完成契约核查，不能强填单个512。
凭据为共享目录`native_hipt_full856_preparation_r1.json`、
`native_hipt_full856_stream_plan_r1.json`、`native_hipt_full856_audits_r1/`、
`native_hipt_full856_retirements_r1/`及`hipt_full856_stream_controller_launch_r1.json`。

## 混合扫描倍率契约的实际校验与安装包（0.132.0）

第三张原图TCGA-MS-A51U实际扫描倍率为20×，其完整13区域/3328行已独立核验，
真实level0步长256；前两张40×源图的20×提取步长为512。
因此不能给整个真实队列统一填512。新schema2以逐切片真实stride解决输入表达，
不缩小队列、不转换原坐标，也不改原教师或下游模型。

开发环境完整测试曾137退出，原因未证明，未记通过；其27项基础检查单独通过。
在现有节点独立CPU源码副本中，原31项加新32项共63项全部通过、零跳过，
含原两个模型108次作者原生训练更新逐位对照、旧格式兼容、混合格式、无伪造网格、
两模型精确续训、选中检查点回读和CLI。进程自身峰值RSS2236436480字节，
不改变原活跃训练代码或调度器。

实际3张、24区域、6144行又通过新读取器：逐区域特征值、坐标和行序未改，
错误统一步长被拒绝。该接口检查只读取当前已完成输入，未创建3患者临床子任务；
正式队列仍为856，完整task manifest、20epoch临床训练及全方法验收仍待完成。
与父0.131.0包比较，512个包内文件仅`__init__.py`版本和`hipt_data.py`不同，
`hipt.py`、教师、网络、训练器和415配置原字节不变。

0.132.0独立wheel已离线安装到私有target，512包文件与当前源码逐字节一致，
模块CLI及全部32项混合几何测试通过、零跳过；全部milplus模块导入路径确认为该安装目标。
这是借用既有依赖的离线安装检查，不是全新依赖解析或全库4955/5053回归。
wheel SHA256为`d50f74a9a119b9ead470eeef3e889557db0861d45847518a9827b8d671efa679`。
共享凭据：`hipt_mixed_geometry_all_actual_receipt_r1.json`、
`hipt_real_mixed_stride_prefix_verified_r1.json`及
`hipt_mixed_geometry_wheel_r1/installed_wheel_actual_receipt_r1.json`。

## 原生完整batch256的CUDA容量与连续准备交接

2026-11-06T00:24:24Z，使用已核验原TRAIN切片的一个完整真实4096区域，
原`HIPT_4K.forward_asset_dict`保持256×3×256×256 patch batch和完整两个教师，
仅将两个原公开device参数都绑定到现有单H200的cuda:0。FP32、禁用TF32/AMP，
没有缩小模型、batch或区域。前向耗时1.3888秒；原生分配峰值1953059328字节，
reserved峰值2344615936字节。它是一完整区域的实际容量探测，不是整个WSI耗时、
全856完成或MIL训练峰值。

与同区域CPU产物按预设rtol/atol均1e-4比较通过，不宣称跨硬件逐位一致；
patch最大绝对差约1.111e-4，按组合绝对/相对容差判定。
同CUDA设备的独立原生patch→region路径逐位一致。
初版观测失败保留：原作者直接调用model4k.forward，普通module hook没有触发；
R2只包裹诊断调用以原对象传递参数、返回原输出，3项控制通过，不改变科学前向。

新CUDA逐切片worker通过25项保真/资源准入控制，队列通过10项控制，
包括保留完整已验前缀、所有856患者、原安全清理函数，以及生成的远程启动器编译。
启动与实际GPU分配前均核对未来活跃预约；额外主机地址空间12GiB、主机guard8GiB，
GPU allocator上限4GiB，且保留40GiB GPU guard并扣除全部活跃队列GPU预约。
余量不足等同一进程，不拆batch、不重跑部分输入、不用占用程序填显存。

00:32:49Z仅向原本地输入控制器发送SIGTERM，旧6张已验产物原样接管，
当时仍存活的CPU worker666124与12个训练PID均保留。新控制器先等该CPU切片完整结束，
再按上述准入启用CUDA后续输入；MIL队列owner、rjob和现有训练源码均未改。
权重/输入空间仍为同一原HIPT DINO，逐文件runtime明确CPU或CUDA，不混称UNI结果。
此时全856输入、正式任务注册及后续MIL训练仍未完成。

凭据：`hipt_full256_cuda_capacity_verified_r2.json`、
`hipt_direct_forward_observation_controls_actual_r2.json`、
`hipt_full_cohort_worker_cuda_controls_actual_r2.json`、
`hipt_full856_stream_cuda_safety_controls_local_r2.json`、
`hipt_full856_stream_controller_launch_cuda_r2.json`。
实时状态继续使用`native_hipt_full856_preparation_r1.json`，已记录新控制器身份，
不能再用历史R1启动文件的旧PID判断当前控制器是否运行。

00:41Z后第一张新CUDA整张原图TCGA-AN-A0XW已完整交付并通过独立R2审计：
原153个既有RGB patch像素核验、真实组织/QC与一个完整4K区域，
256行384D及原192D region输出，GPU分配峰值1953059328字节、reserved2344615936字节。
这不是把153行UNI凑成256行，而是从同一原WSI另行执行原生区域提取。
后续CUDA切片继续走同一完整原输入流，已验旧CPU文件不重算。
逐张新收据的runtime明确cuda:0/FP32，CPU与CUDA数值差异不被隐瞒为逐位一致。
逐张审计仍不等于全856完成或HIPT的20epoch下游训练完成。

后续原WSI流经公开AWS完整原字节恢复并增至19/856；每图仍须原GDC MD5、
local/remote SHA和原生输出独立审计。已有分块发生单块TLS超时时只补缺块，
不会重新下载或重新计算已完成患者。32项检查的R4恢复保护只改变IO路径/绑定与
实际终态后的分块恢复函数，所有原worker/准入/审核/退休函数保持不变。
一次性观察器跟踪原控制器PID+starttime；原控制器仍活着时绝不交接，
也不因观测超时、模型失败或全856成功结束而启动第二个控制器。
当前保护处于待命而非已切换，实际状态与证据详见
[BRCA输入流记录](BRCA_UNI_IDC_ILC_20261105.md#原图失败块精确恢复旧roi补齐及一次性保护)。

