# SAM-MIL: source control and explicit objective repair

Source: [FangHeng/SAM-MIL](https://github.com/FangHeng/SAM-MIL),
`ccff640825321a12658daa0a85c03390bc9d367b`, Apache-2.0. This entry is
the ACM MM2024 SAM-MIL attention route. It includes real SAM group membership,
group mean tokens, area-sensitive masking, five random pseudo-bags, all loss
terms, the literal fixed teacher, Adam, cosine scheduling, validation selection
and complete recovery. It is not a generic ABMIL renamed SAM-MIL.

Two execution profiles are deliberately distinct:

| Profile | Behavior |
| --- | --- |
| `sammil_attn_source_control` | Binary released-code control, including documented mask-index and loss-wiring behavior |
| `sammil_attn_objective_repaired` | Explicit global-classification, student-consistency and group-index corrections, with C-way validation |

`SAM-MIL` resolves to the explicitly named repaired profile. Neither profile
claims reproduction of historical clinical/paper results. Current software
fixtures remain permanently marked synthetic. Full-suite, independent-wheel,
native-budget and actual WSI validation are separate evidence requirements.

## Input preparation

The primary bundle stores **ordinary patch features only**, with unique actual
level0 top-left coordinates and their existing FM identity. SAM supplies extra
information; its checkpoint identity and segmentation/membership provenance
must accompany the task. Changing FM never turns synthetic regions into SAM
regions and never creates coordinates by reshaping a sequence.

Each indexed HDF5 membership archive contains:

```text
coords                 [N,2], same original order as primary patch features
groups/seg_0           [K0,2] actual member coordinates, attribute area=int
groups/seg_1           [K1,2] actual member coordinates, attribute area=int
...
```

`area` is the original SAM mask's pixel area, not a group-size count or
normalized fraction. Membership may overlap. The adapter retains every patch,
appends the arithmetic mean feature of each nonempty mask, preserves the
author's `[-1,-1]` group-token sentinel, assigns the minimum covering mask area
to ordinary patches, and uses sentinel1 for uncovered patches/sentinel0 for
group tokens. It rejects missing, duplicated or unknown coordinates and empty
or invented regions. Mean-token construction uses float32 frozen features,
as in the released feature pipeline; it does not retrain or replace the FM.

The original pipeline exports membership coordinates in `seg_files` and stores
mask areas separately in SAM segmentation metadata. Assemble the archive above
from those actual exports, retaining the original mask/weights provenance. No
untrusted pickle is executed by this importer. A task with only ordinary
TRIDENT features still needs genuine SAM segmentation memberships.

An index CSV contains exactly `slide_id,path` once for every registered slide.
The schema JSON contains:

```json
{
  "schema_version": 1,
  "kind": "sammil_measured_region_membership",
  "coordinate_frame": "level0_top_left_pixels",
  "group_feature": "arithmetic_mean_of_registered_member_patch_features",
  "overlap": "minimum_original_mask_area",
  "sam_model": "vit_h",
  "sam_weights_sha256": "REPLACE_WITH_ACTUAL_64_CHARACTER_CHECKSUM",
  "segmentation_provenance": "actual checkpoint, SAM settings and mask export version",
  "membership_provenance": "actual coordinate-to-mask membership export",
  "feature_provenance": "actual FM weights and ordered patch feature extraction"
}
```

```bash
milplus setup --method SAM-MIL
milplus prepare-sammil --data TASK --index memberships.csv \
  --schema sam_schema.json --output sammil/manifest.json
```

Then set the selected encoder's `sammil.manifest` to
`sammil/manifest.json` in `task.yaml`. Manifest hashes bind the feature files,
membership archives, group identity/order, mask areas, source preparer,
patient identities, labels and fixed splits. Every read checks actual file
content. Precomputed inputs do not imply that this repository has executed SAM
or reproduced an FM's features on real WSI.

```bash
milplus run --data TASK --method SAM-MIL --output runs/sam
milplus evaluate --run runs/sam --subset test
```

## Complete source recipe and repairs

The README attention command uses a512-wide projection and bag embedding,
128-wide attention, dropout.25, ReLU, area sigmoid slope.0005/offset5000,
mask ratio.9 with the author's `random_ratio=.001`, five random pseudo-bags,
global weight1, pseudo-bag weight.5 and consistency weight1000. The released
feature-distillation implementation is retained; the published command's
`cl_alpha` default is0. Nothing silently sets the nonzero bag/consistency
coefficients to zero to make training pass.

Both routes retain the actual consistency kernel. For each instance it computes
the Euclidean norm of its difference vector to every member of the same area
group, then averages those per-instance norms, with its original
normalization and float32 accumulator. It is not pairwise mean-square error.
Group-token sentinel0 also participates exactly as in the source.

Source observations at the pinned commit:

1. `get_mask` indexes the batch coordinate from `torch.where([1,N])[0]`.
   The source-control profile preserves this, including repeated token IDs.
   Repair uses patch indices, compacts areas with their true patch indices,
   and maps retained/masked IDs back before prepending the actual group IDs.
2. `engine.train_loop` overwrites global logits and distillation values in its
   pseudo-bag loop. Main classification therefore uses the last pseudo-bag.
   Repair keeps the global branch for main CE/distillation while retaining
   every pseudo-bag CE and the same source call/random-draw order.
3. `compute_attn_with_grad` is called on the teacher for consistency, but the
   SAM branch neither optimizes nor EMA-updates that teacher. Source-control
   teacher gradients accumulate without weight updates; their complete bank
   is saved and restored. Repair computes consistency on student attention,
   retaining the source's detached input projection. No invented EMA stage
   or pretrained teacher is introduced.
4. The released validation assumes binary labels. Source control remains
   binary; repair explicitly extends validation to macro one-vs-rest AUROC.
   All tasks use disjoint registered validation and test patients. The source
   default `val_ratio=0` test-as-validation shortcut is not used.

Default training is the released200epoch cap, batch1 RandomSampler,
Adam2e-4/weight-decay1e-5 and200step cosine decay, without AMP, clipping or
patch truncation. `recipe=tcga` uses patience20 and zero-based stop_epoch70;
`recipe=camelyon16` uses patience30 and stop_epoch130. Early stopping resets
on exact AUC ties, while the published best-AUC save uses strict improvement,
therefore the first best-AUC checkpoint is selected. The teacher remains the
literal same-initialization deepcopy and uses train-mode dropout during the
source masking calls. All five pseudo-bags retain the remainder in the last
group. No test predictions are computed during fitting.

Every epoch saves source sampling order, actual Adam updates, cosine state and
LR history, student/teacher parameters, source-control accumulated teacher
gradients, source early-stop state and RNG. Independent evaluation reloads the
selected checkpoint. Short software tests are not evidence for the complete
native200max budget, and synthetic performance is not a clinical result.

## BRCA 原始 SAM ViT-H 权重补齐（2026-11-06）

从固定作者README明确指向的Meta官方HTTPS地址获取完整`sam_vit_h_4b8939.pth`，
2564550879字节，实测MD5为`4b8939a88964f0f4ff5f5b2642c598a6`，与文件名前缀相符。
完整SHA256实测为`a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`；
这是官方HTTPS下载后的本地测量，不冒称另有独立公开SHA来源。

初次本地CPU验证器以137终止，旧代码与失败凭据保留；当时没有阶段插桩，不能宣称已证明原因。
所观察本地cgroup上限15032385536字节、历史峰值15032389632，但failcnt与oom_kill均为0，
不能仅凭137编造已确认的内核OOM诊断。现有GPU训练完全不受信号。

R2仅改变资产加载时的存储分配：固定作者完整模型在meta设备构建，避免先分配一份巨大随机参数，
然后`weights_only=True,mmap=True`读取同一完整原权重，`strict=True,assign=True`加载。
原factory的六个非持久pixel mean/std值从原Git对象AST字面值读取并在CPU恢复；
最后全部参数和buffer都实际位于CPU，没有残留meta权重或替代小网络。
641090608参数、594个完整state张量全量有限且与已加载状态逐位一致；missing/unexpected均为0。
原32层/1280维/16头/1024输入保留，未执行图像前向、SAM分割或MIL训练。

原文件留在本地`/data/.../sam_vit_h_asset_cache_20261106_r1/sam_vit_h_4b8939.pth`，
canonical路径和初始下载路径是同一inode的两个名称，不新占第二份权重空间。
没有将2.56GB权重复制到接近满的共享卷，也未传入GPU节点或新增作业。
这是真实权重资产，不是可随意删除的过程文件；不是SAM区域已经准备完成的证据。

共享凭据为`sammil_original_sam_vit_h_local_asset_verified_r2.json`及
`sammil_original_sam_vit_h_canonical_asset_r1.json`；初次失败见
`sammil_original_asset_initial_verifier_exit137_r1.json`。
SAM-MIL仍须全856张实际WSI掩膜、原始mask像素面积及真实patch成员映射；
当前原WSI源任务已结束，新CPU读取任务尚未获准，不能用拼接缓存RGB或人工区域冒充。

后续22:43Z已找到无需新增作业的官方原WSI本地读取路线：全856原文件元数据对应，
一张原TRAIN WSI完整169patch像素SHA与GPFS源交付一致。上述原任务结束不再意味着只能
等待新CPU作业；仍须逐文件校验并按原分割配方生成实际掩膜/面积/成员，当前没有新增分割结果。
凭据`gdc_original_wsi_local_access_route_r1.json`，详见BRCA研究文档最新访问进展。

## 同一原始权重在现有节点实际到位与严格加载

2026-11-06T00:36:42Z，同一完整2564550879字节原checkpoint经153个16MiB有界块
传入现有作业的私有权重目录。目标每块及最终完整SHA256实际核对，和上述原始下载一致。
没有在接近满的共享卷上放一份大权重，也不新增rjob；本地canonical资产保留。

00:37:52Z，现有节点CPU专用子进程再次执行完整原模型严格加载。
GPU仅在该子进程隐藏；原32层/1280维/16头/1024输入、641090608参数、594个张量
和全部原持久/非持久buffer均在CPU实际物化，无meta残留。
完整checkpoint SHA/MD5、8份原源码SHA、稳定tensor映射SHA均与本地先前核验一致，
加载值逐位一致，missing/unexpected为0。稳定映射SHA仍为
`746ad5c0e42faf944ac0089a9fc3af0b633b7346ece3ca7ab0c1d82b96439093`。
使用weights_only/mmap及meta构建后strict assign，未少加载任何层或改为小网络。
自身峰值RSS2988396544字节，启动前扣除活跃主机预约并保留资源保护。

节点真实路径：`/root/milplus_brca_uni_20261105/native_sammil_sam_vit_h_assets_20261106_r1/sam_vit_h_4b8939.pth`。
这是已到位、可读取的真实资产，不是GPU图像前向、SAM分割、全856掩膜/原面积/成员映射，
也不是SAM-MIL训练完成；这些阶段仍须按原处理代码与真实坐标完成。
凭据：`sammil_original_sam_vit_h_existing_node_transport_verified_r1.json`及
`sammil_original_sam_vit_h_existing_node_asset_verified_r1.json`。

## 原始真实 ROI、完整分割预算与 UNI 成员几何

2026-11-06T00:57:04Z，TCGA-AC-A23G 的同一原始 TRAIN WSI 已通过作者的
TCGA 组织筛选（a_t16/a_h4，非 HIPT 的100/16）读取完整 SAM ROI。
实际 RGB 为527×566，原始整函数在 generate 边界的纯输入记录，与未改写源码
前缀独立读取的全部像素逐位相同；两条路径均没有产生伪掩膜。
RGB SHA 为`6f20e5653434f4652bc1f2d166eff253bf96e72e79b784a044cc6cf9f4907fb7`。
凭据`sammil_original_wsi_roi_pilot_verified_r1.json`。

完整原分割 actor 已在同一节点启动。保留 ViT-H、32×32/16×16 点网格、
每批64点、一层裁剪，即5次完整1024图像编码及2048点/32个解码批次。
8GiB自身 allocator 预算之外保留40GiB GPU guard，并扣除活跃训练的声明预约；
等待资源不是减小模型、点数、batch或裁剪层数。01:15Z仍在该原进程中等待，
尚未产生分割结果；不能将启动收据当成掩膜或完整方法完成。
凭据`sammil_original_full_budget_mask_pilot_launch_r1.json`及对应实时 progress。

01:15:46Z，CPU 实地预检核对了同一袋169×1024原始 UNI 特征和169行原坐标。
现有 UNI patch 是 level1/256，登记的整数 CLAM 视野为1024个 level0 像素；
不是作者原命令的512。原图实际小数 downsample4.0002157497仍保留，不能称为精确4。
原作者成员映射使用`int(level_downsamples[vis_level])*downsample_ratio`，
此片得到32，因此显式注册视野对应的 adjusted_patch_size 为32。
保留源码的 numpy round、patch左上角 bbox 门控及 any(mask slice)，
不静默改成另一个重叠定义或把原mask像素面积换成patch数量。
这是明确的原 UNI 输入条件，不声称原 ResNet/512px 论文预处理复现。
凭据`sammil_original_membership_geometry_verified_r1.json`。

独立 CPU 后继已启动，只等待上述同一 mask actor 的真实成功输出，不启动第二个 SAM。
它会先核对全部 mask 像素、面积、摘要和实际完整点预算，在私有复制特征上运行
固定作者原 membership/mean-token/process_info 函数，再和实际 importer 的
features、coords、group flags、原始面积逐位比对。失败保持失败，不制造区域或跳过患者。
后继启动凭据`sammil_original_membership_pilot_launch_r1.json`；
只有成功生成`sammil_original_membership_pilot_verified_r1.json`才可称该输入试点通过。
当前不建立单患者临床任务、不修改原特征/标签/划分，不放行全856任务或报告 MIL 分数。

## 复用现有原图流保存 SAM 原生 ROI，避免后续重复下载

2026-11-06T01:28Z，独立CPU读取进程加入现有节点，不改任何HIPT控制器、worker或训练。
仅观察已通过官方MD5和传输SHA核验的真实原WSI，在自身私有目录创建临时硬链接租约。
即使HIPT随后删除其自己的暂存名称，读取仍可完成；不移动、不写入原WSI，也不新增下载。
固定作者TCGA16/4组织轮廓、原整函数generate输入边界和独立源码前缀分别读取同一真实ROI，
全部像素相同并回读`.npy`后，才删除自己的私有硬链接名称。失败保留自己的租约和证据。
真实RGB及轮廓保存在现有节点本地盘，只把小收据放入共享目录。

01:29:46Z独立核对首张实际原TRAIN切片：通用读取器与前一独立pilot的全部RGB像素、
组织/孔洞数组、ROI参数、原preset及源码SHA均一致；源WSI仍存在且完整SHA未变，
仅本进程私有租约名称被删除。没有新增SAM模型、GPU前向或伪mask。

此进程在后续HIPT原图流到达时继续保存真实ROI，状态为
`native_sammil_original_roi_preparation_r1.json`，逐图收据在
`native_sammil_original_rois_delivered_r1/`。当前只实证一张通用ROI；
启动之前已经被HIPT清除暂存且没有保留副本的原图仍缺ROI，不能把它们计作完成。
即使以后全856 ROI齐全，也仍不等于全856真实mask/原面积/成员或SAM-MIL训练完成。
启动与独立审计凭据分别为`sammil_original_roi_existing_stream_capture_launch_r1.json`、
`sammil_existing_stream_generalized_roi_pilot_independent_audit_r1.json`。

## 完整原生真实 SAM 掩膜和成员试点通过

01:30:35Z，首版完整模型加载之后在作者predictor的绝对导入
`from segment_anything.modeling import Sam`处失败，这是私有命名空间导入缺口，
不是模型容量问题。失败及其CPU后继终止证据保留。R2仅在导入原始Git源码时暂时
绑定同一个完整Sam类，成功/失败均恢复别名；原四份运行模块字节不改。
9项实际节点检查通过：原6项完整预算/资源控制及3项实际类型/别名清理控制。
测试初版AST边界错误亦保留，不冒充通过。

2026-11-06T01:35:50Z，R2完成同一真实TRAIN ROI的全部原生预算：
5次`[1,3,1024,1024]`FP32编码、32批每批64点、共2048点，
原32/16点网格、一层裁剪及全部筛选参数保留，共生成127个真实二值mask。
完整原模型/权重不变，无AMP/TF32，GPU分配峰值6042877952字节、
reserved峰值6606028800字节。分割用时49.63秒，不代表整队列耗时。

01:36:19Z，独立CPU后继核对全部127个mask的原像素面积、SHA和实际点预算，
在原169×1024 UNI袋私有副本上执行固定作者成员函数、group mean和process_info。
保留显式1024-level0视野、整数映射32及原bbox/round，得到33个非空原mask组、
202行增强特征。与实际`sammil_data.read_groups`的全部features、coords、
group flags和原mask面积逐位相同；未覆盖patch按原sentinel处理。
没有造区域、补实例、改原特征/标签/划分或建立单患者临床任务。

实际mask/面积/成员/增强特征/ROI及小收据共2571863字节，原字节保存在
`sammil_real_native_pilot_evidence_r2/`。成功凭据
`sammil_original_full_budget_mask_pilot_verified_r2.json`及
`sammil_original_membership_pilot_verified_r2.json`；旧R1失败保留。
这只证明一张真实输入试点，不是全856输入齐备、SAM-MIL训练或论文结果复现。

## 全856原输入流水线已启动，原生计算保持不变

2026-11-06T01:54Z，完整856患者/2422295原UNI行的独立输入流水线已启动。
仅消费上述真实原ROI，不新建临床子任务或MIL调度器。一次一个GPU分割子进程，
随后独立CPU成员处理；每张都保持原ViT-H、全部2048点（32批×每批64点）和原面积/均值定义，
并在原8GiB allocator cap、40GiB GPU guard及活跃预约扣除规则下准入。
仅把固定pilot路径/169行假设换成真实每图索引，TRAIN/VAL/TEST按原清单登记；
不会把后续VAL/TEST文件错误写成TRAIN，也不会因为输入缺失删除患者。

16项实际节点保真/资源/接口控制通过，包含原完整导入与资源9项、原生成/无损存储AST
逐语句对照、原成员函数调用对照、完整856不筛人计划、真实PID复用防护等。
初版一项测试选择器因AST tuple括号格式失败，证据保留；仅修测试选择器后16项全部通过，
科学worker代码没有随之变化。

01:57:27Z，通用worker对第一张真实原ROI重新执行全部原生预算，并独立复核：
127个mask的全部像素和原metadata逐位相同，69份成员/增强特征数组逐位相同。
仍为同一患者，不作为第2个已完成病例。流水线已验1/856、169原行，余下病例等待真实ROI。
完整856收据齐备前不注册/训练SAM-MIL；此前已退休暂存的原图还需补取，不能自动补造ROI。

源绑定`sammil_full856_pipeline_source_bindings_r1.json`，启动
`sammil_full856_original_pipeline_launch_r1.json`，实时状态
`native_sammil_full856_preparation_r1.json`。逐图mask/member/audit目录分别为
`native_sammil_full856_masks_delivered_r1/`、`native_sammil_full856_members_delivered_r1/`、
`native_sammil_full856_audits_r1/`。完整首图原生对照凭据
`sammil_full856_worker_real_pilot_equivalence_r1.json`。

恢复原WSI流后，第二个不同患者TCGA-AC-A8OS的实际879原UNI行也已完成：
221个真mask、135个非空组、1014增强行；原面积/均值/flags/coords与作者函数逐位一致。
其原整数映射因子为64、adjusted_patch_size16，保留真实几何而非固定为首图32/32。
当前全队列流实证2/856、1048原行、348个实际mask；不将试点重放重复计数，
也不据此宣称全856或SAM-MIL训练完成。

## 已退休原图的15个真实ROI缺口补齐

2026-11-06，增加独立的输入补齐控制器，使用恢复AWS流之前固定的16图完成前缀，
减去已交付ROI，实际得到15个缺口（原WSI合计1467483990字节）。这是输入修复清单，
不是15人临床训练子集；完整856患者及2422295原行的任务范围不变。
使用固定前缀而不是正在增长的HIPT状态，避免与主ROI读取器抢写新完成病例。

原WSI必须同时通过GDC原MD5和此前独立HIPT运输的完整SHA；只暂存于新的私有
`sammil_original_backfill_wsi_cache_r1`，不重建已经退休的HIPT暂存名称或改其tombstone。
运输沿用原16MiB/双worker代码的AST语句，唯一差异为目的路径。
节点CPU包装器导入并直接调用已冻结原ROI读取器的`native_reader`和`capture`函数，
保留同一科学源码SHA和整函数/独立前缀逐像素核对；不改组织preset、ROI或SAM预算。
完成独立ROI审计后才退休新私有运输副本，原数据、旧失败分块和训练检查点均保留。

12项实际控制通过，绑定与启动证据为
`sammil_original_backfill_source_bindings_r1.json`、
`sammil_original_retired_prefix_backfill_launch_r1.json`，实时状态为
`native_sammil_original_retired_prefix_backfill_r1.json`。
现有完整SAM流水线自动消费同一ROI证明目录，不创建第二个GPU队列或新rjob。
主ROI观察器的计数只统计自己拥有的病例；多个不相交生产者的实际覆盖应以
`native_sammil_original_rois_delivered_r1/`唯一key并集和独立输出校验为准。
启动不等于15图完成，亦不等于全856输入或SAM-MIL训练完成。

后续R1在首个57528662字节原图的55块中54块通过、1块达到运输时限而实际退出，
原进程与失败证据保留，不把超时观测误当终止。R2只改变缓存运输恢复函数；
其他所有控制函数AST及节点原ROI驱动保持不变。20项组合检查通过。
已有成功块逐块重验SHA，缺块最多三次独立尝试，每次保留自己的失败文件；
若原命名已含失败字节，先以私有硬链接保全全部原字节并记录SHA/收据，
再替换该私有缓存别名，绝不覆盖原始数据。完整原GDC MD5及旧HIPT SHA同时通过后
才发布可消费的运输证明。R2绑定/启动文件使用相同名称的`r2`版本，旧`r1`记录保留。

另一张真实原图TCGA-A8-A0AB的143块中142块成功、1块TLS握手超时，
已只重传缺少的1MiB并通过完整MD5，沿用原运输和原HIPT控制器恢复流。
该图的完整原SAM计算于02:51:50Z独立核验：455原行、168真mask、64非空原组。
累计3个不同患者、1503原行、516真mask；不是15图补齐完成，也不是临床训练新增完成。

02:56Z再次独立核对：首个旧前缀缺口TCGA-AC-A62V已补齐真实ROI，并完成原SAM后继审计：
305原行、92真mask，累计4个不同患者、1808原行、608真mask。
该图原WSI完整SHA与旧HIPT记录相同，原ROI数组回读SHA及冻结读取函数SHA再次一致；
未重建旧HIPT退休名称。新私有WSI和运输范围数据仅在ROI审计后退休，
764417字节的原失败分块及完整SHA以私有硬链接保留，可追溯，不是删除失败现场。
原HIPT流完成数已增至18，15图ROI缺口补齐完成1图并在处理下一图。
证据为`sammil_retired_prefix_first_real_roi_independent_recheck_r1.json`及
`progress_original_range_repair_and_sam_retired_prefix_backfill_20261106_r4.json`；所有完整任务声明仍未成立。

后续唯一患者前缀已增至8/856、3176原行、1069真mask，15个历史ROI缺口已补齐4个。
新增TCGA-MS-A51U为174原行/97mask，TCGA-AC-A2FO为309原行/172mask，
TCGA-AC-A6IV为272原行/102mask；新流TCGA-AC-A6NO为613原行/90mask。
`sammil_actual_unique_prefix_independent_recheck_r4.json`在实际节点重新检查这8个患者的
ROI数组字节、三份上游收据摘要和冻结科学驱动；没有把运输试点或旧病例重放计入新增。
这仍是完整856队列的逐图输入进度，不是完整SAM-MIL临床训练结果。
