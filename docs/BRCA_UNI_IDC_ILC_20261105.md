# 第一轮真实BRCA：UNI × IDC/ILC

用户确定的范围：固定UNI特征，预测IDC/ILC病理分型。不是HER2、ER/PR、分子分型或生存任务。

## 实际数据核查

只读来源：`hceph2:medeval/zhonglanfeng/CLAM_features_20260711/TCGA-BRCA/10x_256/uni/h5_files`。
该目录共有1087个特征文件，对应924个DX、151个TS、11个BS和1个MS。
本轮只纳入与原始OncoTree标签精确匹配的856个DX诊断切片，856位患者，IDC679、ILC177。
全部856个特征均存在，共10,009,314,912字节、2,422,295个patch；没有抽取小袋替代完整数据。

每个HDF5含`features: [N,1024]`和同行对齐的`coords: [N,2]`。
UNI官方使用DINOv2 SSL、ViT-L/16和1024维输出，见[官方模型卡](https://huggingface.co/MahmoodLab/UNI)。
但这批文件没有记录提取代码版本、具体权重哈希、MPP或坐标物理单位；
目录名称是UNI及10x_256的来源声明，不能把形状核对等同于权重/空间标定的独立验证。
这些缺项保留在原始任务元数据中。后续已只读找回全部856份CLAM原始patch H5，
其coords与UNI坐标的全部行序SHA一致；patch_size256、patch_level1、整数下采样4，
对应1024 level-0像素宽度。独立空间输入视图绑定这些实际证据及CLAM坐标约定，
不修改正在运行的原始任务，也不宣称已查明提取代码版本或MPP。

## 标签与既有患者划分

已从原始共享GPFS中的`TCGA-OT-csv/datasets/tcga-ot_{train,val,test}.csv`
重新读取这856位患者的标签和划分，并核对其与20261014保存的inventory完全一致。
没有重新随机分配，也没有把同一患者拆到多个子集。

| 子集 | IDC | ILC | 患者数 |
|---|---:|---:|---:|
| TRAIN | 610 | 132 | 742 |
| VAL | 40 | 29 | 69 |
| TEST | 29 | 16 | 45 |

标签编码固定`IDC=0, ILC=1`；以患者为评价单位，当前每位患者恰好一张纳入切片。
TEST仅45人，故本轮为单次既有划分的初步结果，不把排行榜小幅差异解释为显著优势。
临床标签来源一致性不等同于独立病理专家复阅。

## 方法和计算协议

所有164条方法记录逐项进入预检表。没有分类任务入口的方法记为不适用；
需要其他编码器、多尺度、原图、文本资产、组学或额外监督的方法记录缺项。
不会把生存损失改成二分类损失，也不捏造提示、Gleason等级或原图来凑齐164个结果。
通过预检仅表示可发起训练尝试，训练/评价失败仍单独记录，不能算完成。

使用已存在的H200单卡任务，不增加GPU申请。初始并发4，按用户授权删除约76.33GiB
已完成旧合成核验的过程输出和压缩副本后，重试队列并发上限提高到73，仍按实际主机内存
和显存准入。保留必要源码/权重和未完成旧记录，精简核验证据另存；已删大文件不能靠
精简收据完整恢复。已有ABMIL接管及后续MExD训练均未为扩大并发而重启。
实际活跃项以实时状态为准，不能把上限、排队或预检当作正在训练或已完成。
种子2026，固定同一任务和患者划分。普通入口保留默认50轮、学习率2e-4、patience10等
已有设置；专用原生驱动保留各自公开登记的完整配方。因此这是默认配方的固定数据比较，
不是声称所有方法具有完全相同的优化预算。每次运行的实际配置、原始失败、选模和
预测文件分别保存。新结果目录不覆盖旧实验；不使用TEST做选模、调参或阈值选择。

结果指标为患者级AUROC、balanced accuracy、macro-F1、accuracy及AUPRC。
个别原生入口使用独立slide评价视图：当前每人恰好一张切片，因此仍是相同45位TEST患者，
每次请求明确记录实际unit，不把slide指标字段偷偷改写成patient。
原始训练检查点保留于GPU本地新实验目录；成功结果的预测、指标、请求及可用的best检查点
复制到自己的共享目录。源共享桶和别人的目录均未写入。

## 当前实验位置

- 本任务共享记录：`/mnt/medeval-share-2/xiangyuli/mil_baseline_puls_work/development/brca_uni_idc_ilc_20261105`
- GPU本地工作区：`/root/milplus_brca_uni_20261105`
- 原始标签/划分证据：`source_splits_audit.json`
- 完整特征准备收据：`prepared_task_receipt.json`
- 全方法预检：`preflight_all164.json`、`preflight_all164.csv`
- 第一波原始训练状态：`benchmark_state.json`
- 重试/新输入训练状态：`repair_wave_state.json`，后续依赖修复为`repair_wave2_state.json`
- 合并实时汇总：`current_results_all164.json`、`current_results_all164.csv`
- 25个用户明确排除的方法：`excluded_25_methods.json`，保留21生存/2有序/1回归/1多标签原任务，不实现伪分类版
- 66个缺输入方法逐项记录：`remediation_66.json`，首层输入放行不等同完整训练成功
- 坐标来源与逐行比对：`coordinate_headers_audit.json`、`coordinate_alignment_receipt.json`
- 原生条件的实际数据核查：`native_geometry_audit.json`、`native_input_blockers.json`
- 清理范围及精简凭据：`legacy_cleanup_receipt.json`、`legacy_cleanup_evidence/`
- 已验证依赖修复、显存隔离重试等后续队列：`deferred_runtime_repairs.json`
- 真实RGB输入提取/接收：`raw_rgb_producer_state.json`、`raw_rgb_receiver_state.json`
- 原图初始化和最大完整袋的容器检查：`native_rgb_pilot_contract_checks.json`
- 原生RGB全队列准备/准入：`native_rgb_method_preparation.json`、`native_rgb_ready_inbox_accepted.json`
- 成功项结果：`results/<family>/`

以上目录是新实验，与旧4955项软件回归和原生预算验收分开登记。

## 不可伪造的原生条件

12个原缺输入方法已补齐首层契约进入独立训练尝试；不是宣称12个全部训练完成。
几何核查发现628个原袋不在同一1024px晶格上、204个袋存在栅格单元冲突。
需要严格原生晶格的方法不能通过坐标取整、静默覆盖或丢弃patch放行。
AMIML原配方四个形态簇各R100，但28个实际袋总patch数少于400；不能通过复制patch、
改变R或删除TEST患者伪造满足条件。其余原图、多尺度、区域/实例监督、文本和原始权重
需求按逐方法清单保留，现成UNI1024不能冒充原生CONCH/CLIP/512D输入。

候选仓库现已在AMIML进入完整TRAIN聚类前检查必要容量：以实际患者所有切片之和为单位，
要求至少`4*max(11,R)`个真实patch；满足数量仍不等于每个形态簇满足条件，后续原检查不删。
不改R、聚类、损失、优化器或数据；26项相关机制、失败边界、恢复与评价检查通过。
正在运行的GPU源码仍冻结不变。证据：`amiml_capacity_cpu_rerun.xml`。

只读实地核验进一步确认全部856份原始WSI仍存在，合计约826.67GiB；
共享目录还存在5x_256、10x_256、20x_256的patch目录结构，分别记录于
`real_raw_input_inventory.json`及`existing_rgb_patches_audit.json`。
进一步对本队列逐一检查：三个倍率各有856个匹配切片文件夹，但每个文件夹均为空，
因此不能据目录名宣称有可复用的真实RGB或完整多尺度输入。尚需原图提取/找回真实资产。
未复制826GiB原图到开发盘，也未把生成的breast图像当真实监督或原图使用。

## 原生RGB输入的实际准备

最大切片的16848个实际256RGB patch已完整提取、无缩放/抽样、逐像素回读校验，
压缩HDF5约2.04GiB；转到现有GPU本地盘后再次检查完整SHA，才清除自己的运输副本。
原始WSI和原始坐标文件未写入。C2C、原型PMIL、耦合ICMIL、Center Embeddings
均已对这一个真实完整袋及各自原始初始化/历史辅助源码进行容器契约检查，
这不是856张完整输入准备通过，也不是四个模型的训练结果。

全队列目标保持856患者、2422295个原始patch及既有划分；完整未压缩RGB约443.54GiB，
采用原生HDF5无损LZF逐袋运输到GPU本地盘，不在近满的共享盘保存整批像素。
运输保护条件不改变科学预算；最初8GiB余量条件造成空缓冲等待，已在保存8个完成袋后
有记录地交接CPU提取进程，改为可实时读取的2GiB余量条件，未改变提取函数。
运输/输入检查进程交接期间没有向任何训练进程发信号。

RGB方法单列`input_group=native_rgb256_original_coords`，与`uni_frozen`主组分开，
不能把它们冒充UNI冻结特征结果。只有全部856个袋送达、初始化哈希和原生输入契约核验后，
才通过独立输入队列提交现有修复调度器；不在最大单袋试跑上训练或报告结果。
Center Embeddings原中心输入接口保留。后续核对确认其坐标只用于读图，新增显式真实左上角
缓存表示，逐行验证原作者norm_coord逆映射定位到完全相同的read_region；不将API锚点
冒充物理中心，不取整真实几何中心。详见`CENTER_READER_EQUIVALENCE.md`及下文。

第二轮修复不再强制等待长预算第一轮全部结束；确认第一轮无尚未启动的任务后，
以实际PID+启动时钟核对仍活跃的任务，并为其保留主机内存/显存峰值余量。
独立需要大显存的失败重试继续等待安全的近独占条件。只有一个调度器接收新任务，
避免多个新任务队列分别判断余量而同时超额准入。上限仍73，不以填满进程数代替真实计算。

## 原生CONCH补充输入组（不是UNI主组）

已只读找到原始共享CONCH1.0检查点，802235437字节，SHA-256为
`40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`，
与现有FOCUS/ConMIL原生权重要求一致。完整复制到GPU本地后再次核验SHA，
删除自己的约765MiB共享运输副本；替换此前约127MiB已确认截断的内部复制件。
没有删除原共享权重。此前二进制websocket复制失败的记录保留，不能用退出码代替完整性核验。

MIL-Adapter的作者固定提交公开BRCA原型已逐Git blob导入，float32 `[512,2]`，
列顺序IDC/ILC，SHA-256为`296b79c80a460f94720bd4675cd79c20e0a1b26d6b8f7c05a7ba9f490c0599df`。
作者没有披露该原型的具体生成提示和权重版本，因此不声称精确恢复其未公开历史图文对齐。

视觉提取使用固定CONCH提交`141cc09c7d4ff33d8eda562bd75169b457f71a62`的原始代码，
timm0.9.12，严格加载全部175个视觉张量。公开权重不含caption text_decoder，
因此不初始化或使用该缺失语言生成解码器；未加载的151个非视觉张量不被伪称用于图像推理。
执行原始`encode_image(proj_contrast=True, normalize=True)`，输出float32原生512维L2特征，
使用官方448BICUBIC/CenterCrop、RGB和OPENAI均值方差，不改变原始256RGB读取或坐标。
不使用AMP、TF32、染色归一化、抽样或标签拟合。

真实最大袋的首/中/末三行已通过原生投影参考与逐张/分批推理比较；这只是实现检查，
不是完整袋/完整队列或模型训练完成。四项运输/内容/行身份单元检查通过；初次测试发现
虚拟文件系统时间戳不足以独立保证内容，现已增加RGB源完整SHA的首次使用核验。
初次失败XML与通过重测XML均保留。证据为`native_conch_visual_check.json`、
`native_conch_pipeline_unit_tests_rerun.xml`。

实际全队列逐袋特征提取已启动，目标仍856张/2422295个原始patch，
单列`input_group=native_conch1_projected_l2_rgb256_original_coords`。
冻结提取源码、PID和启动时钟存于`native_conch_feature_launch.json`；
实时进度为`native_conch_feature_preparation.json`，完整袋凭据为`native_conch_features_delivered/`。
逐袋读取原始RGB全部行，原样保留coords和patch_ids，完整特征回读与SHA通过后才发布。
遇到实际资源不足或近独占训练时让出提取GPU占用，不中断任何已有训练。
全856袋及各方法原生契约未完成前，不宣称方法已放行或计入完成数。

### 不改计算配方的双进程CONCH提取

2026-11-05T17:02Z，单进程提取在完整袋边界交接为一个协调器和两个常驻提取进程。
先确认不存在未完成文件、已发布H5与收据一一对应，再停止旧提取控制进程；
不丢弃半袋、不重启训练、不向训练进程组发信号。实际调用仍为冻结原模块的
`extract_bag`，固定batch8、全行、原始float32投影及L2流程，没有改AMP、权重或预处理。
原`protocol.json`继续标识实际未改的计算函数模块；新的协调入口SHA独立记录，
不把旧单进程main伪称为当前入口。每个并行袋另附协调来源及完整输出SHA凭据。

两个进程以原子独占claim分配不同完整袋，不抢占陈旧claim；协调器持有唯一队列锁。
全856袋仍须逐个执行原完整SHA/坐标/patch身份核验且总行数2422295，才标记队列就绪。
真实3858行TRAIN试提取袋以原字节迁入主输出，保留旧试提取收据及迁移映射，没有重复计算。
6项claim/迁移/完整边界控制检查通过。当前负载下观测两路合计约55行/秒，
此前近期单路中位约33行/秒；这是共载观测，不是受控空闲GPU加速比或速度保证。
证据为`native_conch_parallel_handoff.json`、`native_conch_parallel_pilot/receipt.json`、
`conch_parallel_io_tests_r2.xml`和`native_conch_orchestration_receipts/`。
两个提取进程不是新增两个已训练方法；实时完成袋数以`native_conch_feature_preparation.json`为准。

SMMILe的首层缺项不是任意RGB分割：原始NIC/SLIC需要真实物理规则晶格；
同一批628个非单晶格袋亦不满足该条件，已加入`native_input_blockers.json`。
不使用缩略图分割、stride1重新命名或坐标取整掩盖该原生条件。

## 已运行调度交接与完整输入接收

第二轮调度控制进程已从540728交接到549516（2026-11-05T15:40:42Z）。
交接只向核对PID/启动时钟/命令行的旧控制进程发信号，没有向训练进程或进程组发信号。
ASMIL/CARMIL/FCMIL/MAMIL/MicroMIL/MiCo共6个当时活跃子进程的PID和启动时钟逐一保持，
完成的旧状态快照、控制锁和实际子进程清单独立保留于`scheduler_v2_handoff_*.json`。
新控制器复用原始worker/exporter，接管活跃子进程时继续保留完整峰值内存/显存估计，
没有重启训练、修改原任务或清除失败实验。

除原生RGB接收外，新增`verified_input_intake/`证据门控队列。17项控制检查覆盖
实际旧控制器存活拒绝、交接快照改变拒绝、任务排除、运行中/已完成方法拒绝、
INS仅内存保护参数改变，以及不完整CONCH队列拒绝。独立收据记录接受或拒绝，
错误提交不会使其他训练退出。当前允许的通道不表示方法已经适用：
MGGM-MIL原生仍要求ResNet18/PLIP及原始K4/8/12图，不能因CONCH也有512维就冒充可用。
同理，in-context ICMIL还要求固定袋长，TRAIN-only PCA25不足以解决128至16848的不等原始袋长。

INS已在新队列被接受为独立`ins_repair3_guard96`尝试，仅把预分配保护从32GiB改成96GiB，
未改模型、原生ResNet/BN、完整patch、损失或轮数；显存预留96GiB并等待近独占准入。
此前32GiB保护失败及其原始日志留在prior_attempts和旧目录，不记成完成。

MIL-Adapter的全856袋原生输入准备器已启动，严格等待CONCH全部2422295行完成。
6项额外门控检查确保855袋、少一行、错误输入组或未完成标记均不能触发准备。
就绪后才核验全部特征字节/坐标/patch数，建立独立slide评价视图和作者BRCA原型绑定，
检查原生8shot/class支持集全部来自TRAIN，保留742/69/45原患者划分，
再将默认`miladapter_taskres_abmil`提交同一调度器。
准备器不自行发起训练、不改成全监督支持集、不以准备成功冒充模型完成。
来源与实际句柄为`native_conch_method_launch.json`，进度为`native_conch_method_preparation.json`。

2026-11-05T17:21Z，仅交接尚在等待完整输入的准备器550048到563501。
准备阶段支持集显式使用sampling_seed2026，和既有训练引擎的seed2026覆盖行为一致；
此前默认0只影响待生成凭据，尚未生成真实BRCA输入或发起该方法训练。
40患者合成软件夹具执行实际引擎1轮，核对准备/训练两端的16个支持ID相同且全部来自TRAIN。
这是软件一致性检查，不是BRCA完成或原生20轮训练。其他控制/提取进程PID及启动时钟不变，
未改变8shot/class、真实划分、模型、权重或训练预算。
凭据为`miladapter_support_seed_alignment.json`和`native_conch_method_preparer_v2_handoff.json`；
当前准备器源码冻结在`native_conch_method_source_20261106_r2`，旧启动收据仅作历史记录。

15:39Z DGMIL原始训练子进程以-9退出，日志仅到epoch1，无Python错误栈。
同次只读检查cgroup累计oom_kill为6，但累计值没有该PID的时间绑定，
故只记录SIGKILL事实及可能内存压力，不能仅凭它断言该次OOM原因。
未盲目重启或减少TRAIN实例；需进一步查明峰值与恢复条件。

## cDP数值修复与DGMIL精确续跑

真实TRAIN153x1024完整袋上已复现cDP float32协方差失败；同一全部参数的float64拟合
正常收敛，原始float32硬均值/分类器保留。新增显式`aggregation_dtype`选项，默认路径不变；
39项新旧原作者机制/训练/恢复检查通过。独立GPU源码副本仅改cdpmil.py，
新尝试`cdpmil_numeric64_repair2`仍属UNI主组，但记录具体精度条件；
另将一次性内容缓存扩至856袋（约数十MiB质心），避免重复相同确定性预处理。
未改成分数、协方差形式、正则项、迭代上限、patch或分类器预算。
详见`CDPMIL_NUMERICS.md`和`cdpmil_numeric_candidate/`。

DGMIL找回完整epoch1的last.pt及优化器/RNG/密度状态。只读核对实际源进程已终止、
训练库存742张、optimizer_steps1/refinement1、请求签名及当前冻结软件身份完全一致。
原失败目录和陈旧锁不改；仅复制原字节到`dgmil_repair3_resume`，新目录不带旧锁，
完整SHA逐文件核验后用`--resume`从下一轮继续，仍由原生引擎验证全部恢复签名。
仅将主机峰值预留提高至64GiB，不改变配方，也不强制无必要的GPU独占。
凭据为`dgmil_exact_resume_preparation.json`。

为接收这两项，调度器于15:59:53Z交接到551606。5个仍活跃子任务PID/启动时钟
逐一保持；24项接收/恢复控制检查通过。两项提交均已有accepted收据，
这仅表示已进入资源感知队列，不是已开始或已完成。其既有失败均留在prior_attempts
及原目录。实际开始以新子PID和运行请求为准，不能把待调度项计为训练中。

## Center缓存坐标表示的原作者读图等价核验

已确认原先要求所有原图包带整数物理中心，是缓存格式的过强限制，不是模型使用坐标的要求。
新增显式top-left模式，逐行执行原作者整数norm_coord逆映射；保持原始256RGB、真实左上角、
行序和patch IDs，拒绝无来源/错误SHA/伪造网格。5张真实WSI的29137行几何和15个真实patch
逐字节读图比较通过，26项输入检查通过。原始中心模式不变。

完整原作者对照不因开发机137退出而降低预算。更大主机的14项原样K32/256RGB/双轮对照中，
12项通过，2项发现CPU参考的DataParallel自动选CUDA导致标签设备不一致。
失败报告保留；仅在修正CPU参考的设备可见性后重跑原样2项，两项均通过原作者完整参数、
Adam时刻、BN和RNG对照。全部14项case及各自XML/SHA已独立逐项核对。
这不是首轮全绿、完整论文预算或真实BRCA模型完成；仍需全856原图输入准备后发起原生训练。

原图输入准备器已从540717交接到556939，旧进程当时正等待全856袋。
实际控制器、训练、接收和CONCH提取PID/启动时钟保持不变，无训练进程收到信号。
新准备器仅让Center使用独立metadata源码；其余三项仍用原冻结源码。
全部856输入、原初始化、输入契约及完整原作者检查均满足后才交给原调度器。

## SparseConvMIL实际CUDA依赖已补齐

2026-11-05T17:31Z，独立目录内原样编译完整SparseConvNet CUDA后端成功。
此前保留的二进制实际仅支持CPU；系统CUDA13.2亦不能与Torch2.5.1+cu124混编。
现从官方仓库按大小/SHA核验7个CUDA12.4包，只解包到本项目GPU本地前缀，
使用12.4.131/SM90编译，未安装到系统、修改原始源文件或改变活动训练环境。
4项实际GPU输出/梯度/完整RGB256网络Adam检查通过，真实最大16848行原图容器契约通过。
这是补齐依赖，不是新增完成训练；仍等待全856原图及该方法独立manifest。
现有注册路线为原公开脚本ResNet18/5tiles/batch2/200epochs条件，
不冒称不同的论文TCGA ResNet34/128px条件。详见`SPARSECONVMIL.md`和独立构建/核验收据。

2026-11-05T17:40Z，同一调度器交接到V4/PID566749；38项既有/新增准入控制检查通过，
5个当时仍活跃的repair2训练PID与启动时钟保持，原worker及资源保护不变。
新增SparseConvMIL独立输入准备器PID566971，真实316份已交付收据的行身份/坐标SHA/
文件指纹核对通过；严格等待全856原图后，才运行原准备器完整SHA、建立独立manifest、
核验742/69/45划分和IDC/ILC顺序，向现有队列提交原始200轮源码配置。
它不自行训练；V4拒绝不完整队列、CPU假后端、改预算/模型/环境、重复运行与排除方法。
实时来源为`native_sparseconvmil_preparation.json`，不能把待准备项计为完成。

## Prompt-MIL原始BRCA初始化已核验，RGB224仍未就绪

2026-11-05T17:45Z，取得作者README公开BRCA DINO检查点，440218700字节，
SHA-256 `a80864a58cae61c690ff9709b6e34c0148fa08e11112cef5de5c823d1fe88596`。
GPU本地副本完整SHA再次通过，实际固定原作者代码严格加载149个骨干状态张量，
维度192/12层/3头，只有真实提示参数可训练；完整DSMIL分类器保留。
证据为`promptmil_original_brca_assets_verified.json`。这是资产加载，不是RGB前向、完整队列或训练完成。

原作者`script/merge_img_wsi.py`实际提供`read_img_resize`/`cv2.resize`到224，
因此原图256缓存本身不是不可解决的模型限制；仍须明确执行并核验原预处理，
不能把256缓存、Pillow替代插值或UNI向量冒充已经准备的原生RGB224。
原脚本默认JPEG及历史OpenCV版本未恢复，不声称历史像素精确对齐；
该BRCA专用SSL检查点可能包含本队列，未声称历史预训练患者级排除。
作者BRCA示例为5x，本队列为单独声明的10x_256输入，不据此声称论文数据集复现。

2026-11-05T18:11Z，原作者17项完整机制/双轮/恢复及新JPEG入口10项检查均在
实际节点CPU路径通过，真实205行的200+5组JPEG字节和作者Dataset最终张量逐位一致。
选择作者原生的OpenCV可选解码回退，固定OpenCV4.10.0.84，并绑定实际build SHA；
不以Pillow解码替代，也不冒称历史版本对齐。
2899行完整真实袋的所有源RGB像素/SHA、坐标、patch ID和输出JPEG回读核验通过。
全队列原生JPEG准备进程PID573041/start832660574已实际运行，复用该完整袋，不重复计算。
初次父进程读取状态时早于子进程导入结束，之后核对原子PID/启动时钟/命令确认仍活跃；
没有因观察竞态重启它。完整源码及计算协议冻结，现有训练/runtime2未修改。
实时进度`native_promptmil_rgb_preparation.json`，完整袋收据`native_promptmil_jpeg_delivered/`。
这仍是输入准备，不是模型训练或UNI主组结果。详见`PROMPTMIL.md`。

2026-11-05T18:33Z，接收控制器交接到V5/PID576431；64项既有/新增准入检查通过，
4个被接管的实际训练PID/启动时钟保留，无训练进程收到信号。
新增方法准备器PID576845，只在全部856 JPEG袋/2422295行完成后建立独立真实输入契约，
核验原始划分、教师、完整输入和既有40轮BRCA paper_reference配置，再向同一队列提交。
完整袋、27项实际源代码/新格式检查、像素协议和SHA缺一不可；不增加GPU申请或重启已有训练。
当前部分输入明确未被提交，状态见`native_promptmil_method_preparation.json`。

## PAMoE原始组织分类输入与TRAIN-only原型（2026-11-06）

已查明公开源代码的组织分类使用原生224 CONCH，而非目前正在生成的448特征流；
保留40个别名×22模板、九类原始顺序和classID>=4归Other4规则，提示词没有擅自改写。
完整原始factory与独立加载器的视觉状态/真实图像输出，以及全部880条提示词形成的分类权重逐位一致；
实际65行、batch64+tail1的作者分类函数输出一致。
GPU没有原始WSI挂载导致的R1失败记录保留；现有原图节点执行原样Dataset类，
无损转交首/中/末三个真实patch，缓存RGB和官方224处理张量逐位核对通过。
凭据`pamoe_native_tissue_classifier_verified_r2.json`，不是组织临床准确率或MIL训练完成。

组织资产准备器PID586535/start833063596正在实际运行。
原型fit只使用742个TRAIN患者的UNI原始H5视图；实际目录顺序+显式seed2026，
按原作者整袋`cnt>500000`停止规则选中168张/500974行，所有末袋原始patch保留。
完成真实标签后，执行原样四类float32均值；Other4仅从原型拟合中排除，不能从MIL训练bag删去。
四类空缺、坐标/SHA/patch ID/患者范围异常将保留失败，不能制造替代原型。
现有方法代码、UNI特征、默认100epoch/原VAL停止机制未变；这也没有将25个排除方法纳入分类。
公开原始提示词含结直肠相关词，历史UNI权重/队列和倍率限制仍明确保留。

实际节点88项既有/新增准入控制检查通过，队列于19:30:26Z交接到V6/PID587938/start833105052。
CARMIL540737、FCMIL540747、MAMIL540757的启动时钟和原训练进程保持，无训练收到信号。
首次控制检查有4项因新接收器向Path-only摘要函数传入字符串而失败；报告保留，
只修复这两处控制层路径转换后，开发机88项及实际节点88项全部通过，未改模型或科学预算。
提交观察器PID588006只在真实原型/完整856 UNI契约就绪后，向同一队列提交一次原生PAMoE；
当前组织标签仍在生成，不能将准备中或已接收误计为训练。
实时凭据`native_pamoe_tissue_preparation.json`、`native_pamoe_scheduler_submission.json`，详见`PAMOE.md`。

19:31Z独立逐患者TEST审计为56项通过、0项错误；所有项仍为锁定45位测试患者。
该数不是164方法全部完成，也不是论文分数复现。

## 原始RGB完整交付、原生JPEG完整就绪及Campanella缓存路线

2026-11-05T19:50:50Z，全部856原始RGB袋/2422295行交付并完成现有接收核验。
19:55:23Z，原生Prompt-MIL全部856 JPEG袋/2422295行也完成最终核验，`cohort_ready:true`。
这两项是完整输入准备，不能直接增加训练完成数。C2C/PMIL/耦合ICMIL/Center、SparseConvMIL、
Prompt-MIL各自的全队列输入契约准备已自动继续，仍须方法级检查与队列准入。

Campanella新增显式实测RGB区域缓存，以原始level1/256和源码已有`mult=8/7`获得两阶段原生256读取，
随后保留作者PIL224双线性/.5/.1处理。没有将缓存冒充整张WSI，
没有改变完整ResNet34、10步原RNN、100+100epochs或512/128批次。
历史倍率/队列仍不同，不冒称论文数据或结果复现。
真实WSI作者Dataset的CNN3区域+RNN10步与缓存张量逐位相同；原有50项+新9项检查通过。
额外3项完整缓存训练/恢复对照首次2过1失败，仅为原TIFF/新H5文件身份本应不同；
6个身份字段以外的全部数值状态一致。失败证据保留，新增分别验证原身份公式后重跑，
不得将未完成重跑或输入准备误计为临床训练完成。详见`CAMPANELLA.md`。

20:00:10Z，106项实际节点既有/新增准入控制检查通过后，同一调度器交接到V7/PID595003/start833283484。
CARMIL/FCMIL/MAMIL当时的3个repair2训练PID/启动时钟全部保持；其他首轮训练不受信号。
独立Campanella准备器R2/PID594871/start833279792等待完整62项报告，
全部原始输入与原配方契约完成后自动提交同一队列。此前等待中的R1准备器已受控交接，
无训练进程被重启，活动方法源码和runtime2保持不变。

20:02:19Z，Campanella62项原样机制/缓存/恢复/完整数值轨迹检查全部通过。
20:02:39Z完整队列构建因318张源WSI缺省可选quickhash被R1过强限制拒绝，未提交任何真实训练。
全部856源stat前后/level/size/真实RGB仍一致，失败目录和日志保留。
R2仅修复可选元数据契约，真实None保留，不制造quickhash或剔除病例；真实stat/完整RGB与坐标SHA保护不变。
新3项边界与全部既有检查共65项正在独立CPU路径核验；当前活动V7仍固定R1，
R2不得在完整核验和新准入条件绑定前训练。详见`CAMPANELLA.md`末节。

## 65项修正版检查通过与六项原生方法真实入队（2026-11-06）

20:15:23Z，Campanella修正版全部65项原样检查实际通过；13个真实原始WSI输入张量再次逐位通过。
128项既有/新增准入控制在实际节点通过，20:12:43Z同一队列交接到V8/PID597647/start833358723，
3个既有repair2训练PID/启动时钟保持，其他原训练及runtime2不变。
新准备器PID596952/start833342093在独立R2输入目录核对全856归档，
旧失败目录和62项旧条件证据保留，不把修正后的软件通过等同于真实200轮训练完成。

C2C、PMIL-prototypical、耦合ICMIL和Center的全856/2422295行输入契约全部核验完成，
原始native_rgb_ready_inbox已有4项完整accepted回执；Prompt-MIL与SparseConvMIL各自也已accepted。
这六项的状态均为ready_for_training_attempt，而不是已开始/已完成；原始预算、初始化、模型和原始患者划分保留。
20:19Z实时台账为56 completed、17 ready、13个实际训练进程、53 blocked、25 excluded，合计164。
SCL-WC的原CLI失败与目录完整保留；它在构造阶段因未登记PNM参考类/角色而拒绝，未生成last/best，
当前汇总归入需要显式科学终点声明的输入阻塞，不能把IDC自动称为正常组织，也不能称该项已训练成功。

共享桶BRCA/10x_256的16个FM子目录仅做只读目录核对；未在该层找到PathExpert/核计数资产目录，
不据此宣称其他位置不存在，也不以同维其他FM冒充当前UNI或原生文本空间。
两次有界新增CONCH worker探针使用原样batch8完整真实袋和原子逐袋占有，
原2个worker/协调器及模型训练不改；额外完整袋直接形成规范特征与SHA收据，
不进行人工占用或预算删减。是否长期扩并发须依据实际资源及吞吐，不凭GPU占用率判断。

20:31Z Campanella完整R2输入、原初始化和原生200轮两阶段契约实际通过并已accepted；
因此本轮共有7个原生输入方法从缺输入转入现有队列：C2C、PMIL-prototypical、耦合ICMIL、
Center、SparseConvMIL、Prompt-MIL和Campanella。它们此时均只排队，没有被计作新增完成训练。
20:40Z台账为56 completed、18 ready、13个实际存活训练、52 blocked、25 excluded，共164。

两个额外CONCH有界探针产出2个完整规范袋/8435行，原始batch8/权重/函数SHA/预处理不变，
新worker约19.5--19.9行/秒，期间两个原worker已完成袋的中位约22.3行/秒；
此前双worker阶段中位约25.8行/秒/worker。这些为不同袋/负载的观察，不是受控因果加速比。
20项现有IO/新增接管生命周期控制检查在实际节点通过后，20:36:59Z仅交接提取协调器，
新协调器PID604296/start833504289接管原worker560346/560347，并新增604297/604298。
原2个提取worker不重启，全部13个训练PID/启动时钟保持；同一cohort锁和逐袋原子claim防重复。
四个worker都执行原固定`extract_bag` SHA1eb4d015...、batch8；只有单一协调器发布完整队列就绪。
完整856/2422295行及每个文件SHA仍是最终准入条件。
被接管非子进程的退出码明确标为不可观测，必须有自身完整结束状态和全856已核验结果，
不能编造exit0或把进度计数当完整结果。故障保留partial/claim并仅停止明确属于本池的提取身份，不触碰训练。
凭据`native_conch_four_worker_handoff_launch.json`和`conch_four_handoff_1793911018938306325.json`。
20:40Z为CONCH404/856袋、895599行，PAMoE组织标签145/168袋、447457行；均仍是准备进度，不是分类训练完成。

## PAMoE 真实 TRAIN 原型完成并纳入同一队列

2026-11-05T20:51:45Z，PAMoE 原生 224 组织分类完成全部选定 168 个 TRAIN 袋/500974 行，
实际执行原作者整袋前缀和 float32 四类均值，四类均非空。
原型行数依次为 TUM33031、STR189166、LYM24252、DEB20353；Other4 的 234172 行
只排除于组织均值，不能从完整模型训练输入删除。没有使用 VAL/TEST 拟合或选择资产。
全部 856 患者、742/69/45 划分及 2422295 行原 UNI 契约通过；UNI 历史权重未知的限制不变。

20:51:51Z现有V8接受 `pamoe_native_trainpriors1_20261106.json`，原生方法配方未修改。
准备器586535和提交观察器588006均正常完成；没有重启或向已有训练进程发信号。
PAMoE 此时只等待训练，不计为新增分类结果。20:52Z实时台账为：
56 completed、19 ready、13个训练进程、51 blocked、25 excluded，共164。
完整证据见 `PAMOE.md` 新增完成记录和共享目录的原型/契约/accepted 收据。

20:57:12Z，另完成 CATE 作者固定 BRCA 文本的真实资产导出：六组 528 条完整原始查询，
实际 CONCH1.0 文本权重严格加载，原字面循环与完整原始 factory 的所有输出逐位相同，
现有资产读取器回读相同。未生成新临床提示词，未使用任何患者标签，未修改活动环境。
这不能使 CATE 直接进入训练：仍等待全部 856 袋配套 CONCH 图像特征，以及 PIM/SIM 权重的明确实验选择。
论文附录确实公开权重消融，但不能将工程示例 1/0.1 冒充主实验默认值；保留待确认状态。
共享证据 `cate_original_brca_text_verified_r1.json`，详见 `VISION_LANGUAGE.md` 新节。

## MUSE 原始知识库与现有 UNI 输入实际补齐

2026-11-05T21:07:09Z，MUSE 全856患者/2422295行原始UNI1024输入契约实际通过，
原作者完整IDC/ILC知识库各300条、原CONCH文本检查点及固定模型源文件均逐字节绑定。
原模型构造器本就声明input_dim1024及可学习Linear(input_dim,512)适配器；
没有加一个额外投影来冒充图文空间对齐，也没有改现有UNI的行、维度、字节或历史未知权重声明。
独立输入组为 `uni_frozen_muse_released_source_adapter1024`，原生slide评价显式保留；
本队列每人恰好一张切片，742/69/45患者边界不变，不宣称论文原FM/分数复现。

148项既有/新增调度控制在实际节点通过，且真实全部856份特征再次独立重验后，
21:14:57Z仅将V8控制进程597647交接为V9/PID612414/start833732142。
所有13个实际训练PID/启动时钟保持，原worker、runtime2、其他输入门控和并发上限73不变。
21:15:42Z同一队列accepted `muse_native_uni1_20261106.json`，仅等待训练，不计新增完成。
保留每类300知识、20视图独立更新、原200轮上限、Adam/VAL停止机制和全部公开源码差异。
主机160GiB/GPU96GiB及独占准入是未实测峰值前的保守保护，不能据此削减原始预算。
完整输入初始化/临床训练仍须实际执行，详见 `MUSE.md` 新节及独立收据。

21:17Z CARMIL原训练PID540737自然完成100轮、74200个TRAIN minibatch、100个原选择器，
随后独立进程612945完成锁定45人TEST评价，并非调度交接造成训练丢失或重启。
21:18:38Z全部57个已完成方法的患者身份、标签和指标独立复算通过，0项错误。
CARMIL本次AUROC0.9913793、accuracy44/45、balanced accuracy0.96875；
这是单次45人划分，不是论文结果复现或统计显著性声明，详见 `CARMIL_AUTHOR_RECIPE.md`。

现有调度器已于21:17:52Z自然接续启动cDP数值修复尝试，PID613145/start833749672，
继续使用此前已准入的完整原袋和显式float64聚合拟合条件，没有借CARMIL完成而修改其他预算。
21:19Z当前台账57 completed、19 ready、13个实际训练、50 blocked、25 excluded，共164。
原13个训练在交接时全部保留；其后CARMIL正常完成、cDP新启动，不能把旧PID消失误报为训练丢失。

## 单一 BRCA 任务边界与 NcIEMIL 前置容量核查

2026-11-05T21:29:47Z，以实际856患者的slide视图调用未改的AKD-PMP、CoMEL、
QPMIL-VL和KMP-MIL完整输入接口，四项均因只有一个实际任务而明确拒绝。
CoMEL至少需两个二分类任务/四个全局类别位置；QPMIL/KMP至少两个非空任务且首任务两类，
因此当前仅IDC/ILC两类不能通过补文本或patch IDs解决完整持续学习契约。
AKD-PMP同样需要额外真实任务协议，不能擅自改为IDC后ILC的单类递增实验。
这些仍记blocked，不擅自增加用户已经明确排除的25项，也不删除原入口或用首任务分类器冒充完整方法。

同次读取全部856个已交付真实RGB H5形状及坐标SHA，证实8袋不足256个patch、146袋不足1024，
最小128；所有小于1024的袋另外实际核对了patch ID唯一性。
NcIEMIL原生最小K128需要256个不同patch，因此当前袋即使补齐Swin/预算也不能完整放行。
开发候选将原样容量检查提前到过滤器训练之前，不改模型/K/选样/预算，也不复制或丢弃病例。
新增边界和流程控制加全部既有原生图/源码对照，共39项在实际节点通过，未动活动GPU代码。
原开发环境因缺timm的collection失败报告保留；未以本轮软件检查冒充真实Swin/100轮训练。
证据为 `brca_continual_scope_and_nciemil_real_capacity_r1.json`、
`nciemil_capacity_and_native_source_actual_r1.xml`，详见 `NCIEMIL.md` 新节。

21:45Z，补齐上述NcIEMIL改动的独立安装包检查：当前工作树隔离快照构建wheel，
164条登记/415配置及512个包文件逐字节核对，新容量模块确实在包内。
离线pip安装到新目标，不改任何活动依赖；实际节点CPU子进程从该安装位置通过公开模块CLI和原样39项相关检查，
全部已加载milplus模块的路径逐一验证，不从原源码目录借用模块。
这不是164方法全部验收、新的真实训练完成、全Swin/100轮预算或新公开发行版。
证据 `nciemil_capacity_wheel_r1/installed_wheel_actual_receipt.json`，详见 `NCIEMIL.md` 新节。

## MExD 完整完成与 GTP 真实输入复核

2026-11-05T22:06Z，MExD自然完成完整100+200轮后的锁定45人TEST评价。
22:08Z独立核验全部300轮、222600次完整TRAIN访问、74200/148400次阶段更新，
并逐袋复算100条TEST后验的概率、原生分数及配对t检验；未重新推理或用TEST调参。
原生采样和协议证据另存`results/mexd/native_posterior_evidence_r1/`，非仅保留最终指标。
本次MExD AUROC0.8900862、accuracy41/45；完整58项TEST身份/指标独立审计0错误。
明确是UNI1024参考重建条件，不是原论文编码器/私有初始化/分数复现或当前新wheel完整预算验收。

同一队列22:00:41Z已启动SparseConvMIL原生RGB训练PID622706/start834006497，
保留200轮、5patch有放回抽样、batch2和真实CUDA稀疏后端；不是另起第二个调度器。
CONCH四worker仍在交付完整原始袋，22:08Z为526/856、1352979行；部分特征不准入训练。

GTP既有尝试先被原1GiB dense保护拒绝。随后对全部856原坐标调用原生格点接口，
证实另有628袋物理1024px网格不合法，不能仅靠增加保护解决。
原batch8最大TRAIN袋的46.09GB源码估计不是实测峰值；不降batch或伪造网格。
保留原失败目录/原台账，以`gtp_real_native_prerequisites_diagnostic_r1.json`将汇总归为输入blocked，
没有加入25项用户排除名单。详见`GTP.md`新增实际诊断。

## HIPT 真实教师资产补齐（不冒充完整金字塔训练）

2026-11-05T22:18Z，取得并核验官方固定ViT256教师704238867字节父文件，
42块传输及最终原SHA通过；安全导出完整150张量骨干并私有备份。
真实缓存RGB首/中/末三张的原作者处理及严格/原factory/便携回读三路384D前向逐位相同。
既有ViT4K父文件/便携教师也重新校验通过，原生权重不再是未取得资产。
无新增CPU/GPU作业，没有GPU试算、完整特征提取或HIPT训练。

HIPT仍须全856病例的真实完整16×16物理区域和原生384D特征；
三张原UNI坐标缓存RGB不是完整金字塔，不能按每256行凑区域或冒称原论文倍率。
原WSI源任务已经结束，新CPU读取任务尚待授权；现有GPU调度/训练/四worker继续。
更新输入blocked原因而不改原失败台账，完成数仍58、排除数仍25。
证据`hipt_vit256_native_teacher_real_rgb_verified_r1.json`，详见`HIPT.md`新增资产核验。

## SAM-MIL 原始权重补齐与 FEATHER 访问权限复核

2026-11-05T22:31Z，SAM-MIL作者指定的Meta官方ViT-H权重完整取得并在本地CPU验证。
2564550879字节原文件MD5匹配`4b8939`前缀；完整SHA由本地实测并明确其来源。
未改的641090608参数/594张量原模型严格加载、完整state逐位一致，无missing/unexpected。
R1验证器137终止未计通过、原因未证明；R2只避免一份冗余随机参数分配，
所有原权重实际materialize到CPU，不裁剪架构或预算。没有GPU计算/分割/训练。
权重保存在本地资产目录，不复制2.56GB到共享卷；SAM-MIL仍缺全856真实掩膜、原面积及成员映射。
凭据`sammil_original_sam_vit_h_local_asset_verified_r2.json`，详见`SAMMIL.md`。

22:28Z对FEATHER原始UNI固定safetensors作官方HEAD复核：匿名401/GatedRepo，
已有缓存认证403/GatedRepo；凭据不输出，不自动申请权限/接受条件或替换随机预训练。
只新核验此UNI变体，不把此结论扩展到另两个变体。
历史UNI视觉权重/版本仍未知，不能填入锁中的当前参考SHA来伪造历史提取来源。
凭据`feather_uni_official_access_recheck_r1.json`；两项状态仍是blocked，不增加训练完成数。

## 实际任务边界：HMIL/TaxoMIL与AHDMIL不可混为一类

2026-11-05T22:36Z，用原始856患者slide视图调用未改接口，HMIL与TaxoMIL均明确拒绝缺失的层级标签。
仅内存中的虚构父类控制证明：一个共同父类不符合当前接口，两细类/两粗类只能一一映射。
后者虽能通过元数据接口，也不是已有临床层级，未写入真实task或用于训练。
不能把下载GPT2或补特征说成解决了额外临床任务定义，仍保留原生方法等待真实层级/明确实验协议。

同次核实AHDMIL原始BRCA配方恰为2类，当前IDC/ILC类别数匹配；
登记全856袋最少128patch也超过原k_sample32。实际缺的是原生物理配对RGB16/高分特征契约，
不是“二分类入口不兼容”。未建立网络、消耗GPU或减少原200+200预算。
三项具体原因写入输入blocker补充表，原失败台账/25项排除/58项完成数不变。
证据`brca_hierarchy_and_ahdmil_actual_scope_r1.json`。

## 无新增作业的原WSI本地访问路线已验证

2026-11-05T22:42Z，使用原清单文件UUID从GDC官方公开接口取得一张原始TRAIN切片，
不是另建队列或替换患者。确定性IO pilot选择742个TRAIN中最小的原文件，41783236字节。
官方文件ID/名称/字节数与原清单相符，下载字节通过官方MD5校验；本地OpenSlide的quickhash、各级尺寸和downsample
也与此前GPFS源读取凭据一致。按原169行坐标完整重新读取256RGB，整袋像素SHA逐位匹配
已交付真实RGB的`97a1658624fab8f1cc5a411038c705f8af725cd7a6220c247f38f5f73d8102a2`。
此pilot不是新的UNI特征提取或原历史权重身份声明。

22:43Z，以官方只读POST搜索核对全部856个原UUID：文件名、大小、原患者ID、项目、SVS格式
和open访问均一致，各有官方MD5；原文件总887634241775字节，患者与742/69/45划分未变。
这些是全队列元数据匹配，不是856份文件都已下载/验过实际字节。
只下载一张约39.8MiB到本地/data，未复制826.7GiB队列、申请新CPU/GPU或触碰已有训练。

因此，之前源CPU作业结束/GPFS1不可挂载仍是原路径事实，但不再是继续真实WSI准备的唯一门槛。
当前本地环境能通过官方原文件路径继续；后续须逐文件校验官方MD5，并有界处理原WSI，
不能一次缓存全队列，也不能把元数据可获取误报为物理金字塔、SAM掩膜或多尺度特征已完成。
HIPT、SAM-MIL、HDMIL/AHDMIL等仍需各自原生编码器和真实物理配方核查；完成数不因此增加。

共享凭据：`gdc_original_wsi_local_access_route_r1.json`、
`gdc_original_wsi_local_pilot_verified_r1.json`、`gdc_original_cohort856_metadata_verified_r1.json`。
官方接口说明：[GDC Search and Retrieval](https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/)。

## HIPT 真WSI→物理区域→原生特征链路完整 pilot

2026-11-05T23:21:50Z，同一张原TRAIN WSI实际产生4个原生组织筛选区域，
每区4096×4096@20×、完整16×16patch，共1024行384D DINO特征及4行192D原region CLS。
保持作者完整batch256、固定真实教师及完整前向；原独立读取器的像素和张量逐位一致，
真实level0步长512、1024坐标唯一完整，native alignment通过；不是用UNI旧行生成伪网格。
23:23Z独立审计通过，备份全部小型输出1685955字节，无重跑推理或训练。

该pilot是在现有H200节点CPU侧完成，原13训练未重启、无GPU计算/新增作业。
本地与节点早期失败目录、依赖导入及NumPy日志错误均保留；成功路线原样one-shot读图，
未采用失败的分块缓存或缩小区域/推理batch。
输入条件明确为固定CLAM默认与原HIPT教师的`native_HIPT_ViT256_DINO20x_CLAM_default`，
不是UNI主组，不是历史region列表复现，TCGA预训练重叠也未被证明排除。
HIPT仍缺全856真实输入及后续MIL训练，因此总完成数仍58，不把pilot升格成方法完成。
凭据`hipt_native_real_physical_pilot_evidence_r7/independent_audit.json`，详见`HIPT.md`。

## HIPT 从单张试点进入原856队列的连续准备

2026-11-05T23:51:58Z，第二张原TRAIN切片TCGA-AC-A62V的完整7区域/1792行
原生特征独立复核通过；合计2张、11区域、2816行，仍不是HIPT训练完成。
新逐张worker在实际节点通过21项保真控制，完整队列控制另通过5项，
包含完整856患者不变以及12类错误删除目标的拒绝子例。

23:53Z已启动CPU侧原856连续准备：官方UUID逐文件MD5及传输SHA，
完整原RGB像素身份复核，原生20×组织区域、完整batch256教师提取，再独立验产物。
只在产物通过后删除本流程新下载的私有原图副本，并保存可按官方UUID恢复的收据；
第一次仅退役两端各57528662字节，既有原图、UNI/RGB、原训练、特征和失败证据不动。
没有新rjob或新MIL队列owner，未降低原方法预算，也没有把缺输入的方法再排除。

输入条件仍单列`native_HIPT_ViT256_DINO20x_CLAM_default`，不混入UNI主结果。
全部856物理文件、完整任务manifest和后续MIL训练仍待完成；混合倍率不得虚填统一步长。
连续状态见`native_hipt_full856_preparation_r1.json`，源计划、启动句柄和逐张独立audit均已保存。

## 实际混合倍率输入契约与新增cDP-MIL结果

2026-11-06T00:09:21Z，当前3张完成原图共24区域/6144行通过0.132.0逐切片stride读取。
其中TCGA-MS-A51U真实20×扫描，level0步长256；前两张40×源图步长512。
schema2明确逐slide登记并要求恰好覆盖任务全部ID，旧schema1不变；错误统一步长被拒绝。
特征、坐标、区域及患者未改，未建立3患者临床子任务；完整原队列仍856。
原模型/教师/训练器/415配置与父包逐字节相同，512包文件仅版本文件和`hipt_data.py`变化。
独立节点CPU源测试63项全通过（含作者108次逐位训练更新），离线安装包32项全通过，
均无跳过；本地137失败保留而未冒充通过。完整真实输入、task注册与后续20epoch临床训练仍待完成。
凭据及wheel见`hipt_real_mixed_stride_prefix_verified_r1.json`与`HIPT.md`最新节。

同时cDP-MIL的显式float64数值条件已完成本轮配置：17实际epoch、上限50、patience10，
原856患者/2422295行/742-69-45划分不变。锁定45患者TEST于00:02:11Z完成，
独立AUC0.9892241379、accuracy0.9333333333；59方法TEST身份和指标审计全部通过、零错误。
源请求/完整历史/输入/协议/环境已原字节保存在`results/cdpmil/source_run_evidence_r1/`。
这是当前benchmark早停配置完成，不宣称全50epoch或原论文完整复现；旧失败证据仍保留。

## 原生HIPT CUDA输入流与SAM真实权重在现有节点到位

2026-11-06T00:24:24Z，原HIPT完整256 patch batch在现有H200通过真实区域探测：
FP32、禁用TF32/AMP、原两个教师、公开device参数均cuda:0；分配峰值约1.82GiB，
原生前向1.3888秒。同CUDA独立原生序列逐位相同，CPU对照按预设组合容差通过，
不声称跨硬件位相同或整个WSI具有同样耗时。初版直接forward观测hook失败保留，
R2仅修观测并通过3控制；源方法未改。

25项worker保真/资源控制与10项队列控制通过后，00:32:49Z只交接本地输入controller，
已验6张原文件及当时CPU worker/12个训练PID均保留；该CPU切片随后正常完成。
后续原WSI已开始真实CUDA整张提取和独立审核，首张TCGA-AN-A0XW已通过。
4GiB allocator cap、40GiB GPU guard并扣除活跃未来预约，不压缩原batch/model或新增rjob。
当前仍是原856队列的输入准备，HIPT尚未完成全队列注册/下游MIL训练，不混称UNI主结果。
当前输入controller启动凭据改为`hipt_full856_stream_controller_launch_cuda_r2.json`；
实时状态仍为`native_hipt_full856_preparation_r1.json`。

同时SAM原2.56GB ViT-H已经153个SHA核验块传入该现有节点，未占共享大文件空间。
00:37:52Z在节点CPU侧完整严格加载，594张量/641090608参数、8原源码SHA、
完整checkpoint SHA/MD5和tensor映射摘要全部与本地原资产一致，无随机替代或meta残留。
未运行SAM图像前向、分割、面积/成员导出或SAM-MIL训练；全856真实掩膜仍缺。
详见`HIPT.md`、`SAMMIL.md`新增章节，及两个existing_node SAM收据。

## SAM 原 ROI 与真实 UNI 成员几何后继

2026-11-06T00:57:04Z，固定作者 TCGA preset16/4 在同一原 TRAIN WSI 上读出
527×566真实 SAM RGB ROI，两条原源码读取路径像素逐位一致，没有生成伪mask。
01:15:46Z已核对同袋完整169行原坐标、169×1024 UNI特征及登记的1024-level0像素视野，
明确它不同于作者原CLI512；原整数映射因子32、adjusted_patch_size32。
此为显式 UNI 条件，不冒称原 ResNet/512px 论文预处理结果。

完整原ViT-H分割试点正在同一现有节点等待资源，保留5次图像编码、2048点提示及64点batch；
独立CPU后继只等待其真实成功文件，随后逐mask审计并运行原始成员/均值/面积函数与实际importer对照。
尚无掩膜/成员成功文件，不据启动凭据增加完成数；全856及后续训练仍未完成。
原训练、主调度器、HIPT/CONCH流均未重启；25项任务不匹配排除不变。
凭据`sammil_original_wsi_roi_pilot_verified_r1.json`、
`sammil_original_membership_geometry_verified_r1.json`、
`sammil_original_full_budget_mask_pilot_launch_r1.json`、
`sammil_original_membership_pilot_launch_r1.json`。

## GMMamba 全856原分组容量核验

2026-11-06T01:23:21Z，全部856患者/2422295原UNI行通过作者原始坐标KMeans10
及未修改的容量接口，组输出逐位相同；没有空组或单例，最小实际组含8个patch。
逐袋原坐标SHA全匹配，未减组、补实例、改坐标或丢患者。
这是原始输入前置核验，不是将1024维UNI冒充原生512维模型输入。
仍需用户确认是否另列CONCH512补充实验；未获批准前不切换FM、不加投影或提交训练。
方法仍记blocked，25项任务不匹配排除不变，不因输入诊断增加方法完成数。
证据`gmmamba_complete856_original_group_capacity_verified_r1.json`，详见`GMMAMBA.md`。

## 共享现有原WSI流的独立SAM ROI读取

2026-11-06T01:28Z，在现有节点单独运行CPU读图进程，观察HIPT已经完成官方MD5/传输SHA
核验的原WSI，用自己的临时硬链接租约读取作者原TCGA16/4完整ROI并保存真实RGB/轮廓。
不改变HIPT或训练，不新增下载或rjob；独立回读核验后只移除自身私有租约名称。
01:29:46Z首张原TRAIN切片的全部像素、轮廓、参数与既有独立pilot逐位相同，原图SHA未变。
这使后续可从同一原图流准备SAM输入，而不必等HIPT暂存清除后再整批重复下载。
之前已清除暂存的原图仍缺ROI；当前未证明全856或任何新增掩膜/成员/训练完成。
状态`native_sammil_original_roi_preparation_r1.json`，审计
`sammil_existing_stream_generalized_roi_pilot_independent_audit_r1.json`；详见`SAMMIL.md`。

## CONCH全队列完成、MIL-Adapter第60项及SAM真实掩膜试点

2026-11-06T01:30:28Z，CONCH完整856袋/2422295原patch行完成；全归档/坐标再次核验后，
MIL-Adapter进入现有调度器并完成原生20epoch、320次更新和01:32:49Z锁定TEST。
作者原k8/class选出16个TRAIN支持病例，不冒称每轮训练完整742病例。
保留完整袋、512维、AdamW/cosine及仅最终轮选模。45患者TEST AUC.9870689655，
accuracy.9333333333；该原生CONCH补充组不混入UNI主比较，不宣称论文复现。
完整源运行证据在`results/miladapter/source_run_evidence_r1/`；60项独立TEST审计零错误。

SAM首版绝对导入缺口及失败留证。R2只修私有包名连接并恢复别名，9控制通过，原模型不变。
01:35:50Z在真实原ROI完成5次完整FP32编码和2048点/32×64批次，生成127个真mask。
01:36:19Z独立核对全部mask面积/字节，并执行作者原成员/均值函数：169原UNI行产生33个
非空组及202增强行，与实际importer的特征/坐标/flags/面积逐位相同。
小体积真实产物保存在`sammil_real_native_pilot_evidence_r2/`；仍只有一张输入试点，
全856及SAM-MIL训练未完成，不增加SAM方法完成数。
CATE配对CONCH视觉输入已齐，但PIM/SIM权重实验选择仍须明确，不自行编造。

随后01:40:02Z，Loss Attention的显式冻结UNI条件完成独立TEST：
100epoch、每轮742完整TRAIN袋、74200次连续Adam更新且无重置；原instance weight.1、
sum归约及VAL error/bag CE选模保留。AUC.9892241379、accuracy.9111111111。
源请求/协议/输入/完整历史原字节保存在`results/loss_attention/source_run_evidence_r1/`。
当前61项已完成TEST独立重算全部通过、零错误；60项旧审计保留为
`completed_predictions_audit_before_new61_r1.json`。25项任务不匹配排除不变。

## SAM 全队列输入流与公开原图读取路径核查

2026-11-06T01:54Z，SAM原生输入流覆盖固定856患者/2422295原行，16项保真控制通过；
按真实ROI逐张原模型GPU分割、原作者CPU成员处理及独立字节审计，不建临床子集。
01:57:27Z，通用worker重放首张原病例，127个mask及69份成员/增强数组与原试点逐位相同。
仍只计1个患者；原生2048点（32批×64点）、完整ViT-H和既有资源准入保持，完整输入未齐不训练。
详见`SAMMIL.md`全队列节及`native_sammil_full856_preparation_r1.json`。

当前GDC原图读取仍有明显慢读。按[AWS官方开放数据目录](https://registry.opendata.aws/tcga/)
核对公开TCGA对象，实际全部856原UUID/文件名的对象头存在且大小与冻结GDC元数据一致，
总887634241775字节；这不是全部文件字节已验证。GCS文档旧路径实测404，不假设可用。
GDC本身也实测支持精确HTTP206 range，两个真实原range与本地保留字节相同。

公开AWS的单连接完整试读超过180秒诊断时限，失败保留且未切换现有流。
随后8路1MiB range在108.89秒完整重组原TRAIN WSI41783236字节；每块HTTP206、
Content-Range/Length及If-Match核对，整文件MD5和SHA256与独立GDC原文件相同：
`69577c95f8b87e992df71f0bcf3d0635`、
`3a4455515ba916bbf52f29f7cd77f6722d4b218d96e03c012f874351e7ccee49`。
multipart ETag仅作版本条件，不冒充MD5；不将不同负载观察宣称为稳定因果提速比。
已开始当前慢读原图的独立并行预取，放在新私有cache，不覆盖旧下载、不中断原stager，
后续只有完整GDC MD5/实际SHA核验通过才考虑严格交接。本节预取启动时尚未切换来源；后续实际恢复见下节。
证据`aws_public_original_cohort856_headers_verified_r1.json`、
`aws_original_parallel_pilot_verified_r1.json`；预取收据成功后才进入
`aws_original_wsi_acquisitions_r1/`，缺失/失败不当成功。

## 原图真实超时后的输入恢复与第二张原生SAM结果

旧GDC stager日志明确报`TimeoutError: The read operation timed out`，原输入controller
也以保留失败状态终止。两者PID/starttime均已核实终止，并非仅凭观察超时判断，
没有向原训练、GPU预备worker或rjob发送停止信号。16张已完成HIPT输入原字节/审计保留，
完整旧失败state备份为`native_hipt_full856_before_aws_recovery_r3.json`，旧部分下载不删。

当前原切片TCGA-AC-A8OS经8路141个完整range取得146991482字节，实际452.06秒。
完整MD5匹配冻结GDC值`2970daf0be5aeb929a94f855a15ba9e9`；本次实际SHA256为
`e2d8239df8416434e57d6479949d16f8c20395ca3e4af1edfa9aba68a87279ea`。
这是此次文件实际测量，不冒称预先知道了该文件独立完整SHA。
通过完整原MD5和同一未修改HIPT传输程序的local/remote SHA检查之后才使用；
新来源另有actual-source收据，不把镜像读取谎称为实际从GDC下载。

15项实际控制（原流10项+新原字节/暂存回收5项）通过。02:20Z新controller仅恢复输入阶段，
保留原完整856顺序及16张完成前缀，原CUDA worker、准入、独立审核和原暂存退休函数AST不变。
当前切片已进入同一原生256batch CUDA worker；全任务注册及后续训练仍未完成。
仅在完整原MD5和远端传输核验成功后自动回收本次141个已校验分段约140.2MiB及冗余硬链接名；
正常WSI暂存仍保留到原生输入独立审计通过，旧失败部分下载/源WSI/权重不动。
这些新临时分段可由同一公开原对象恢复，不是删除原数据集或核验失败证据。

新输入controller启动凭据`hipt_full856_stream_controller_launch_aws_r3.json`；
当前状态文件仍为`native_hipt_full856_preparation_r1.json`。真实来源补充收据在
`native_hipt_full856_actual_aws_source_receipts_r3/`；完整源worker/审核SHA绑定未改。

相同原WSI同时被既有SAM ROI读取器读取，随后全预算SAM分割与原生成员审核完成：
第二不同患者879原行、221mask、135非空组、1014增强行，原面积/均值/flags/coords逐位一致。
其实测映射因子64、adjusted_patch_size16，未套首图32/32。
SAM现已实证2/856、1048原行、348mask；不重复计首图重放，仍未完成全856输入或SAM-MIL训练。

部分早期传输收据的`source_pipeline_or_training_interrupted=false`字段命名过宽：
它只能说明该传输代码没有发送停止/重启信号，不能证明外部旧输入进程一直存活。
旧流在此期间确实超时退出，已按真实终态恢复。原收据保留以保持payload摘要引用，
澄清见`transport_probe_historical_liveness_clarification_r1.json`；不改变任何字节/MD5/SHA或患者断言。

## 原图失败块精确恢复、旧ROI补齐及一次性保护

TCGA-A8-A0AB原图143块中142块通过、1块TLS握手超时，实际旧下载和控制器已终止。
逐块复核已有142块SHA，只重新取得缺失1MiB；完整GDC MD5与本次实测SHA均通过后，
沿用原HIPT运输、控制器及CUDA worker继续，未用局部字节或成功退出码替代完整原图证明。
失败记录和旧控制器state保留，恢复启动为
`hipt_full856_stream_controller_actual_range_recovery_resume_r4.json`。

旧原图暂存早于SAM观察器退休导致的15个ROI缺口，现由独立输入补齐队列处理，
只选择固定16图历史前缀里的真实缺口，不把正在增长的完成前缀加入抢写范围。
每张要求原GDC MD5和旧HIPT独立完整SHA同时一致，调用同一冻结原ROI函数。
已完成输入范围回读时，HIPT为19/856；SAM为8个不同患者、3176原行、1069真mask，
其中15个旧缺口已补齐4个。所有完整原生预算、原面积/均值/flags/coords条件保持。
`sammil_actual_unique_prefix_independent_recheck_r4.json`重新核对全部8份唯一患者的
ROI/mask/member收据SHA、实际ROI数组字节和原科学驱动SHA；没有把同一患者重放重复计数。

R4运输控制器另补全失败块恢复功能，原GPU worker、准入、审核及退休函数保持不变，
控制器只有IO路径/绑定版本变化；32项相关保护检查通过。
一次性观察器只跟踪明确的旧PID+starttime，不向旧进程发送信号：
旧控制器活着时绝不替换；只有旧下载和控制器均实际终止、完整失败清单一致、
错误明确来自原图下载且没有原模型worker失败时才允许启动新输入控制器。
成功完成全856时不会重启，其他失败也不会盲目重试。
当前R4是待命保护，不把它写成已发生交接；真实状态见
`hipt_full856_actual_failure_handoff_state_r4.json`，绑定/启动见
`hipt_full856_range_recovery_standby_r4.json`及
`hipt_full856_verified_failure_handoff_watch_launch_r4.json`。

现有GPU节点直读同一公开原图的1MiB网络诊断实际超时，未获得可验证字节，
因此未切换读取端，也未新建rjob。诊断失败收据
`original_public_aws_existing_node_network_probe_r1.json`保留，不能称为已提速。
本轮临床实验仍61完成；输入准备进度不计作新增训练完成。
