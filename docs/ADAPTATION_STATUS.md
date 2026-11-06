# 接入状态与剩余范围

当前候选 164 条，已写入 135 个家族，执行配置 285 个。正式验收仍为0.62的117条／219配置，18条候选待完整验收，29条待实现。

CPU 合成验证、GPU 验证、真实 WSI 实验和论文指标复现分别登记。当前执行配置不等于原论文复现数。

## 已写入执行配置

| ID | 家族 | 软件状态 | 配方 |
|---|---|---|---|
| ahdmil_camelyon16_source_repaired | ahdmil | implemented_unvalidated | Full HDMIL DMIN preparation then actual two-expert AHDMIL joint optimization, NumPy attention/hard distillation, alternating Adams and simultaneous crossEMA, source retention-gated VAL selection/reset; explicit RCC lower_ratio and entry-name repairs |
| ahdmil_nsclc_source_repaired | ahdmil | implemented_unvalidated | Full HDMIL DMIN preparation then actual two-expert AHDMIL joint optimization, NumPy attention/hard distillation, alternating Adams and simultaneous crossEMA, source retention-gated VAL selection/reset; explicit RCC lower_ratio and entry-name repairs |
| ahdmil_brca_source_repaired | ahdmil | implemented_unvalidated | Full HDMIL DMIN preparation then actual two-expert AHDMIL joint optimization, NumPy attention/hard distillation, alternating Adams and simultaneous crossEMA, source retention-gated VAL selection/reset; explicit RCC lower_ratio and entry-name repairs |
| ahdmil_rcc_source_repaired | ahdmil | implemented_unvalidated | Full HDMIL DMIN preparation then actual two-expert AHDMIL joint optimization, NumPy attention/hard distillation, alternating Adams and simultaneous crossEMA, source retention-gated VAL selection/reset; explicit RCC lower_ratio and entry-name repairs |
| hdmil_camelyon16_source | hdmil | implemented_unvalidated | Complete released two-stage DMIN512/256/ChebyKAN/SmoothTop1/Gumbel self-distillation then native RGB16 MobileNetV4 LIPN; TRAIN weighted replacement, two Adams200+200max, source VALmeanCE stages and full-state teacher transfer |
| hdmil_nsclc_source | hdmil | implemented_unvalidated | Complete released two-stage DMIN512/256/ChebyKAN/SmoothTop1/Gumbel self-distillation then native RGB16 MobileNetV4 LIPN; TRAIN weighted replacement, two Adams200+200max, source VALmeanCE stages and full-state teacher transfer |
| hdmil_brca_source | hdmil | implemented_unvalidated | Complete released two-stage DMIN512/256/ChebyKAN/SmoothTop1/Gumbel self-distillation then native RGB16 MobileNetV4 LIPN; TRAIN weighted replacement, two Adams200+200max, source VALmeanCE stages and full-state teacher transfer |
| hdmil_rcc_source | hdmil | implemented_unvalidated | Complete released two-stage DMIN512/256/ChebyKAN/SmoothTop1/Gumbel self-distillation then native RGB16 MobileNetV4 LIPN; TRAIN weighted replacement, two Adams200+200max, source VALmeanCE stages and full-state teacher transfer |
| mgtrans_source_repaired | mgtrans | implemented_unvalidated | Complete released192width/3heads/512anchor two-scale graph Transformer, actual3layer GENConv, negativeCE semantic consistency and32slide RenyiMI window; original Adam/200maxepochs/VALerror latest-tie selection; explicit device/API/latent-query repairs |
| zoommil_bright_source_repaired | zoommil | implemented_unvalidated | Complete original three-scale dual-gated/100sample differentiableTopK/Kronecker/sum ZoomMIL; source weighted TRAIN replacement sampler, Adam/plateau meanVALCE/100fixed epochs and weightedF1 selection; explicit saver/device ports, native8readers, cached features only |
| zoommil_crc_source_repaired | zoommil | implemented_unvalidated | Complete original three-scale ZoomMIL with CRC K16, actual weighted sampler/Adam/plateau and100fixed epochs; missing source VALloss-saver input explicitly repaired; no two-level or onlineFM substitution |
| augdiff_amil_released_time_reconstruction | augdiff | implemented_unvalidated | Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original amil CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and released_time_reconstruction |
| augdiff_amil_schedule_repaired_reconstruction | augdiff | implemented_unvalidated | Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original amil CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and schedule_repaired_reconstruction |
| augdiff_lossattn_released_time_reconstruction | augdiff | implemented_unvalidated | Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original lossattn CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and released_time_reconstruction |
| augdiff_lossattn_schedule_repaired_reconstruction | augdiff | implemented_unvalidated | Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original lossattn CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and schedule_repaired_reconstruction |
| augdiff_dsmil_released_time_reconstruction | augdiff | implemented_unvalidated | Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original dsmil CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and released_time_reconstruction |
| augdiff_dsmil_schedule_repaired_reconstruction | augdiff | implemented_unvalidated | Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original dsmil CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and schedule_repaired_reconstruction |
| vlsa_conch_source_repaired | vlsa | implemented_unvalidated | Full source CONCH native512 VLFAN/Text-TaskRes,4-base ordinal interpolation, IFMLE+EMD,32-patient window Adam/10epochs/final checkpoint; explicit TRAIN-only terminal time-edge repair |
| slip_clip_vit_b16_source_repaired | slip | implemented_unvalidated | Complete original SLIP actual native512 CLIP-ViT-B16 text/context/tissue attention and C*C contrastive objective, source prompt-only SGD10/final-epoch with explicit unified-loader and precision/device ports |
| slip_clip_rn50_source_repaired | slip | implemented_unvalidated | Complete original SLIP actual native1024 CLIP-RN50 text/context/tissue attention and C*C contrastive objective, source prompt-only SGD10/final-epoch with explicit unified-loader and precision/device ports |
| amiml_paper_reference | amiml | implemented_unvalidated | Complete four TRAIN-only morphology clusters, full native8-channel minmax/decoder/residual/selfattention and explicit V8->C paper-output reconstruction, weighted-logit CE/Adam/300/plateau10/VAL-only two-level selection |
| amiml_source_repaired | amiml | implemented_unvalidated | Full AMIML with explicitly repaired class/input dimensions and released probability-CE/Adam/300/plateau5 control, all four clusters and source checkpoint/early-stop semantics; missing original loader reconstructed |
| dtmil_source_crop_reference_train | dtmil | implemented_unvalidated | complete pinned DT-MIL full spatial/deformable-encoder/ordinary-decoder/query-head, released crop/binary contract with explicit finite-padding port; complete named50epoch AdamW/LR-group/StepLR reference training because public trainer is absent |
| dtmil_repaired_reference_train | dtmil | implemented_unvalidated | complete pinned DT-MIL and physical cached-view/data augmentation recipe; explicit current-height crop/C-way labels/finite-padding ports, full reference AdamW/LR-groups/StepLR/clipping/firstVAL and exact resume |
| codmil_source_repaired | codmil | implemented_unvalidated | complete author two-scale chain-of-diagnosis and masked-normal-text objective, physical row-map/text binding, explicit C-way/full-D/T input ports and complete original one-bag Adam/VAL-loss/stopping recipe with exact state resume |
| ins_frozen_source_repaired | ins | implemented_unvalidated | complete frozen-feature INS route with actual native ResNet18 feature-pseudoimage/query-key/prototype/queue/partial-label/SupCon kernels and full source two-phase SGD+bag-CE recipe; named weak-label, valid-queue, two-class bag-head and scratch-initialization repairs |
| sclwc_paper_reference | sclwc | implemented_unvalidated | complete explicitly reconstructed paper CDA/PNM/WSCL; native512/256 class-specific attention, active live-anchor cross-slide objective and real TRAIN-only typed banks; full balanced one-bag Adam/cosine/validation stopping and exact resume |
| sclwc_literal_source_control | sclwc | implemented_unvalidated | literal pinned source model and actual original train_loop, full-D input port; preserves missing PNM and disabled WSCL as an explicitly named incomplete-source control; complete author balanced Adam/cosine/selection loop |
| deepgraphsurv_agcn | deepgraphsurv | implemented_unvalidated | literal2018 adaptive feature graph attention32/1 and native256/256/128AGCN with complete original mixed-event window trainer, TF1Adam, update-count LR/BN schedules and full checkpoint; full-dimensional frozen-FM input condition and explicit VAL-only selection/inference stream corrections |
| triagemil_source | triagemil | implemented_unvalidated | complete original MASS quality/four-axis/FPS selection, semantic hierarchical hypergraphs and native full-D six-head/four-layer model; source Cox/AdamW/accumulation/per-epoch AMP/VAL-first-tie/minimum-epoch stopping with exact resume |
| c2mil_source | c2mil | implemented_unvalidated | full original adaptive TRAIN-reference bias removal, causal graph three-view network, five losses and native batch/Adam/VAL-selection recipe; complete dynamic cache and exact epoch resume |
| gatmamba_source | gatmamba | implemented_unvalidated | complete original physical GAT/Mamba1.2.0.post1 network, source Cox+all-parameterL1 and full shuffled-patient Adam/plateau/last-tie selection recipe; explicit full-D input port and original inactive edge conditioning |
| gatmamba_edge_conditioned | gatmamba | implemented_unvalidated | complete native GAT-Mamba recipe with explicit edge_dim16 and singleton normalization corrections; actual physical/subtype/FM-space graph provenance, all registered feature channels |
| protosurv_source | protosurv | implemented_unvalidated | complete original dual-view graph/prototype network, three objectives and full source survival sampler/augmentation/gc/Adam/cosine/selection; original supported category/time domain |
| protosurv_repaired | protosurv | implemented_unvalidated | complete ProtoSurv recipe with named full-vocabulary compatibility, local pad RNG, tied-time Breslow and partial-tail corrections; actual graph/category/prototype provenance |
| pathrwkv_source | pathrwkv | implemented_unvalidated | complete pinned TimeMix/ChannelMix/2D position/MTL model and heterogeneous CE/Huber/source-Cox, source sampling/AdamW/cosine/early stop; original chunk shift resets and Cox broadcasting retained explicitly |
| pathrwkv_repaired | pathrwkv | implemented_unvalidated | complete native PathRWKV heterogeneous recipe with explicit observed masks, Breslow-Cox repair and bounded GPU streaming carrying both token shifts and WKV state; actual author CUDA backend or explicitly selected CPU tensor reference |
| c2aug_transmil_paper_reference | c2aug | implemented_unvalidated | complete named paper-reference recipe: all three TRAIN-only cross-bag augmentations, original TransMIL, EMA teacher, active bag FIFO InfoNCE, learned prototypes and all-group forward KL; explicit paper/source differences |
| agdrmil_yaml_source | agdrmil | implemented_unvalidated | complete released YAML hard grading/cosine/low-mask/high+full Sigmoid-CE recipe, native Adam and source shuffle/validation RNG; no invented grader gradient |
| agdrmil_tcga_source | agdrmil | implemented_unvalidated | complete source model/trainer with explicit README TCGA thresholds/mask/dual-loss coefficient; hard-gradient and stochastic-eval source behavior retained |
| h2mil_source_compat | h2mil | implemented_unvalidated | complete original two-stage RAConv/IHPool and summed-batch Adam recipe; explicit shape repair, literal four-probability CE and binary metric path |
| h2mil_repaired | h2mil | implemented_unvalidated | complete original hierarchy/model/optimizer recipe with explicit shape, C-way head and logits-CE repair; hard pooling gradient boundary retained |
| hipt_lgp_frozen_source | hipt | implemented_unvalidated | complete original local-global pretrained HIPT classifier, actual frozen teacher, balanced sampler and source CE/32-per-bag Adam; default final epoch |
| hipt_lgp_finetune_source | hipt | implemented_unvalidated | complete original pretrained HIPT classifier with local ViT finetuning, native physical pyramids and balanced source CE/32-per-bag Adam |
| simil_source | simil | implemented_unvalidated | complete original dual-branch SI-MIL and coupled BCE/KD/attention recipe; literal published TopK derivative, TRAIN-only PathExpert statistics |
| simil_full_vjp | simil | implemented_unvalidated | complete original SI-MIL model and optimizer with explicitly corrected full Gaussian TopK VJP, TRAIN-only PathExpert statistics |
| mamba2dmil_source | mamba2dmil | implemented_unvalidated | complete native physical 2D source scan, literal CPU backward; original CE/AdamW/cosine and discrete NLL-per-patient recipe |
| mamba2dmil_autograd | mamba2dmil | implemented_unvalidated | complete native 2D model and recipes, exact published CPU forward with explicitly corrected automatic derivatives |
| mamba2dmil_paper | mamba2dmil | implemented_unvalidated | complete native model/recipes, explicit vertical downsweep and derivative corrections independently checked against primary horizontal/vertical equations |
| ttis_abmil_source | ttis | implemented_unvalidated | original TTIS six-head source CE/LookaheadAdamW/cosine recipe, complete TEST-only physical/semantic FPS and feature-level fusion |
| ttis_clam_source | ttis | implemented_unvalidated | original TTIS six-head source CE/LookaheadAdamW/cosine recipe, complete TEST-only physical/semantic FPS and feature-level fusion |
| ttis_rrt_source | ttis | implemented_unvalidated | original TTIS six-head source CE/LookaheadAdamW/cosine recipe, complete TEST-only physical/semantic FPS and feature-level fusion |
| ttis_s4_source | ttis | implemented_unvalidated | original TTIS six-head source CE/LookaheadAdamW/cosine recipe, complete TEST-only physical/semantic FPS and feature-level fusion |
| ttis_transmil_source | ttis | implemented_unvalidated | original TTIS six-head source CE/LookaheadAdamW/cosine recipe, complete TEST-only physical/semantic FPS and feature-level fusion |
| ttis_wikg_source | ttis | implemented_unvalidated | original TTIS six-head source CE/LookaheadAdamW/cosine recipe, complete TEST-only physical/semantic FPS and feature-level fusion |
| ttis_clam_full_loss | ttis | implemented_unvalidated | complete original CLAM combined bag/instance objective as explicitly named training repair, followed by original TTIS selection/fusion |
| msattention_source | msattention | implemented_unvalidated | original three-depth residual attention, full CE/Adam supplementary recipe; explicit frozen-WSI transfer with controlled stochastic eval dropout |
| msattention_repaired | msattention | implemented_unvalidated | complete original residual attention/CE/Adam recipe with named mode-aware eval dropout repair and independent validation selection |
| pmil_shapley_abmil_source | pmil_shapley | implemented_unvalidated | native Shapley marginal logits, progressive pseudo bags and fresh multi-round ABMIL, original gradient carry and dual validation selection |
| pmil_shapley_abmil_repaired | pmil_shapley | implemented_unvalidated | complete PMIL Shapley/plateau/progressive/multi-round recipe with named per-pseudo-gradient reset and constant-score repair |
| pmil_shapley_clam_sb_source | pmil_shapley | implemented_unvalidated | complete Shapley/progressive/EM-round recipe with actual author CLAM-SB instance constraints and original pseudo-gradient carry |
| pmil_shapley_clam_sb_repaired | pmil_shapley | implemented_unvalidated | full PMIL CLAM-SB recipe including instance CE, named gradient and constant-score repairs, native round/plateau policy |
| pmil_shapley_clam_mb_source | pmil_shapley | implemented_unvalidated | complete author CLAM-MB/Shapley recipe with source branch0 candidate search, gradient carry and full multi-round training |
| pmil_shapley_clam_mb_repaired | pmil_shapley | implemented_unvalidated | full PMIL CLAM-MB with explicit target-branch ranking, per-pseudo-gradient reset and constant-score repairs |
| mammoth_mean | mammoth | implemented_unvalidated | native multi-head slot routing/shared low-rank experts, feature mean, full weighted AdamW/cosine classification recipe |
| mammoth_max | mammoth | implemented_unvalidated | native MAMMOTH experts and paper single-winning-slot logits, full weighted AdamW/cosine recipe |
| mammoth_abmil | mammoth | implemented_unvalidated | native MAMMOTH and author gated attention/classifier, full weighted AdamW/cosine/CE recipe |
| mammoth_clam | mammoth | implemented_unvalidated | native MAMMOTH and author CLAM instance constraints, binary SmoothTop1 and complete weighted AdamW/cosine recipe |
| mammoth_dsmil | mammoth | implemented_unvalidated | native MAMMOTH and author critical-instance dual stream, source averaged-logit CE and full weighted AdamW/cosine |
| mammoth_transmil | mammoth | implemented_unvalidated | native MAMMOTH and author Nystrom/PPEG TransMIL, complete weighted AdamW/cosine/CE recipe |
| mammoth_transformer | mammoth | implemented_unvalidated | native MAMMOTH and author CLS global transformer/MLP, complete weighted AdamW/cosine/CE recipe |
| mammoth_ilra_repaired | mammoth | implemented_unvalidated | native MAMMOTH and author ILRA/GAB/NLP; named routing-width128 repair of invalid256/256 YAML, full weighted AdamW/cosine |
| campanella_rnn | campanella_rnn | implemented_unvalidated | complete trainable ResNet34 hard-MIL selection, validated best CNN, native recurrent final-step CE and source Adam/SGD two-stage recipe on real streamed ROIs |
| monai_mean_tutorial | monai_mil | implemented_unvalidated | native trainable ResNet50 mean-logit baseline; cumulative ordinal BCE, complete AdamW/cosine RGB tutorial |
| monai_max_tutorial | monai_mil | implemented_unvalidated | native trainable ResNet50 max-logit baseline; cumulative ordinal BCE, complete AdamW/cosine RGB tutorial |
| monai_attention_tutorial | monai_mil | implemented_unvalidated | native trainable ResNet50 attention baseline; cumulative ordinal BCE, complete AdamW/cosine RGB tutorial |
| monai_transformer_tutorial | monai_mil | implemented_unvalidated | full native trainable ResNet50 plus dependency Transformer; ordinal BCE, two AdamW groups and epoch cosine, raw full-level stochastic tiling |
| monai_pyramid_tutorial | monai_mil | implemented_unvalidated | complete native ResNet50 four-stage feature-map pyramid Transformer, ordinal BCE and full two-group AdamW/cosine RGB tutorial |
| stablemil_classification_source | stablemil | implemented_unvalidated | full physical fusion/semantic/local-global/RoPE network and Adam/CE; source cross-epoch accumulation tails, named execution wiring fixes |
| stablemil_classification_repaired | stablemil | implemented_unvalidated | complete classifier recipe with explicit degenerate/small-bag and tail accumulation repairs |
| stablemil_survival_source | stablemil | implemented_unvalidated | full hazard network and original clamped NLL/gc/Adam loop with explicit missing launcher/model wiring completion and TRAIN-only bins |
| stablemil_survival_repaired | stablemil | implemented_unvalidated | complete StableMIL hazard recipe with explicit physical small-bag and tail weighting repairs, patient NLL selection |
| setmil_source | setmil | implemented_unvalidated | full author physical PPE/TPMF/iRPE, CE/AdamW/StepLR and spatial augmentation; original stale-height crop and all-subset cached views retained |
| setmil_repaired | setmil | implemented_unvalidated | complete SETMIL recipe, explicit corrected center crop and TRAIN-only cached views; coherent checkpoints and sealed test evaluation |
| gtp_source | gtp | implemented_unvalidated | author GCN/BN, mincut/orthogonality and 3-blockViT, full Adam/MultiStepLR and batch8/drop_last; source graph/BN conditions retained, explicit frozen-FM input |
| gtp_repaired | gtp | implemented_unvalidated | full author GTP recipe and true gradient/LRP GraphCAM; explicit zero-edge/singleton-BN/constant-map repairs with otherwise identical source path |
| probsa_abmil_source | probsamil | implemented_unvalidated | author-library Gaussian attention MC-BCE plus graph KL, cyclical coefficient and Adam/LinearLR; source mean-logit inference; padded sparse KL repaired |
| probsa_abmil_probmean | probsamil | implemented_unvalidated | author-library full ABMIL ProbSA recipe; paper Eq19 mean-probability inference, sparse/padding KL repairs and fixed query random stream |
| probsa_transformer_source | probsamil | implemented_unvalidated | author-library two-layer global transformer plus sampled Gaussian attention/graph KL, cyclical Adam recipe; source mean-logit inference |
| probsa_transformer_probmean | probsamil | implemented_unvalidated | full global-transformer ProbSA author-library recipe; paper Eq19 predictive mixture and explicit sparse/padding/SDPA repairs |
| micromil_frozen | micromil | implemented_unvalidated | explicit frozen-FM WSI transfer; author DCE, hard-Gumbel representatives and cosine graph, original single-head GAT, full source BCE/Adam and validation F1 |
| loss_attention_classical | loss_attention | implemented_unvalidated | paper A2/A4 classical feature MLP, author shared class/attention head, bag CE plus .1 weighted-instance CE sum, full Adam; explicit WSI feature adaptation and validation-only error/CE selection |
| loss_attention_mil_source | loss_attention | implemented_unvalidated | MNIST source-head WSI adaptation, shared classifier and bag CE plus 2 weighted-instance CE mean, complete source Adam recipe; validation replaces source test-based selection |
| loss_attention_temporal | loss_attention | implemented_unvalidated | image-temporal recipe explicitly adapted to WSI bags: shared head, weighted-instance CE mean, aligned TRAIN EMA consistency, original schedules and legacy-Adam state reset every epoch |
| pamil_prototype_source | pamil_prototype | implemented_unvalidated | author dual branches, source CE-on-softmax and one-center collate clustering, warm/joint/push/last; train instance annotations; explicit bounded atomic push repairs |
| pamil_prototype_repaired | pamil_prototype | implemented_unvalidated | author full staged recipe with explicit CE-on-logits, local KMeans and class-specific instance mining repairs; train-only extra instance supervision |
| pamil_prototype_multilabel | pamil_prototype | implemented_unvalidated | author independent sigmoid/BCE route, two-level KMeans and complete warm/joint/annotation-filtered push/last; explicit instance-index/device repairs |
| smmile | smmile | implemented_unvalidated | author spatial conv/detection, SP sampling and score dropout, validation-best two-stage refinement/MRF with reset Adam and source validation objective; explicit numerical repairs |
| smmile_binary | smmile | implemented_unvalidated | author one-positive-class binary branch, native basic-plus-augmentation BCE, optional negative-bag consistency, two-stage pseudo-label refinement and label-free instance outputs |
| smmile_multilabel | smmile | implemented_unvalidated | author independent multilabel sigmoid route, original bi-tempered augmentation, combination-balanced sampling, two-stage refinement/MRF and source validation objective |
| momil | momil | implemented_unvalidated | author three-order Mamba2, literal channel reverse and padded-token fusion; explicit constructor-name repair and complete CE/Adam source last-or-validation checkpoint policy |
| momil_survival_extension | momil | implemented_unvalidated | explicit author survival-head and generic source NLL trainer composition; train-only intervals and flushed tail accumulation, not a published working MoMIL survival factory |
| recomil_single | recomil | implemented_unvalidated | author single-scale contextual prototypes, MLP reduction, rare gate, orthogonal loss and full warmup/tau/freeze Adam recipe; train-only clustering |
| recomil_graph | recomil | implemented_unvalidated | author single-scale contextual prototypes and graph prototype reduction with complete source recipe |
| recomil_dual | recomil | implemented_unvalidated | author dual-scale MLP prototype branches and learned sigmoid fusion, full separate branch schedules and objectives |
| recomil_dual_graph | recomil | implemented_unvalidated | author dual-scale graph prototype reduction and learned fusion with complete source recipe |
| mammil | mammil | implemented_unvalidated | author Mamba2 topology scans and GENConv, explicit full-layer-loop repair, frozen HNSW/MST/DFS/BFS graphs, source RAdam and CE/NLL |
| mammil_source | mammil | implemented_unvalidated | literal published zip(layers,ppeg) execution; original single-slide graph and RAdam CE/NLL, inactive final scan explicitly retained |
| mammil_patient_union | mammil | implemented_unvalidated | explicit corrected-loop patient survival extension using disjoint physical slide graphs and concatenated traversals; source NLL/RAdam |
| mfemil_mean | mfemil | implemented_unvalidated | author shared adapter, window-masked normalized-feature reconstruction and MeanMIL; joint CE/NLL and Adam |
| mfemil_max | mfemil | implemented_unvalidated | author shared adapter, MFE decoder and MaxMIL; joint CE/NLL and Adam |
| mfemil_abmil | mfemil | implemented_unvalidated | author shared adapter, MFE decoder and DAttention; joint CE/NLL and Adam |
| mfemil_clam_sb | mfemil | implemented_unvalidated | released MFE-CLAM-SB bag-only head with shared adapter and reconstruction; instance loss not used by source MFE recipe |
| mfemil_clam_mb | mfemil | implemented_unvalidated | released MFE-CLAM-MB bag-only head with shared adapter and reconstruction; instance loss not used by source MFE recipe |
| mfemil_transmil | mfemil | implemented_unvalidated | released survival MFE-TransMIL, shared adapter and reconstruction, censored NLL, train-only intervals and Adam |
| msmmil | msmmil | implemented_unvalidated | author AFWMamba three scan branches with shared BiAttn; CE, source RAdam/Lookahead, two-bag accumulation and validation-loss selection |
| mambamil_srmamba | mambamil | implemented_unvalidated | author SR-Mamba transpose-padded two-branch scan; explicit torch/CUDA backend, full bag CE or discrete NLL with training-only bins |
| mambamil_bimamba | mambamil | implemented_unvalidated | author Bi-Mamba forward/backward scan; explicit backend, full bag CE or discrete NLL with patient accumulation |
| mambamil_unidirectional | mambamil | implemented_unvalidated | author unidirectional Mamba ablation; explicit backend, full bag CE or discrete NLL with patient accumulation |
| dgmil | dgmil | implemented_unvalidated | author dynamic extreme-instance projection and negative-cluster Mahalanobis; stable unique ties, train-only inference density, raw-score AUROC |
| dgmil_source_extremes | dgmil | implemented_unvalidated | released heapq first-match extreme selection and full-batch Adam; train-only inference density repair, raw-score AUROC |
| pamoe_transmil | pamoe | implemented_unvalidated | author TransMIL with explicit batch/expert indexing repair; tissue-prior CE, author batch Cox or classification CE, event sampler, AdamW/cyclic cosine |
| pamoe_transmil_source | pamoe | implemented_unvalidated | literal released TransMIL expert slicing; tissue-prior CE, author batch Cox or classification CE, event sampler, AdamW/cyclic cosine |
| pamoe_transformer | pamoe | implemented_unvalidated | author Transformer with explicit batch/expert indexing repair; tissue-prior CE, author batch Cox or classification CE, event sampler, AdamW/cyclic cosine |
| pamoe_transformer_source | pamoe | implemented_unvalidated | literal released Transformer expert slicing; tissue-prior CE, author batch Cox or classification CE, event sampler, AdamW/cyclic cosine |
| pathomics_concat_source | pathomics | implemented_unvalidated | author concat branch, two-stage MSE alignment and censored NLL, two Adam optimizers, explicit source freeze and checkpoint/tail repairs |
| pathomics_image_source | pathomics | implemented_unvalidated | author image branch, two-stage MSE alignment and censored NLL, two Adam optimizers, explicit source freeze and checkpoint/tail repairs |
| pathomics_omics_source | pathomics | implemented_unvalidated | author omics branch, two-stage MSE alignment and censored NLL, two Adam optimizers, explicit source freeze and checkpoint/tail repairs |
| mmp_panther_source | mmp | implemented_unvalidated | author PANTHER prototypes with full-attention fusion, train-only KMeans, pathway SNN, batch Cox/NLL, AdamW warmup cosine and explicit source repairs |
| mmp_panther_uot_source | mmp | implemented_unvalidated | author PANTHER prototypes with differentiable UOT fusion, train-only KMeans, pathway SNN, batch Cox/NLL, AdamW warmup cosine and explicit source repairs |
| mmp_ot_source | mmp | implemented_unvalidated | author OT prototypes with full-attention fusion, train-only KMeans, pathway SNN, batch Cox/NLL, AdamW warmup cosine and explicit source repairs |
| mmp_ot_uot_source | mmp | implemented_unvalidated | author OT prototypes with differentiable UOT fusion, train-only KMeans, pathway SNN, batch Cox/NLL, AdamW warmup cosine and explicit source repairs |
| pibd_source | pibd | implemented_unvalidated | pinned author prototype bottlenecks, PoE, disentangled transformer, joint CLUB fitting, weighted patient minibatches and complete survival/proxy/KL/MI objective |
| advmil_esat | advmil | implemented_unvalidated | pinned author continuous ESAT + RLIP discriminator; two Adam optimizers, censored reconstruction, noise inference and explicit label masking; corrected discriminator BCE |
| advmil_abmil | advmil | implemented_unvalidated | pinned author continuous ABMIL + RLIP discriminator; two Adam optimizers, censored reconstruction, noise inference and explicit label masking; corrected discriminator BCE |
| advmil_esat_source | advmil | implemented_unvalidated | pinned author continuous ESAT + RLIP discriminator; two Adam optimizers, censored reconstruction, noise inference and explicit label masking; literal released discriminator BCE |
| advmil_abmil_source | advmil | implemented_unvalidated | pinned author continuous ABMIL + RLIP discriminator; two Adam optimizers, censored reconstruction, noise inference and explicit label masking; literal released discriminator BCE |
| hmil_source_repaired | hmil | implemented_unvalidated | author hierarchical coarse/fine attention, SupCon, source curriculum and bag alignment; explicit scalar-head repair from prior author commit |
| tpmil | tpmil | implemented_unvalidated | author trainable K+1 prototypes, detached attention pseudo-targets, .5 CE + .5 KL and balanced sampling; explicit constant-attention repair |
| cdpmil_source | cdpmil | implemented_unvalidated | public source path: per-slide Bayesian DP centroids and sampled Gaussian class log densities, CE and Adam/cosine; no variational gamma updates in released trainer |
| rankmix_dsmil | rankmix | implemented_unvalidated | author binary DSMIL, supervised initialization, natural/balanced sampling and dual BCE; repaired ranked patch IDs |
| rankmix_dsmil_source | rankmix | implemented_unvalidated | binary DSMIL RankMix preserving literal author argsort-position selection; explicit source-behavior comparator |
| rankmix_frmil | rankmix | implemented_unvalidated | author binary FRMIL, paired supervised initialization, mixed-bag BCE plus original-pair magnitude; repaired IDs and padded selection |
| rankmix_frmil_source | rankmix | implemented_unvalidated | binary FRMIL RankMix preserving literal author argsort-position/padding behavior; explicit source comparator |
| cate | cate | implemented_unvalidated | author binary CATE-MIL; concept-guided IB, top-10 InfoNCE, KL and detached CFI; explicit weighted objective |
| vilamil | vilamil | implemented_unvalidated | author dual-scale prototype/context decoders and learned CLIP prompts; corrected EOT position, frozen text encoder per paper, CE |
| vilamil_source | vilamil | implemented_unvalidated | literal public ViLa prompt EOT and trainable text encoder; dual-scale decoders and CE; separately named source behavior |
| he2rna | he2rna | implemented_unvalidated | author tile Conv1D network; spatial supertiles, shared random top-k, train-mean bias, MSE regression and project/gene Pearson selection |
| healnet | healnet | implemented_unvalidated | author shared latent multimodal fusion, weighted censored NLL, L1 and per-batch Adam/OneCycle; patient inputs and train-only preprocessing |
| camil_channel | camil_channel | implemented_unvalidated | author four Nystrom/MCAB blocks and gated pooling; CE, author RAdam/Lookahead and two-bag gradient accumulation |
| camil_context | camil_context | implemented_unvalidated | TensorFlow equation port: spatial-cosine neighbor graph, literal Nystrom axes, squared-gate global/local fusion; Keras Adam and validation-loss plateau |
| dyhg | dyhg | implemented_unvalidated | author Gumbel soft hypergraph incidence, node-edge-node messages and gated pooling; CE, Adam and isolated stochastic inference |
| mfcmil_abmil | mfcmil | implemented_unvalidated | author ABMIL, multiresolution filters, frequency projections and high/low memory interventions; CE and step warmup/cosine |
| mfcmil_transmil | mfcmil | implemented_unvalidated | author Nystrom TransMIL with frequency, multiresolution and memory intervention; CE and step warmup/cosine |
| sdfp_abmil | sdfp | implemented_unvalidated | author S-DFP with raw-feature attention; shared prompt then slide-aware joint training, CE and step cosine |
| sdfp_gated | sdfp | implemented_unvalidated | author S-DFP with raw-feature gated attention; shared prompt then slide-aware joint training, CE and step cosine |
| sdfp_dsmil | sdfp | implemented_unvalidated | author S-DFP/DSMIL; two-stage prompts plus all-instance CE and bag CE, joint Adam and step cosine |
| sdfp_transmil | sdfp | implemented_unvalidated | author S-DFP/Nystrom TransMIL; two-stage joint prompting, CE and step cosine |
| mil_dropout_abmil | mil_dropout | implemented_unvalidated | author non-gated ABMIL with three MIL-Dropout layers; bag CE, AdamW and checkpointed Lookahead |
| mil_dropout_gated | mil_dropout | implemented_unvalidated | author gated ABMIL with three MIL-Dropout layers; bag CE, AdamW and checkpointed Lookahead |
| mil_dropout_dsmil | mil_dropout | implemented_unvalidated | author MIL-Dropout DSMIL bag output; CE, AdamW and epoch cosine; critical selection remains discrete |
| mil_dropout_transmil | mil_dropout | implemented_unvalidated | author MIL-Dropout TransMIL/Nystrom/PPEG; CE, RAdam and checkpointed Lookahead |
| packmil_abmil | packmil | implemented_unvalidated | author sequence packs, residual hyperslides and ADS; CE plus focal auxiliary objective, Adam/cosine |
| packmil_survival | packmil | implemented_unvalidated | author ABMIL packs/ADS and hyperslide censor targets; train-only patient event bins, main and mixed censored NLL |
| scmil | scmil | implemented_unvalidated | author SoftFilter and sparse context attention; continuous mixture censor likelihood; explicit CPU clustering and patient slide-mixture adaptation |
| deepattnmisl | deepattnmisl | implemented_unvalidated | author phenotype Conv/pooling/attention; batch patient-mean Cox plus L1; explicit patient KMeans or provided labels |
| mico | mico | implemented_unvalidated | author cluster routes/reducers; train-only FAISS initialization; CE; author Adam decay grouping |
| mico_survival | mico | implemented_unvalidated | author per-slide context and patient joint pooling; train-only FAISS centers and event bins; discrete NLL |
| efficientmil_gru | efficientmil | implemented_unvalidated | author APS plus bidirectional GRU; dual bag/max-instance BCE, L2, Adam and cosine schedule |
| efficientmil_lstm | efficientmil | implemented_unvalidated | author APS plus bidirectional LSTM; dual bag/max-instance BCE, L2, Adam and cosine schedule |
| bayesmil_vis | bayesmil | implemented_unvalidated | author variational weights and logistic-normal attention; CE + weight regularizer; MC mean probability prediction |
| bayesmil_sdpr | bayesmil | implemented_unvalidated | Bayes-MIL with literal author class-dependent patch regularizer, checkpointed plateau schedule and MC inference |
| bayesmil_apcrf | bayesmil | implemented_unvalidated | full spatial APCRF plus SDPR, real canvas and corrected raster stride; author regularizer and MC probability average |
| caprmil_mean | caprmil | implemented_unvalidated | author soft context tokens, token attention and de-slicing; mean pooling, CE and AdamW/cosine |
| caprmil_attention | caprmil | implemented_unvalidated | author CAPRMIL context blocks with learned attention pooling; CE and AdamW/cosine |
| caprmil_gated | caprmil | implemented_unvalidated | author CAPRMIL context blocks with gated attention pooling; CE and AdamW/cosine |
| varmil | varmil | implemented_unvalidated | author raw-feature attention-weighted mean/variance concatenation and CE; frozen-feature VarMIL, not SimCLR pretraining |
| smmil_early | smmil | implemented_unvalidated | author early feature smoothing on spatial adjacency, BCE and epoch warmup |
| smmil_mid | smmil | implemented_unvalidated | author mid attention-projection smoothing on spatial adjacency, BCE and epoch warmup |
| smmil_late | smmil | implemented_unvalidated | author late attention-logit smoothing on spatial adjacency, BCE and epoch warmup |
| smmil_transformer | smmil | implemented_unvalidated | author SmTransformer plus early Sm attention pool, BCE and epoch warmup |
| psamil | psamil | implemented_unvalidated | 2026 author spatial decay attention, active head-diversity objective, class balancing and separate decay learning rate |
| slidegraph | slidegraph | implemented_unvalidated | author hierarchical feature-spatial clustering, EdgeConv, binary pair ranking and train-only Platt scaling; single validation checkpoint |
| csmil | csmil | implemented_unvalidated | author raw cross-scale scoring, masked bag attention, per-slide fine-scale KMeans and cluster sampling; binary NLL with summed accumulation |
| hagmil_ce | hagmil | implemented_unvalidated | author three-stage four-layer IAM models, detached hierarchical selection, CE instance clustering and per-level optimizer updates; explicit CE profile |
| cmta | cmta | implemented_unvalidated | pinned author cross-modal encoders, translators and decoders; discrete NLL + detached-target path/omics L1 alignment |
| motcat | motcat | implemented_unvalidated | pinned author OT co-attention, dual Transformers and fusion; full-bag discrete NLL; explicit entropy UOT regularizer |
| motcat_micro | motcat | implemented_unvalidated | author shuffled micro-bag partition, mean per-micro NLL and mean risk; streamed backward; fixed-seed isolated inference |
| cbmil | cbmil | implemented_unvalidated | author adaptive cluster selection, patch/phenotype fusion, CLAM instance loss, two-view NT-Xent and asymmetric feature mixing; explicit FM-to-512 adapter |
| sgpmil | sgpmil | implemented_unvalidated | author sparse variational GP attention; Monte Carlo mean logits CE + inducing KL / training bags; mean probability prediction; AdamW |
| mmil | mmil | implemented_unvalidated | author multi-level group attention and message-token communication; random group masking; CE; fixed-seed unmasked inference |
| mmil_coords | mmil | implemented_unvalidated | author physical-coordinate KMeans grouping, group masking and message-token communication; CE |
| ptcmil | ptcmil | implemented_unvalidated | author prompt clustering, local Nystrom attention and learned token merging; CE + PTC orthogonality; checkpointed within-epoch moving prompts |
| survpath | survpath | implemented_unvalidated | pinned author pathway/histology multimodal attention; train-only molecular normalization and event-time bins; censored discrete NLL |
| mcat | mcat | implemented_unvalidated | pinned author six-signature co-attention, dual Transformers and fusion; train-only normalization/bins; discrete NLL |
| porpoise | porpoise | implemented_unvalidated | pinned author gated WSI attention + omics SNN + gated bilinear fusion; train-only normalization/bins; discrete NLL |
| patchgcn | patchgcn | implemented_unvalidated | pinned author residual GENConv and concatenated depths; exact physical kNN within slides; patient disjoint union; discrete NLL |
| psemix_abmil | psemix | implemented_unvalidated | ProtoDiv phenotype refinement, fixed checkpointed partitions, pseudo-bag mixing and content-weighted soft BCE; PULS ABMIL base |
| prodiv_abmil | prodiv | implemented_unvalidated | ProtoDiv pseudo-bag sampling via PseMix authors implementation; PULS ABMIL base and BCE |
| remix_abmil | remix | implemented_unvalidated | train-bag spherical KMeans and covariance shifts; same-class joint replacement/append/interpolation/shift; author ReLU ABMIL + BCE |
| frmil | frmil | implemented_unvalidated | pinned MICCAI model; binary class-paired episodes, train-maximum padding, bag CE + max-instance BCE + feature magnitude |
| focusmil | focusmil | implemented_unvalidated | binary variational max pooling; reparameterized latent and active KL; deterministic posterior-mean inference |
| dgrmil | dgrmil | implemented_unvalidated | pinned author model; class-conditional EMA centers, warmup, one-hot BCE, log-determinant diversity and triplet alignment |
| rrt | rrt | implemented_unvalidated | pinned author RRT-MIL: regional MSA, cross-region MSA and EPEG; CE; current refactored source |
| chowder | chowder | implemented_unvalidated | Owkin later implementation adaptation: classwise extreme scores and shared sigmoid MLP |
| varpool | varpool | implemented_unvalidated | projected attention variance pooling with mean branch and CE classification head |
| varpool_cox | varpool | implemented_unvalidated | projected attention variance pooling with full-cohort Cox objective |
| h2t | h2t | implemented_unvalidated | train-only prototype mining, normalized per-cluster means, linear classifier (PANTHER authors H2T comparator) |
| panther | panther | implemented_unvalidated | train-only prototype mining, diagonal NIW MAP-EM allcat representation and linear classifier |
| ibmil | ibmil | implemented_unvalidated | supervised warmup, training-bag spherical clustering, interventional attention, CE; sklearn kmeans backend |
| additive_mil | additive_mil | implemented_unvalidated | attention-weighted instance prediction then exact additive class contributions |
| linmil | linmil | implemented_unvalidated | ReLU-kernel linear attention, CLS pooling, CE |
| fouriermil | fouriermil | implemented_unvalidated | adaptive Fourier filtering blocks and CLS pooling, CE |
| retmil | retmil | implemented_unvalidated | local chunk retention, global retention and gated pooling, CE |
| pneamil | pneamil | implemented_unvalidated | class-specific nonnegative evidence, max aggregation and active L1 sparsity |
| mrblock | mrblock | implemented_unvalidated | gated attention with frozen affine maps and nonlinear low-rank residuals |
| carmil | carmil | implemented_unvalidated | factual CE + counterfactual effect CE + attention regularization |
| fcmil | fcmil | implemented_unvalidated | spatial/frequency attention + drop/replace causal regularization |
| asmil | asmil | implemented_unvalidated | bag CE + normalized-sigmoid EMA anchor consistency; repaired EMA updates |
| attrimil | attrimil | implemented_unvalidated | frozen-feature variant: attribution pooling + physical-neighbour spatial constraint + checkpointed cross-slide ranking queues |
| nnmil | nnmil | implemented_unvalidated | classification recipe: train-only bag-size planning, balanced batches, feature subspaces, masked padding and probability ensemble |
| madmil | madmil | implemented_unvalidated | split-feature independent gated multihead attention with concatenated pooled heads |
| litemil | litemil | implemented_unvalidated | author cross-attention queries; validation-only model selection; no test-weighted fold ensemble |
| mhim_abmil | mhim | implemented_unvalidated | v1 attention equations; explicit supervised warm-start, teacher EMA, high/low/random masking and feature distillation |
| ilra | ilra | implemented_unvalidated | low-rank compression/expansion, nonlocal pooling; CE; one pooling seed |
| wikg | wikg | implemented_unvalidated | chunked feature-derived top-k graph, knowledge attention and bi-interaction |
| acmil_ga | acmil | implemented_unvalidated | official GA equations: multi-branch supervision, stochastic top-k masking and diversity |
| aem_abmil | aem | implemented_unvalidated | ABMIL + full-bag negative attention entropy regularization |
| mean_linear | mean_mil | implemented_unvalidated | raw-feature mean + train-only standardization + logistic regression |
| mean_mil | mean_mil | implemented_unvalidated | nonlinear instance logits, mean pooling |
| max_mil | max_mil | implemented_unvalidated | nonlinear instance logits, classwise max pooling |
| abmil | abmil | implemented_unvalidated | WSI feature adaptation, non-gated attention, CE |
| gated_abmil | abmil | implemented_unvalidated | WSI feature adaptation, gated attention, CE |
| clam_sb | clam | implemented_unvalidated | vendored official model, CE instance clustering + bag loss |
| clam_mb | clam | implemented_unvalidated | vendored official multibranch model, instance clustering + bag loss |
| dsmil | dsmil | implemented_unvalidated | dual-stream BCE; sigmoid stream ensemble normalized for exclusive classes |
| transmil | transmil | implemented_unvalidated | MIL_BASELINE Nystrom + PPEG adaptation; CE |
| dtfd | dtfd | implemented_unvalidated | two optimizers, group CE, AFS/MaxS/MaxMinS distillation; deterministic evaluation grouping |
| abmil_cox | abmil | implemented_unvalidated | PULS adaptation: full-cohort Cox Breslow, patient bags |
| gated_abmil_cox | abmil | implemented_unvalidated | PULS adaptation: full-cohort Cox Breslow, gated attention |
| pamil_policy_paper_reference | pamil_policy | implemented_unvalidated | Complete explicit paper-reference native512 SFFR/fused history/TCM, LIIS topM or GMSS/GHSS, WSL/STL/stopgradSIA/actualTRAINreward/PPO/AdaMax300/fullstate |
| pamil_policy_released_sampling_reference_training | pamil_policy | implemented_unvalidated | Actual released LIIS weighted replacement/padding/floorT/retainfirst/rawhistory control, with complete named reconstructed paper training/objectives/reward/PPO |
| focus_source | focus | implemented_unvalidated | Original FOCUS native full trainable CONCH text/prompt, complete visual selection/compression/8head attention, one-bag CE, VAL-error firstbest |
| dispro_source_repaired | dispro | implemented_unvalidated | Full original native UniPro-P30 + UniPro-G50 + MultiPro50; explicit actual-Parameter/CLS/cosine/mask/presence repairs, outcome-free VAL-selected teachers and full-state continuation |
| lrmil_source_repaired | lrmil | implemented_unvalidated | Complete native CONCH768/12-layer patch KD, actual VAL-selected HRteacher and LRstudent source CE/bagKL/softKL/hardCE; independent source phase resets, selected-encoder features and all3Adam states. Explicit layout/teacher-tuple ports and declared200/200/200 reference caps. |
| kmgmamba_source | kmgmamba | implemented_unvalidated | Full original96width/9neighbor/top-degree prototype graph, cross-scale4state selective scan,128attention,three weighted CE/SVM losses,Adam100max/VALAUC latest ties; original binary heads |
| kmgmamba_multiclass_repaired | kmgmamba | implemented_unvalidated | Full original96width/9neighbor/top-degree prototype graph, cross-scale4state selective scan,128attention,three weighted CE/SVM losses,Adam100max/VALAUC latest ties; explicit C-class attention/prototype head repair |
| sammil_attn_source_control | sammil | implemented_unvalidated | Complete released512/128 attention SAM-MIL, all five pseudo-bag CE and area consistency objectives, fixed deepcopy teacher, Adam/cosine200 source recipe; explicit source loss-overwrite/teacher-gradient/mask-index control |
| sammil_attn_objective_repaired | sammil | implemented_unvalidated | Complete released512/128 attention SAM-MIL, all five pseudo-bag CE and area consistency objectives, fixed deepcopy teacher, Adam/cosine200 source recipe; explicit global CE/student-consistency/compact region-index repairs |
| c2c_cluster_source | c2c | implemented_unvalidated | Full end-to-end original pretrained ResNet18/256/64/32attention; each-epoch actual FAISS8group and up-to8patch/group sampling; CE+.01patchCE+.01uniform-KL, all-patch128streaming/source25epoch Adam and latest VAL-accuracy tie; original Notebook BN with exact Torch1.5 semantics |
| c2c_cluster_default_bn_source | c2c | implemented_unvalidated | Full end-to-end original pretrained ResNet18/256/64/32attention; each-epoch actual FAISS8group and up-to8patch/group sampling; CE+.01patchCE+.01uniform-KL, all-patch128streaming/source25epoch Adam and latest VAL-accuracy tie; explicit original constructor-default BN condition |
| c2c_cluster_multiclass_repaired | c2c | implemented_unvalidated | Full end-to-end original pretrained ResNet18/256/64/32attention; each-epoch actual FAISS8group and up-to8patch/group sampling; CE+.01patchCE+.01uniform-KL, all-patch128streaming/source25epoch Adam and latest VAL-accuracy tie; original Notebook BN with exact Torch1.5 semantics; explicit C-way weak patch label repair |
| sparseconvmil_raw_source_c32 | sparseconvmil | implemented_unvalidated | Full original end-to-end ResNet18, two3x3 c32 original compiled submanifold sparse convolutions, active-only mean/32classifier,200epoch Adam/with-replacement tile sampling; source spatial transforms and final weights; explicit seeded evaluation extension |
| sparseconvmil_raw_source_c128 | sparseconvmil | implemented_unvalidated | Full original end-to-end ResNet18, two3x3 c128 original compiled submanifold sparse convolutions, active-only mean/32classifier,200epoch Adam/with-replacement tile sampling; source spatial transforms and final weights; explicit seeded evaluation extension |
| conmil_pos_source | conmil | implemented_unvalidated | Complete native512/256/128 gated semantic attention,learned alpha and original cubic B-spline KAN; actual frozen CONCH positive/negative text; TRAIN-only4096cap,source50epoch Adam/cosine and strict VAL accuracy; original binary target and selection branch |
| conmil_posneg_source | conmil | implemented_unvalidated | Complete native512/256/128 gated semantic attention,learned alpha and original cubic B-spline KAN; actual frozen CONCH positive/negative text; TRAIN-only4096cap,source50epoch Adam/cosine and strict VAL accuracy; original binary target and selection branch |
| conmil_posneg_multiclass_repaired | conmil | implemented_unvalidated | Complete native512/256/128 gated semantic attention,learned alpha and original cubic B-spline KAN; actual frozen CONCH positive/negative text; TRAIN-only4096cap,source50epoch Adam/cosine and strict VAL accuracy; explicit C-way labels/metrics/first-checkpoint repair |
| miladapter_zsmil_bgap_zs | miladapter | implemented_unvalidated | Actual complete author ZSMIL/BGAP/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_bgap_random | miladapter | implemented_unvalidated | Actual complete author ZSMIL/BGAP/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_bgmp_zs | miladapter | implemented_unvalidated | Actual complete author ZSMIL/BGMP/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_bgmp_random | miladapter | implemented_unvalidated | Actual complete author ZSMIL/BGMP/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_abmil_zs | miladapter | implemented_unvalidated | Actual complete author ZSMIL/ABMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_abmil_random | miladapter | implemented_unvalidated | Actual complete author ZSMIL/ABMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_transmil_zs | miladapter | implemented_unvalidated | Actual complete author ZSMIL/TransMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_transmil_random | miladapter | implemented_unvalidated | Actual complete author ZSMIL/TransMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_wikgmil_zs | miladapter | implemented_unvalidated | Actual complete author ZSMIL/WIKGMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_wikgmil_random | miladapter | implemented_unvalidated | Actual complete author ZSMIL/WIKGMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_ilramil_zs | miladapter | implemented_unvalidated | Actual complete author ZSMIL/ILRAMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_ilramil_random | miladapter | implemented_unvalidated | Actual complete author ZSMIL/ILRAMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_rrtmil_zs | miladapter | implemented_unvalidated | Actual complete author ZSMIL/RRTMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_zsmil_rrtmil_random | miladapter | implemented_unvalidated | Actual complete author ZSMIL/RRTMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_taskres_bgap | miladapter | implemented_unvalidated | Actual complete author TaskRes/BGAP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_taskres_bgmp | miladapter | implemented_unvalidated | Actual complete author TaskRes/BGMP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_taskres_abmil | miladapter | implemented_unvalidated | Actual complete author TaskRes/ABMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_taskres_transmil | miladapter | implemented_unvalidated | Actual complete author TaskRes/TransMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_clipadapter_bgap | miladapter | implemented_unvalidated | Actual complete author CLIPAdapter/BGAP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_clipadapter_bgmp | miladapter | implemented_unvalidated | Actual complete author CLIPAdapter/BGMP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_clipadapter_abmil | miladapter | implemented_unvalidated | Actual complete author CLIPAdapter/ABMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_clipadapter_transmil | miladapter | implemented_unvalidated | Actual complete author CLIPAdapter/TransMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_tipadapter_bgap | miladapter | implemented_unvalidated | Actual complete author TIPAdapter/BGAP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_tipadapter_bgmp | miladapter | implemented_unvalidated | Actual complete author TIPAdapter/BGMP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_tipadapter_abmil | miladapter | implemented_unvalidated | Actual complete author TIPAdapter/ABMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| miladapter_tipadapter_transmil | miladapter | implemented_unvalidated | Actual complete author TIPAdapter/TransMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool. |
| center_embeddings_bag_classifier_repaired | center_embeddings | implemented_unvalidated | Full source original ResNet34/FPN/32x256RGB bagCNN/shared512embedding, actual original ImageNet weights and torchvision0.7 preprocessing; exhaustive per-epoch source TRAIN and VAL candidate scores/top32/ordered bags, complete instance CE + bag CE + center loss, separate two-group Adam and center Adam, literal source staircase and20epoch last-tie VAL balanced-ACC recipe. Explicit undefined bag_feat -> actual normalized bag_embed repair; bag_classifier inference. |
| center_embeddings_soft_assignment_repaired | center_embeddings | implemented_unvalidated | Full source original ResNet34/FPN/32x256RGB bagCNN/shared512embedding, actual original ImageNet weights and torchvision0.7 preprocessing; exhaustive per-epoch source TRAIN and VAL candidate scores/top32/ordered bags, complete instance CE + bag CE + center loss, separate two-group Adam and center Adam, literal source staircase and20epoch last-tie VAL balanced-ACC recipe. Explicit undefined bag_feat -> actual normalized bag_embed repair; soft_assignment inference. |

## 待实现条目

| ID | 名称 |
|---|---|
| iibmil | IIB-MIL |
| gmmamba | GMMamba |
| longmil | LongMIL |
| rmdl | RMDL |
| attention2majority | Attention2majority |
| pmil_prototypical | PMIL Prototypical |
| promptmil | Prompt-MIL |
| icmil_coupled | ICMIL Iteratively Coupled |
| macnet | MacNet |
| convl | ConVL |
| dyko | DyKo |
| libramil | Libra-MIL |
| muse | MUSE |
| taxomil | TaxoMIL |
| hybrid_supervision | Hybrid Supervision Learning |
| dspamil | DSPA-MIL / DSPA-U-MIL |
| lpgmil | LPGMIL |
| m4 | M4 |
| protomil | ProtoMIL |
| akdpmp | AKD-PMP |
| comel | CoMEL |
| qpmilvl | QPMIL-VL |
| kmpmil | KMP-MIL |
| feather | FEATHER / MIL-Lab models |
| icmil_context | ICMIL In-context |
| mggmmil | MGGMMIL autofluorescence |
| lnplmil | LNPL-MIL |
| nciemil | NcIEMIL |
| mexd | MExD |
