# Third-party source notices

PMIL Shapley uses externally fetched RenaoYan/PMIL at a5df40276fa5848056a89c0cde221500fad4099d (MIT, Copyright2024 Renao Yan). Author networks and complete main/training/ranking definitions are checked against fixed Git blobs; upstream source is not redistributed. PULS provides the task bridge, complete round/plateau/checkpoint control and separately named gradient, branch-ranking and constant-range repairs. Native input dimensions, complete objectives, selected-round networks, partitions and RNG are verified independently; CPU synthetic evidence is not natural WSI or paper-number reproduction. See docs/PMIL_SHAPLEY.md.

MAMMOTH uses externally fetched mahmoodlab/MAMMOTH at aeeaffcd12500ed8dddec67665a33eb129c9ed84. Its author-declared CC-BY-NC-ND-4.0 terms remain separate; no MAMMOTH or MIL-Lab source is redistributed. The core tracked files are executed unmodified, and actual MIL model definitions are loaded from verified Git objects. PULS supplies the data bridge and explicit trainer; paper/source differences and the named ILRA routing-width repair are described in docs/MAMMOTH.md. The optional broken nonshared expert branch is rejected. Binary SmoothTop1 comparisons use external oval-group/smooth-topk at 12c1645f187e2fa0c05f47bf1fe48864d4bd2707 (MIT), also not redistributed. Software agreement is not paper-score reproduction.

HEALNet (konst-int-i/healnet, Apache-2.0) and HE2RNA (owkin/HE2RNA_code, GPL-3.0-or-later) are fetched separately at the commits in sources.lock.json. Named architecture and objective definitions are loaded from verified Git blobs; fetched source is not redistributed. HEALNet exception propagation, missing-modality slot/loop repairs, weighted NLL and OneCycle recipe, and HE2RNA signed-feature validity, explicit expression targets, random top-k and sklearn spatial clustering backend are documented in docs/MOLECULAR_REGRESSION.md. CPU synthetic checks do not establish clinical or original-paper replication.

Vendored CLAM model source: mahmoodlab/CLAM, commit 53e2409d4a8189c682c173382964a85f114f923c, GPL-3.0; license preserved next to source.

Vendored TransMIL adaptation and Nystrom attention: lingxitong/MIL_BASELINE, commit e1c5d9fcd771f30f94437e993ecf24c0535df991, MIT; license preserved next to source. This adapter is not a claim of original TransMIL numerical reproduction.

DTFD algorithm adaptation is informed by hrzhang1123/DTFD-MIL, commit 10964f4dcc27c65ce110a0e9a3b9240bff58da8a. Its MIT notice is preserved in docs/DTFD_LICENSE.

The eleven modules in `src/milplus/vendor/baseline_expansion` are copied from the same fixed MIT-licensed MIL_BASELINE commit. `MANIFEST.json` records upstream paths, original file hashes and the local DropPath import change. They are third-party adaptations, not a claim that MIL_BASELINE reproduces each original paper. Available additional author licenses for Lin-MIL, AttriMIL, PNEA-MIL, FC-MIL and MR-Block are preserved as `LICENSE.author-*` alongside these files.

The new FC-MIL original-mechanism profiles load `7FFDW/FCMIL` at
`315ac88808d7517429ca0b55073e75dce03ddfd9` privately. Its actual fixed-tree
LICENSE is Apache-2.0. The original architecture/loss/manager bodies and explicitly
chosen benchmark settings are documented in `docs/FCMIL_ORIGINAL_MECHANISM.md`;
the older `fcmil` port retains its separate provenance.

The separately named `pneamil_author_*` profiles load the original PNEA-MIL model and training definitions from Chen-Jxiang/PNEA-MIL, commit `ef204ff5179ce7ea4bb7d04744f1553bee2f039c`, using verified external Git blobs. That author's root license contains the GNU GPL version 3 text. These profiles are distinct from the existing `pneamil` MIL_BASELINE port. Native training defaults, the literal TRAIN selector, the separately named VAL repair and immutable selected-checkpoint repair are documented in `docs/PNEAMIL_AUTHOR_RECIPE.md`. No blanket license or reproduction clearance is inferred from a source being available.

LiteMIL's model is copied from hkussaibi/LiteMIL at the commit in `sources.lock.json`, with its Apache-2.0 license and source notice preserved. The author's fold/test-based model-selection code is not included.

MAD-MIL's split-head algorithm is adapted from tueimage/MAD-MIL, with its GPL-3.0 notice in `docs/LICENSE.madmil`. FocusMIL's binary variational route is adapted from Raymvp/FocusMIL-and-other-max-pooling-methods, with its MIT notice in `docs/LICENSE.focusmil`.

CHOWDER, VarPool, H2T, PANTHER, IBMIL, PseMix/ProtoDiv and ReMix have source and algorithm references in `sources.lock.json` and `docs/EXPANSION.md`. The implementation includes explicitly documented new data, checkpointing and numerical handling. Upstream licensing varies (including noncommercial terms in PANTHER); this repository's root license does not relicense external repositories or model weights.

FRMIL, RRT-MIL and DGR-MIL author code is fetched separately at locked commits and loaded in isolated namespaces. It is not bundled in the wheel or source archive. Compatibility changes to imports, DropPath and RRT input width are visible in `src/milplus/upstream.py`. Each fetched repository retains its own license. No externally fetched datasets, author checkpoints or FM weights are distributed.

SurvPath, MCAT, PORPOISE and Patch-GCN are also fetched separately from the Mahmood Lab repositories at commits in `sources.lock.json`. Their author model files are not redistributed. The wrappers preserve the actual multimodal or graph networks and document device, input-width and output-contract changes in `docs/SURVIVAL.md` and `upstream.load_survival`. These repositories have their own GPL/noncommercial academic usage statements; consult each original repository before use. The PULS task alignment, train-only normalization/time discretization and discrete NLL trainer are controlled-protocol adaptations, not numerical reproduction claims or permission to redistribute external code and weights.

SGPMIL (mandlos/SGPMIL), MMIL-Transformer (hustvl/MMIL-Transformer) and PTCMIL (ubc-tea/PTCMIL) are separately fetched at pinned commits. PULS extracts named definitions from verified Git blobs and documents compatibility edits in `docs/GP_PROMPT_MULTILEVEL.md`. SGPMIL has a modified MIT license restricting commercial use; MMIL has an MIT license. No license file was found at the root of the pinned PTCMIL checkout, so its license review remains pending. External source and weights are not redistributed or relicensed by this package. The classification adapters preserve their core GP, message-token and prompt-clustering mechanisms; training-protocol differences are disclosed separately from software validation.

CBMIL (wwu98934/CBMIL, commit `980252ac796e4f68ad97d7b0b0de4aef4bb00157`) is separately fetched. Its CLAM, cluster-selection, phenotype pooling and contrastive definitions are loaded from fixed Git blobs without redistribution. The pinned source has an unresolved local `models.mil` import; PULS resolves it to the author's existing `models/clam.py`. The FM input projection, explicit KMeans/provided-cluster data path and singleton-tail batching difference are documented in `docs/CBMIL.md`. No root license file was found at the pinned source; license review remains pending.

CMTA (FT-ZHOU-ZZZ/CMTA) and MOTCat (Innse/MOTCat) are fetched separately at locked commits. Their README files specify non-commercial academic purposes. The author networks are loaded from verified Git blobs, without redistribution. Device/input compatibility, the explicit legacy entropy regularizer for newer POT, singleton/zero-cost handling and streamed micro-batch gradients are documented in `docs/CROSSMODAL.md`. The architecture and core objectives are retained within a controlled PULS training protocol; this is not a claim of the original experiments' numerical reproduction.

CS-MIL (hrlblab/CS-MIL) and HAG-MIL (BearCleverProud/HAG-MIL) are separately fetched at commits in sources.lock.json and are not redistributed. CS-MIL has a root MIT license; no root license was found for the fixed HAG-MIL source, so review remains pending. Architecture, device/input compatibility patches, stable loss/mask behavior, the explicit HAG CE instance-loss profile and backend-version differences are documented in docs/MULTISCALE.md. Admin-Torch 0.1.0 is an external dependency: its bundled LICENSE says MIT, while package metadata declares Apache License 2.0. This discrepancy is retained for review; no dependency code is vendored. H²-MIL is pinned for audit only and remains unadapted.

SmMIL (Franblueee/SmMIL), PSA-MIL (SharonPeled/PSA-MIL) and SlideGraph+ (wenqi006/SlideGraph) are separately fetched at pinned commits. The first two carry Apache-2.0 licenses. SlideGraph root LICENSE.md is GPL-3.0, while its Python headers state GPL-2.0-or-later; both statements are retained for review. Author code is loaded from verified Git objects, not redistributed. docs/SPATIAL.md records exact-solver, eval-dropout and numerical compatibility corrections; training-only standardization/calibration; degenerate graph extensions and the single-checkpoint protocol.

Bayes-MIL (ralphc1212/bayes-mil), CAPRMIL (mandlos/CAPRMIL) and VarMIL (NKI-AI/dlup-lightning-mil) are fetched separately at the commits in sources.lock.json. Bayes-MIL has a GPL-3.0 root license and an additional noncommercial academic README statement; CAPRMIL has a modified MIT license with a noncommercial restriction; VarMIL has an Apache-2.0 license. No fetched author code is redistributed. docs/BAYES_CONTEXT_VARIANCE.md records spatial stride, zero-variance derivative and Monte Carlo inference corrections, the unchanged literal patch regularizer, the CAPR entrypoint repair and VarMIL logging compatibility. VarMIL admission covers its frozen-feature classifier, not the DeepSMILE SimCLR pretraining pipeline.

MiCo (junjianli106/MiCo) and EfficientMIL (chengyingshe/EfficientMIL) are separately fetched at pinned commits, without redistribution. No root license was found in either fixed checkout; license review remains pending. docs/CONTEXT_SEQUENCE.md records MiCo training-only CPU FAISS initialization and optimizer choices, patient-level context handling, and the EfficientMIL GRU/LSTM recipe. The unverified custom Mamba backend is not silently replaced. AGDR-MIL (shuiguoc/AGDR-MIL) and PAMIL (Jiashuai-Liu/PAMIL) are fetched separately at pinned commits. See their dedicated entries for the current integration scope; the root-license status remains unresolved and no author code is bundled.

SCMIL (yang-ze-kang/SCMIL) and DeepAttnMISL (uta-smile/DeepAttnMISL) are fetched separately at locked commits. SCMIL carries an MIT license; no root license was found for the pinned DeepAttnMISL source, so review remains pending. No author source is redistributed. docs/CONTINUOUS_SURVIVAL.md discloses the CPU clustering backend, patient mixture and evaluation protocol, stable inverse softplus, masked exponent and Cox logsumexp repairs, generalized Conv input width and explicit phenotype preprocessing. AdvMIL (liupei101/AdvMIL, MIT) is pinned for audit only; its discriminator formula and region ordering are recorded without claiming a runnable adaptation.

MIL-Dropout (ChongQingNoSubway/MILDropout, MIT) and PackMIL (FangHeng/PackMIL, Apache-2.0) are fetched separately at pinned commits without redistribution. docs/DROPOUT_PACKING.md records device/dtype compatibility and explicit all-masked rejection for MIL-Dropout; its four author networks, CE and optimizer settings; and PackMIL author ABMIL packing, ADS, focal/hyperslide censor objectives. Unselected PackMIL backbones and PANDA-specific training are not claimed.

- S-DFP：`thomascong121/S-DFP@22ef367ee3c99ca2dbd1ed70d13130c1c354623e`；未发现根目录许可证，许可待核。通过独立源码加载提示与四个骨干，修正无效 RGB 调用，保留两阶段联合训练；不重新分发作者代码。
- MFC-MIL：`WissingChen/MFC-MIL@145494ec0f221c8b1cc32843efeafb28426634c0`；未发现根目录许可证，许可待核。独立加载完整两骨干与频域/多分辨率/记忆模块；注册可训练记忆、通用宽度及类别适配已披露，不重新分发作者代码。
- DyHG：`Yuxuan-Chen7/DyHG@dd41a86d76d41c69ae010f97c653877058b0619c`；未发现根目录许可证，许可待核。独立加载原始模型；修正 singleton 轴，保留 Gumbel 推理抽样，不重新分发作者代码。

- 通道 CAMIL：`maojy0914/CAMIL@5481d32a59b0146d9a96f1129bb12ad493acb81a`；未发现根目录许可证，许可待核。固定来源单独下载；原网络和 RAdam 定义在运行时载入。其优化器文件保留 Ross Wightman / 原实现者的版权说明。Lookahead 状态恢复修正见 `docs/CAMIL.md`，不重新分发原代码。
- 上下文 CAMIL：`olgarithmics/ICLR_CAMIL@5f8ef577e040293794d0f695ceb810f890c5e9f7`；未发现根目录许可证，许可待核。PULS 实现为公开方程和固定源码行为的 PyTorch 适配；独立测试从 Git 对象执行作者 TensorFlow 定义作数值对照，不打包作者源码。卷积轴、一维 softmax、初始化、特征投影和训练协议差异见 `docs/CAMIL.md`。

## CATE and ViLa-MIL (0.16)

- CATE: https://github.com/HKU-MedAI/CATE, commit a10fd45146849239ac0c052c45d76c02c81bba6b. No root license found. Author definitions loaded from a separately fetched, verified Git blob; source not redistributed. Full weighted CIB/CFI objective implemented in PULS; CPU/width/std compatibility patches disclosed in docs/VISION_LANGUAGE.md.
- ViLa-MIL: https://github.com/Jiangbo-Shi/ViLa-MIL, commit 68a11cf0d5cf092dd980f0da1cb38ccac8747a82. No root license found. Named author definitions loaded externally; source not redistributed. CPU/singleton compatibility, EOT correction and frozen-text recipe have separately named profiles.
- OpenAI CLIP: https://github.com/openai/CLIP, commit d05afc436d78f1c48dc0dbf8e5980a9d471f35f6, MIT. Text transformer/tokenizer definitions and BPE fetched separately. No pretrained checkpoint is included.

## RankMix (0.17)

RankMix: https://github.com/willpower057/RankMix, commit a6d51301ec878540f4dd185544c265fad5ef1dd2, MIT (Copyright 2023 willpower057). Author backbones and magnitude loss are loaded from separately fetched verified Git definitions; no source or checkpoint redistribution. PULS staged trainer and explicit indexing/padding repairs are documented in docs/RANKMIX.md. The MIT notice is retained in docs/LICENSE.rankmix. IIB-MIL remains audit-only; its incomplete training contract is recorded in docs/IIBMIL_AUDIT.md.

## TPMIL / cDP-MIL / INS (0.18)

Pinned source identities and adapters are documented in docs/PROTOTYPE_DP.md.
TPMIL: LitaoYang-Jet/TPMIL @ 0d89a1f98f164803d92c6f6c686451711873eeca.
cDP-MIL: HKU-MedAI/cDPMIL @ 83f7ed596618266589745c5b1e977894958c85df.
INS: miccaiif/INS @ cd97ba2fa7df2a951f218e6de405f664129931c8 (audit only, no execution profile).
No root license was found in these snapshots; original source is fetched separately and is not vendored or relicensed.
The cDP-MIL profile follows the published trainer's infer/CE path; uncalled variational gamma updates are not claimed implemented.

## HMIL and SCL-WC review (0.19)

HMIL: https://github.com/ChengJin-git/HMIL, GPL-3.0, current commit d4f5ad18d0258a8e1bebe2c5a37e909cc349a946 and historical scalar-head reference af6f85ae38303fa0afb14ea479c47355fb3eebbe. Sources are fetched separately and named definitions loaded from verified Git objects. Architecture/device repairs and differences from the paper are documented in docs/HIERARCHICAL_MIL.md. Historical source is validation evidence, not another method.
SCL-WC: https://github.com/Xiyue-Wang/SCL-WC, commit d8d0ff44e887cadeeaff199cd771af9ee23e5855. README states GPLv3 and academic use, without a root LICENSE in this snapshot. Source review only; no execution profile, code redistribution or license reinterpretation.

## AdvMIL（0.20）

MIT，liupei101/AdvMIL，固定提交 6853786624bbab36c2a08b35dcf20c7bdff1a105。运行时加载已验证的作者 Generator/PrjDiscriminator、ABMIL/ESAT、区域池化、噪声和损失定义。transpose 后 view 使用 reshape 兼容；标准 BCE 修正版单独命名，原字面目标保留 source profile。训练缩放、尾批和推理随机流差异详见 docs/ADVERSARIAL_SURVIVAL.md。

## PIBD (0.21)

Original source: https://github.com/zylbuaa/PIBD, commit `bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78`. Yilan Zhang and collaborators. The pinned README declares Apache-2.0; the pinned tree has no standalone LICENSE. Author code is fetched separately, not redistributed in this package. Named Git-blob definitions are loaded with device-only rewrites and isolated CPU seeding. PULS adaptation and explicit differences are documented in `docs/PIBD.md`.

## MMP (0.22)

Original source: https://github.com/mahmoodlab/MMP, commit `1fd75e37592f7ab47b787f737733ba3c5a7e714c`, Andrew H. Song and collaborators. LICENSE.md declares CC-BY-NC-SA-4.0. Author code and its embedded PANTHER/OT components are fetched separately and loaded from verified Git definitions, not redistributed or relicensed. Device, token assembly, numerical and source behavior differences are documented in `docs/MMP.md`. POT 0.9.6.post1 is called with explicit legacy entropy regularization.

## PathOmics (0.23)

Original author source: https://github.com/Cassie07/PathOmics at 9705d6137f2974923c978597c18e7198bf888e23. No root license found. Author definitions are fetched separately and executed from verified Git blobs, not redistributed under the PULS license. Dimension/device/output adaptations and training repairs are documented in docs/PATHOMICS.md.

## PAMoE (0.24)

Original author source: https://github.com/wjx-error/PAMoE at 7a26f4cc842d2875cfdee33e77de1495cae9dabe. No root license found. Author models, tissue-prior utilities, Cox loss and event-conditioned sampler are read from verified external Git blobs; they are not redistributed or relicensed. Native 1024/512 defaults, dimension adaptation, source vs repaired expert slicing, prototype preparation and recipe boundaries are documented in docs/PAMOE.md. Nystrom-attention 0.0.14 is required for the TransMIL backbone.

## DGMIL (0.25)

Original source https://github.com/miccaiif/DGMIL at a80e5fe51c7e5a3879d4a9b060e49c6b431f70e1, identified by the authors' arXiv 2206.08861. No root license found. Author network, Faiss cluster and extreme-selection definitions are loaded from an external verified checkout, not redistributed. Dynamic training adaptation, covariance cache, source-selection vs stable ties, train-only inference repair and probability reporting differences are documented in docs/DGMIL.md.

## MambaMIL

Source: https://github.com/isyangshu/MambaMIL at `9aea39efc13fdcd3703b5b54c28a64887b4acbd7`. The author README allows non-commercial academic use; the bundled Mamba component has its own Apache-2.0 license. Original source is fetched separately and loaded from pinned Git objects, not redistributed in this package. Model/scan/trainer adaptations and validation boundaries are recorded in `docs/MAMBAMIL.md`.

## MSMMIL

The AFWMamba model, sequence transforms, initializer and custom RAdam are loaded from the user's external checkout of `cool-breeze-and-rain/MSMMIL`, commit `80382115f5e548bc39248fd726bae3db2ead8b7f`. No root model license was found. Its source is not redistributed here; bundled Mamba and optimizer components retain their original attributions. Training and checkpoint adaptations are documented in `docs/MSMMIL.md`.

## MFE-MIL and its pinned timm decoder

The original MFE-MIL shared adapter, mask/decoder and MIL heads are loaded from the user's external `AtlasAnalyticsLab/MFE-MIL` checkout at `813c8e8abf46c05afc2402c06a6c7ac77284fff2` (CC BY-NC-SA 4.0). The required timm 0.9.8 Mlp/Attention/Block definitions are loaded from external `huggingface/pytorch-image-models` at `9afe0bb78e91ea4183cd1bf076f160ec50e9078b` (Apache-2.0, Ross Wightman and contributors). Neither source is redistributed in this package. Original method, training adaptations, inactive source parameters, and numerical-environment differences are documented in `docs/MFE_MIL.md`.

## MamMIL (0.29)

Original author source: https://github.com/Vison307/MamMIL at `64736e1a06a23e6814dfee664ff02b27c72dfedb` (GPL-3.0 root; bundled Mamba Apache-2.0 and component-specific original attributions). Network, SSD reference definitions, graph-notebook utilities, custom RAdam and trainer definitions are read from verified external Git objects. They are not redistributed or relicensed by this package. Original timm0.9.8 initialization definitions use the existing external timm_mfemil lock (Apache-2.0). Numerical backend, source/repaired layer loop, serial HNSW graph assets and named patient extension are documented in docs/MAMMIL.md.

## ReCo-MIL (0.30)

Source https://github.com/zsszc/ReCo-MIL at `27cefbc8d4a9a897b61ceb78b77626bc21d16685`. No root license or verifiable paper link was found in the pinned source. The numerical network, dual wrapper, rarity gate and orthogonal objective are loaded separately from verified author Git objects; no author source is redistributed or relicensed. Source initialization, train-only prototype fit, schedule and public-code behavior are documented in docs/RECOMIL.md.

## MoMIL (0.31)

Author source https://github.com/YuqiZhang-Buaa/MoMIL at `73f0814d56701ff0012054d7c11cbe9637c6cdb3`, linked by the publisher DOI10.1016/j.imavis.2026.105918. No root model license found; bundled Mamba2.2.4 retains Apache-2.0 and its original attributions. Network and numerical definitions are loaded from external verified Git objects, not redistributed. Constructor correction, literal feature-channel order, source checkpoint policy and the named survival-head/trainer extension are documented in docs/MOMIL.md.

## SMMILe spatial and numerical prerequisites (0.31.1; full adapter pending)

Source https://github.com/ZeyuGaoAi/SMMILe at `4c4c79e364e77c16207f211775ef04dda40ff20a`; author README links DOI10.1038/s43018-025-01060-8. Root LICENSE is GPL-3.0 and README additionally states academic/noncommercial purposes. These statements are retained without asserting a new licensing interpretation. Author preprocessing and three network definitions are loaded from verified external Git blobs, not distributed or relicensed. Input identity, device/degenerate numerical repairs, tests and remaining full-method work are documented in docs/SMMILE.md.

## SMMILe full recipes (0.32)

The same locked external source now supplies the original BCE/bibce definitions, data-loader sampling/optimizer semantics and complete numerical epoch/validation loops used in independent tests. Author code remains external and is not redistributed. Serial loading, automatic best-initialized stages, preserved validation reductions, first-refinement-checkpoint correction, finite endpoint/gradient handling and original-order label-free HDF5 export are documented in docs/SMMILE.md. No paper-score reproduction or GPU validation is implied.

## PAMIL prototype attention (0.33)

Source https://github.com/Jiashuai-Liu/PAMIL at `7fe90096f64697bd6b68d4d7209e31f437a6b08f`; no root license file was found in that fixed tree. The model definitions remain in externally obtained verified Git objects, not bundled here. Independent tests execute the original numerical training/validation, clustering and push bodies with listed compatibility repairs. The source CE-on-softmax and collated local-center behavior are separated from explicit repairs; train-only extra instance annotations, frozen prototypes, staged updates and checkpoint selection are documented in docs/PAMIL.md. This is software adaptation evidence, not paper-score reproduction.

## Loss-Attention (0.34)

Author source https://github.com/xsshi2015/Loss-Attention at `71926177d84fc1859b70d135090b163ea6bd58bf`; no root license file was found. The source attention, weighted loss, schedules and legacy Adam definitions remain external verified Git objects. Paper and supplementary material included in that author repository were read to distinguish the classical feature recipe (A2/A4), MNIST MIL code and image-temporal code. Original CNN/ResNet image encoders are replaced by an explicitly documented frozen-WSI-feature boundary; these are named adaptations, not original WSI implementations. Shared classifier weights, loss reductions, EMA and optimizer resets, singleton/NumPy/device compatibility and validation-only selection are described in docs/LOSS_ATTENTION.md. No original paper-score or GPU reproduction is implied.

## MicroMIL / original DGL GAT numerical reference

MicroMIL official repository: https://github.com/kimjongwoo-cell/MicroMIL, commit `66e637a891b7358198896c338b0e37828f3fc1f1`, linked by MICCAI 2025. No root LICENSE found in this commit. Its code is **not redistributed**; the adapter loads named definitions from a separately obtained verified Git object. Frozen-FM WSI transfer, validation/mode/device repairs and retained source-versus-paper differences are recorded in `docs/MICROMIL.md`.

Runtime graph operations use installed DGL 1.1.3 (Apache-2.0). Independent numerical checks obtain the original DGL 0.8.1 GAT class from https://github.com/dmlc/dgl, commit `1453bf2ceb23f413c23e3304dd94b40bf96b4cd7` (Apache-2.0). This reference code also remains external and retains its LICENSE in the sparse checkout. No DGL source files are bundled in this project's wheel or source archive.

## ProbSA-MIL, author-maintained torchmil implementation

The original https://github.com/Franblueee/ProbSA-MIL README explicitly links the author-maintained https://github.com/Franblueee/torchmil implementation. The adapter uses pinned commit `2472b8bee84b906eac9351f1ec49d430c577bbcd` (Apache-2.0), fetched as an external sparse checkout with its LICENSE. Models, attention, transformer, graph builders, schedulers and the independent reference trainer are loaded from verified Git definitions; none of those source files are distributed here. Padding/sparse KL, evaluation dropout and predictive-mixture differences are documented in `docs/PROBSA.md`. This is an explicit author-library version, not unpublished original experiment scripts or paper-score reproduction.

## GTP / official GraphCAM

Author repository https://github.com/vkola-lab/tmi2022, commit `0d395a160a969578fa5ce66a8a7f0fd0817712b4`, MIT, Copyright (c) 2021 Kolachalama laboratory. It identifies the TMI2022 publication. The adapter fetches an external sparse checkout with its LICENSE and loads verified Git definitions for GCN, Classifier, ViT, LRP and independent training references; no author source files are bundled here. Complete graph classifier training and native GraphCAM are preserved at an explicit frozen-FM boundary. Device/logit compatibility and named zero-edge/BN/visualization repairs are recorded in `docs/GTP.md`. PyG2.6.1 installed min-cut is used as the runtime backend and independently checked against formulas.

SETMIL uses external verified Git objects from Louis-YuZhao/SETMIL at4243f633adb9e8a84cea11f92abf3d3b0ede126e. No author source is redistributed. The original transformer block includes a Clear BSD notice referring to a LICENSE absent from the pinned tree; license remains unverified. Python iRPE fallback, model and augmentation definitions retain original mathematics with explicit import compatibility and crop/view repairs described in docs/SETMIL.md.

StableMIL uses externally fetched author Git objects at67b279157680c8c96104823a9e6f2ef20b679d11; no root license was found, so author source is not redistributed. Its original timm1.0.21 Mlp/LayerScale helpers come from Apache2.0 source625386b96151ec49dede06190b0deaa3909cd4de, also external. See docs/STABLEMIL.md for complete task recipes and explicit input/execution/accumulation repairs.

MONAI1.5.1 / Project-MONAI tutorial: Apache-2.0, external pinned source and exact-version MONAI transforms; native torchvision0.20.1 ResNet50 is BSD-3-Clause. OpenSlide Python/binary follow upstream LGPL-2.1 licenses. No upstream code or ImageNet weights vendored; local initialization checked by full SHA256.

Campanella MIL-RNN uses external Git objects from MSKCC-Computational-Pathology/MIL-nature-medicine-2019 at daecb7293b5ea886ba222008f2592e1e29e9dac6. No LICENSE was found in the fixed author tree, so license status remains unverified and no author source is redistributed. Native torchvision0.20.1 ResNet34 and local official ImageNetV1 checkpoint retain their upstream terms; weights are not bundled. The original recurrent definition/group selectors are loaded from verified Git objects. The full hard-MIL/Adam and recurrent/SGD recipe, raw ROI data bridge and explicit compatibility boundaries are described in docs/CAMPANELLA.md.
# MultiAttentionMIL

The adapter imports the original `mdsatria/MultiAttentionMIL` definitions at commit
`fae5840f801c0485eb22f2e7119d392f2fa975f3` from a separately fetched Git checkout.
No license file was present in that pinned tree; its code is not bundled or
relicensed by this project. The adapter and validation disclose the supplementary
MNIST-to-frozen-WSI transfer, validation selection, and named inference Dropout repair.

## TTIS

The adapter uses external verified Git objects from https://github.com/QuIIL/TTIS,
commit `ceb08d3bd8bc795845358850e314132b6231a921`. No root license file was found;
the embedded MUSK subtree separately includes CC-BY-NC-ND-4.0 terms. Neither author
code nor MUSK weights are redistributed. Original head, sampler, fusion, clustering
and Lookahead definitions are loaded with documented import compatibility changes.
Source bag-CE versus full CLAM loss, primary versus aligned-secondary semantic spaces,
CPU versus native GPU clustering, and FP32 validation boundaries are in docs/TTIS.md.

## 2DMambaMIL

Official https://github.com/AtlasAnalyticsLab/2DMamba, fixed commit
`9fe12aaa866ddbb83b66acc4ef400abf8613ce6a`, remains external verified Git source.
No root LICENSE was found. Its Mamba-derived components retain their original
attributions in that external tree; no author code or binary is redistributed.
Literal PyTorch behavior, explicitly repaired derivatives and two vertical-scan
repairs are separate execution profiles. Native CUDA builds require independent
source/binary/ABI provenance. Full recipes and boundaries are in docs/MAMBA2D.md.

## SI-MIL

Original https://github.com/bmi-imaginelab/SI-MIL, fixed commit
`3ffb218672ffe0e4a183924af50e5920b498483d`, is loaded from external verified
Git objects. No root license was found; author source, nuclei segmentation
weights and PathExpert datasets are not redistributed. The complete dual-branch
model, patch resampling and training recipe retain their original attribution.
TRAIN-only normalization and the full Gaussian top-K derivative are explicit
documented repairs; the literal original derivative remains a separate profile.
See docs/SIMIL.md for raw descriptor provenance, recipe and interpretation limits.

## HIPT

External verified source: https://github.com/mahmoodlab/HIPT, fixed commit
`780fafaed2e5b112bc1ed6e78852af1fe6714342`. The root license is Apache-2.0
with the Commons Clause; it must not be treated as unrestricted Apache-2.0.
Author code, pretrained teachers, DINO patch checkpoints and data are not
redistributed. The complete original HIPT_LGP_FC/ViT4K definitions and optimizer
are loaded from verified external Git objects. Supplied public teacher tensors
are strictly authenticated; the adapter does not substitute random weights.
Native classification recipes, geometry, pretrained-space and survival-source
boundaries are documented in docs/HIPT.md. Cite the original HIPT, ViT and DINO
work when using the original method.

## H2-MIL

The adapter loads external verified Git objects from https://github.com/lin-lcx/H2-MIL,
commit `ea472fefd5eab553c58041a5acc203ac7d467d06`. No root license file was
found in that pinned tree. Author code, data, KimiaNet weights and tools are not
redistributed. Complete original RAConv/IHPool/GCN definitions are used with
explicit shape/device/PyG compatibility and exact sparse edge contraction.
Literal source classification and the C-way/logits repair are named separately.
Actual three-level data requirements, hard-fitness gradient boundary, complete
source training recipe and differences from the paper are in docs/H2MIL.md.

AGDR-MIL uses the separately fetched original source at `shuiguoc/AGDR-MIL@8e830683c0a695e1e1983f01d35dc85471bf327c`; no root license was found in that fixed tree. The actual hard grading/cosine redistribution/low-mask/dual Sigmoid-CE model and source training recipe are loaded from verified Git objects. The only source instrumentation records mask row indices. The adapter retains blocked grading gradients and stochastic evaluation, including native sequential-loader base-seed consumption; completely identical inputs and undefined scores are rejected explicitly. Two named YAML/TCGA presets, their differences, full recipe and standardized versus source metrics are documented in docs/AGDRMIL.md. No author source, weights or datasets are redistributed.

## C2Aug external source and complete paper-reference adaptation

The C2Aug native TransMIL definitions are loaded from the external author
repository `https://github.com/weiaicunzai/mixup`, commit
`14cb53bc71f5161105b0dbef45351243d7330561`. No license was found in that pinned
tree, so its code is not redistributed in this package. The adapter's complete
paper-reference training mechanisms, its explicit source differences and
independent implementation choices are documented in `docs/C2AUG.md`.

## PathRWKV

External author source: `https://github.com/Puzzle-Logic/PathRWKV`, commit
`8818fa5711666bd8d30672f02bdcd98634389b70`, Apache-2.0. The source is fetched
separately; this distribution does not bundle its CUDA or Python files. Full
TimeMix/ChannelMix, task projection/heads, losses and optimizer are loaded from
verified Git objects. The native CUDA kernels have an explicit tensor-reference
alternative for identified CPU experiments, without automatic fallback.
Exact factorized position storage, source/corrected Cox, missing-label masks,
streamed token-state repairs and benchmark selection are documented in
`docs/PATHRWKV.md`. Cite the original PathRWKV and RWKV work when using them.

## ProtoSurv (0.53 candidate)

wjx-error/ProtoSurv at551eaa3c0a2844e2c3557ef412d8a1fd2fec6e20. No root source license found. Full original graph/prototype/fusion and loss definitions plus augmentation/sampler are loaded from separately fetched verified Git objects; no original code, category model, pretrained prototype or checkpoint redistribution. Full vocabulary, RNG, time ties and accumulation corrections are named explicitly in docs/PROTOSURV.md. Independent metric checks load the MIT lifelines0.30.0 reference at47afb1c1a272b0f03e0c8ca00e63df27eb2a0560, also externally fetched. Synthetic CPU validation is distinct from native GPU or paper-score reproduction.

## GAT-Mamba (0.54 candidate)

rina-ding/gat-mamba at18979497990a66ffe6a75880f0336f85914552f0, GPL-3.0. Full network,21category edge mapping and original trainer mathematical definitions are loaded from separately fetched verified Git objects; source and classifier weights are not redistributed. The declared dependency is state-spaces/mamba v1.2.0.post1 at34076d664838588a3c97727b263478ab9f621a07, Apache-2.0, also externally fetched. CPU execution uses the actual same-version author Mamba slow branch and selective_scan_ref; the explicit CUDA backend requires author-declared mamba-ssm1.2.0.post1/causal-conv1d1.0.2, verifies its installed scan wrapper and records native binaries. FM input-dimension parameterization, separately named edge_dim16/singleton corrections, input provenance and complete training details are documented in docs/GATMAMBA.md.

C²MIL: mimic0127/C2MIL at fd4d54fc115466e43019c77015cdef01d90c85c3, MIT (Copyright 2025 mimic0127). Externally fetched source, private import adaptation and device/data/checkpoint bridges; see docs/C2MIL.md.

TRIAGE-MIL: barathi-1993/TRIAGE-MIL at e97e52a32f1358974b0145532fdec5f5d75c6864. No license file is declared in the checked original tree. Author source is externally fetched, not bundled; private source definitions and input/training bridges are documented in docs/TRIAGEMIL.md.

### DeepGraphSurv candidate

uta-smile/DeepGraphSurv at3149752deba68067f1a39b9f3d212cb16ffaf560. No license file found in the pinned tree; author definitions are fetched externally and verified before private loading. The independent Torch equation port, original TensorFlow comparison and explicit source I/O/protocol differences are documented in docs/DEEPGRAPHSURV.md. Natural WSI/paper reproduction is not claimed.

### SCL-WC complete paper reference (0.58 candidate)

Xiyue-Wang/SCL-WC at d8d0ff44e887cadeeaff199cd771af9ee23e5855 is loaded privately from verified Git objects. The repository README declares GPLv3 and academic-use conditions, with no root LICENSE found; those terms are retained in the source lock. No author implementation or paper PDF is bundled. New reference code reconstructs the complete downstream CDA/PNM/live-anchor WSCL method, with all source/paper differences and unresolved implementation choices named in docs/SCLWC.md. The literal original source-control profile is separately labelled as incomplete and is never substituted for the full method.
# INS frozen-feature route

`miccaiif/INS` at `cd97ba2fa7df2a951f218e6de405f664129931c8` has no
root license in the reviewed tree. Its definitions are fetched privately by
`milplus setup --method ins` and are not copied into this distribution. The
adapter and `docs/INS.md` identify the original files, complete recipe and named
source repairs; permission to redistribute author code is not inferred from
public GitHub visibility. PyTorch/torchvision remain separate dependencies.

# CoD-MIL source-repaired recipe

Jiangbo-Shi/CoD-MIL at `0fd748dc91ffd889bf13dbffd4db633b158809d6`
has no root license in the inspected tree. Author definitions are privately
fetched at setup time and are not vendored into the package. `docs/CODMIL.md`
records complete source objectives, physical/text requirements and explicit
C-way/dimension/I-O compatibility repairs. Public visibility does not imply
redistribution permission.


## FOCUS and CONCH text dependency (0.61 candidate)

FOCUS author source `dddavid4real/FOCUS@66c4015d5ba09657f4c8183bc06947faecd5b01f` is Apache-2.0. Private original definitions are loaded without vendoring. Actual full native text dependency is reviewed `mahmoodlab/CONCH@141cc09c7d4ff33d8eda562bd75169b457f71a62`, whose independent CC-BY-NC-4.0/non-commercial terms remain applicable; original FOCUS does not pin a historical dependency commit. No CONCH model weights, BPE assets or dependency source are redistributed in this package. User-provided native weights and actual local pinned tokenizer are required. Author README public checkpoint provenance and all complete original text/visual/recipe differences are recorded in docs/FOCUS.md.

## PAMIL dynamic policy (0.62 candidate)

`titizheng/PAMIL@d62960af2c115f87d570a0fbf8852049745b4e94` has no root license in the inspected tree. Original SFFR, history attention, base/class transformer, Gaussian-GRU actor/critic and PPO are loaded privately from the pinned Git objects, not redistributed. Complete training is explicitly reconstructed from the primary CVPR2024 paper and public supplement; missing historical choices, source-vs-paper sampling/history discrepancies, device and scalar/single-step ports are described in docs/PAMIL_POLICY.md. This is distinct from prototype-attention PAMIL. Neither private model weights nor the original supplemental PDF is bundled.

## DT-MIL

Official yfzon/DT-MIL@f52530811d0b1b591e10eab2fad9fd2ac1573b98 (Apache-2.0), Tencent2021; derived source headers also acknowledge Deformable DETR/SenseTime2020. Fixed author modules are loaded privately, including the original Torch deformable sampler; no clinical sample data or external weights are distributed. Complete training is explicitly reconstructed because the public release provides inference only. See docs/DTMIL.md.

## AMIML

Boomwwe/AMIML, commit `7c038261d1a52a4eb20d4b4f444e730c36bf5cd2`, GPL-3.0. Original class definitions are loaded from private verified Git objects, not silently substituted or bundled as unlicensed third-party code. Complete named source/output/input/loader/training repairs are documented in docs/AMIML.md. Primary preprint arXiv:2206.00455v1, paper SHA256 `cc6d541b00e66219c030ad135810903f732ea2613c529ff17d335c25eb494075`.

## DisPro (development, no admitted profile)

Innse/DisPro@98473003de24e5a8f28143442d3dd48bd159aff1. The author README declares Apache-2.0; no standalone LICENSE blob is present in this pin. Original definitions execute from private verified Git objects. BioBERT-v1.2 weights remain external, fixed by revision and content SHA256; see docs/DISPRO.md.
# SLIP development source and original CLIP assets

SLIP source: LTS5/SLIP at 9351c4480ec8d7a8216316a1861025b4f086ac78.
No standalone repository license is present at that fixed commit. Author source
is fetched separately and imported privately; it is not republished in this
package. Actual original CLIP checkpoints and the source BPE vocabulary are
verified separately and are not bundled. See docs/SLIP.md for the exact assets,
preserved source mechanisms, complete recipe and explicitly named ports.

VLSA: liupei101/VLSA, fixed commit 915f37afcbd86cd6c33ebb30ab0caa5def0f9ab9, CC-BY-NC-ND-4.0. Author definitions are read from the separately fetched original source; no original source or weights are redistributed. Native CONCH checkpoint and ordered prompt/query assets are explicitly bound; see docs/VLSA.md for recipe ports and the unresolved author-example numeric discrepancy.

ZoomMIL privately loads the MIT-licensed histocartography/zoommil source at commit da7bb7fa71629d34e9b04859c1db791ac8330f5d (https://github.com/histocartography/zoommil). Full three-level architecture and perturbed-top-K definitions remain original; device/saver ports, metadata, complete trainer integration and software fixtures are documented in docs/ZOOMMIL.md. Original author files and model weights are not redistributed.

AHDMIL: JiuyangDong/AHDMIL at 2f38354386165167815eb2998424ad63792a65bf. README declares GPLv3 and non-commercial academic intent. Required DMIN preparation uses JiuyangDong/HDMIL at ef3d4681054970208e94f8205bd96b3180c00926. Both sources are fetched separately and read from verified Git objects, with no source or pretrained weights redistributed. Complete recipe and explicit source repairs: docs/AHDMIL.md and docs/AHDMIL_AUDIT.md.

## KMG-Mamba audit and timm DropPath reference

KMG-Mamba source: https://github.com/dmhdmhdmh/KMG-Mamba at 4b5075bb2d3eddf46f9bc4f46175d250b134e27c. Overall license not declared in this pinned tree; author implementation is fetched privately and not vendored.

The unmodified test reference `tests/fixtures/kmgmamba/timm_v0_4_12_drop.py` comes from https://github.com/huggingface/pytorch-image-models/blob/v0.4.12/timm/models/layers/drop.py, SHA256 99cd8bf3d9e503ca4633264141fda078662e4bb7e94367a17c7a5fda646eab4d. Copyright 2020 Ross Wightman. Apache-2.0 license retained at `tests/fixtures/kmgmamba/LICENSE.timm`. Runtime legacy DropPath follows the same attributed uniform-plus-floor algorithm; this does not grant a blanket license to the KMG-Mamba repository.

## LRMIL source under review

Pinned source https://github.com/hvcl/LRMIL at 66ea1c5a447ed2a830bb1af6739ab9261e61000d declares Apache-2.0. The source is loaded privately. Its CONCH vision dependency and pretrained weights retain the original CONCH terms; neither the LRMIL license nor this adapter relicenses those weights. The isolated implementation is not yet registered as a runnable method.


## SAM-MIL

FangHeng/SAM-MIL, commit ccff640825321a12658daa0a85c03390bc9d367b, Apache-2.0. Original definitions are loaded from the pinned external source, not republished. Full attention recipe, source controls and explicit repairs: docs/SAMMIL.md. No historical paper-score or natural-WSI claim.


## Cluster-to-Conquer (C2C)

YashSharma/C2C, commit d53b6350372117bf3e5251044a856e0251d810b2, MIT. Original network/loss/clustering/train definitions are loaded from fixed external Git objects. Historical support assets are checksum-pinned torchvision0.6.1 ResNet definitions/weights (BSD-style torchvision license), PyTorch1.5.1 BatchNorm (BSD-style PyTorch license), and Albumentations0.4.6 ToTensor/Compose (MIT); retrieved as external assets, not silently substituted by modern behavior. Full route and explicit ports: docs/C2C.md. No natural-WSI or historical paper-score reproduction claim.

## SparseConvMIL and SparseConvNet

SparseConvMIL: MarvinLer/SparseConvMIL,d7dd705bc3dd4ebfb6a36f0589698400c9fa4c15,AGPL-3.0. Author code remains separately fetched and verified, not vendored. SparseConvNet: facebookresearch/SparseConvNet,cf251d058959a9dbaccb25fc919dc4f4548be232,BSD-3-Clause. Full original C++ extension and Python/autograd layer compiled unchanged in a dedicated overlay. Adapter independently binds data/training/evaluation; public-recipe differences from the paper and all modern device/constructor ports are explicitly documented.

## ConMIL

Author code anxuanhan/ConMIL,aabb5386d15a6002a3a39437ea46c0d99d29a2f3,MIT; fetched separately and verified, not republished under the package license. Complete original attention/KAN and training operators retained. Immutable zero-stride KAN knot grids are materialized without numerical changes for checkpoint restoration. CONCH source141cc09c7d4ff33d8eda562bd75169b457f71a62 and actual pretrained weights keep their original research/noncommercial terms; weights are not redistributed. Actual native causal/CLS text encoding, prompt review provenance and visual/text compatibility are recorded separately.

## MIL-Adapter (0.80 development)

Original source: https://github.com/cvblab/MIL-Adapter at `ade11bb19c3f09a35da9e8255a87a079217aef48`. No LICENSE file was present at that revision. Source and public text prototype matrices are privately fetched and verified against original Git objects; neither is vendored in this distribution. Complete model definitions are loaded into isolated namespaces with explicitly documented device/temperature/ABMIL tuple/KEEP input-width ports. Imported text generation details remain undisclosed. See docs/MIL_ADAPTER.md.

## MIL with Center Embeddings (0.81 development)

PhilipChicco/MICCAI2020mil, commit `871b1ecad3f0480c526536b11fafe4d759273a25`, GPL-3.0. Original network, center objective, selection and loader definitions are read from separately fetched, verified Git objects; author source is not redistributed. Original torchvision0.7 transforms and ImageNet ResNet34-333f7ec4 weights are hash-pinned external assets, retaining their own original terms. The full original-image joint recipe, undefined bag_feat repair and explicit recovery/data ports are documented in docs/CENTER_EMBEDDINGS.md. No real-WSI experiment or paper-score reproduction has been completed.

## ConVL (registered development; acceptance pending)

Original source junjianli106/ConVL, f1017a7ea0f51ff8d436b1334f937b85aa86c46b. No repository-wide license was found at the pinned revision. Original model, concept dictionary, text fork, survival objective and optimizer definitions are fetched privately and not redistributed. Native CONCH weights retain the original CONCH terms and are not bundled. The complete source patient trainer, serialization ports, data-label provenance, original-loop comparisons and remaining native/release gates are documented in docs/CONVL.md. Two explicit FP16/FP32 profiles are registered as implemented development candidates; formal acceptance and real-WSI/paper-score reproduction remain pending.

## M4

Original source Bigyehahaha/M4, commit2b0f92ef872fcd9997f1ae6b4ba6565fc53eb332, MIT copyright2024yezhang. Original model/training/collate definitions are loaded from the fixed private checkout; original license retained in docs/LICENSE.m4. No upstream patient CSV or weights are redistributed. Full source recipe, split protocol and FM ports are documented in docs/M4.md. Native, whole-library and final-wheel gates remain separate.

## MUSE (released-code development condition)

Original source JiahaoXu-god/CVPR2026_MUSE, commit0bd0e1abb402e772ed71cf208cfbd4b870175a00. No LICENSE file was present at this revision. Original definitions and all original knowledge CSV files are fetched separately and verified against Git objects; neither is redistributed. Actual CONCH weights and tokenizer retain their original terms and are not bundled. The complete released20-view trainer and explicit device/serialization/diagnostic-RNG ports are documented in docs/MUSE.md. The released code has material differences from the paper, including short class-token overwrite and recursive text-layer reinitialization; this profile is explicitly a released-code condition and does not establish paper-faithful reproduction or formal admission.

## GMMamba and IIB-MIL development

GMMamba: titizheng/GMMamba, commit70576f1aaeef0d7041d6860799ade1ac905b85aa, no LICENSE at the pinned revision. Definitions are privately loaded from original Git objects, not redistributed. Its omitted BiMamba dependency is explicitly bound to actual isyangshu/MambaMIL9aea39efc13fdcd3703b5b54c28a64887b4acbd7 (original noncommercial academic terms; bundled Mamba Apache-2.0). This is a documented dependency reconstruction, not a claim to possess the unpublished GMMamba backend. Complete recipe, ports and remaining verification are in docs/GMMAMBA.md.

IIB-MIL: TencentAILabHealthcare/IIB-MIL, a354ec6ea2b9ea2159e6784d54a6afb355c04b50, no LICENSE at pinned revision. Complete model/prototype/confidence definitions remain external. Native component checks do not establish a complete trainer; no runnable IIB-MIL profile is registered. See docs/IIBMIL.md.

Coupled ICMIL uses external Dootmaan/ICMIL at c2fd2faa4e85cf85527e228573b434a8cc461cba (MIT); original method source is privately loaded. The DTFD condition vendors the unmodified torch/optim/adam.py from the official PyTorch1.4.0+cpu wheel under its retained BSD notice in src/milplus/vendor/torch14_adam/LICENSE. MANIFEST.json records the exact source SHA256. The small optimizer base bridge keeps the original zero_grad default; source gradient decay and p.data mutation semantics are preserved, not replaced by delayed steps or saved-tensor hooks.


LPGMIL: FPXMU/LPGMIL at6b14bb01dff042a09bd34ebe956d696e665a7cd0 has no repository LICENSE. Original definitions are loaded privately from pinned Git objects, not redistributed. Mamba1.1.1 uses the official Apache2.0 source atfb7b5310fa865dbd62aa059b1e26f2b431363e2a. Complete reference repairs and protocol differences are recorded in docs/LPGMIL.md.

MExD: JWZhao-uestc/MExD at 1d4d438e2b19d0fee9cb55328faa84b6ca37fe0b has no repository LICENSE. Original TransMIL, conditional denoiser, shifted diffusion and RAdam definitions are loaded privately from verified original Git objects and are not redistributed. Nyström attention is explicitly bound to this project's retained TransMIL reference implementation and license. The incompatible released MoE graphs, absent private checkpoint, complete equation-based reconstruction and native source/wheel gates are documented in docs/MEXD.md. No real WSI or paper-score reproduction is claimed.
