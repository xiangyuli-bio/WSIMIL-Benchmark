"""Reconstruct the user's admission scope. This is a scope ledger, not evidence of reproduction."""
import json
from pathlib import Path

GROUPS = {
"A_frozen": "mean_mil:MeanMIL|max_mil:MaxMIL|abmil:ABMIL|chowder:CHOWDER|clam:CLAM|dsmil:DSMIL|transmil:TransMIL|dtfd:DTFD-MIL|frmil:FRMIL MICCAI 2022|cbmil:CBMIL|additive_mil:Additive MIL|varmil:VarMIL / DeepSMILE|h2t:H2T|iibmil:IIB-MIL|ilra:ILRA-MIL|ins:INS|mmil:MMIL-Transformer|tpmil:TPMIL|acmil:ACMIL|cdpmil:cDP-MIL|dgrmil:DGR-MIL|focusmil:FocusMIL|madmil:MAD-MIL|mambamil:MambaMIL|pamil_policy:PAMIL dynamic policy|pamil_prototype:PAMIL prototype attention|panther:PANTHER|retmil:RetMIL|camil_channel:CAMIL channel attention|efficientmil:EfficientMIL|fouriermil:FourierMIL|gmmamba:GMMamba|linmil:Lin-MIL|mico:MiCo|msmmil:MSMMIL|ptcmil:PTCMIL|agdrmil:AGDR-MIL|caprmil:CAPRMIL|litemil:LiteMIL|momil:MoMIL|pneamil:PNEA-MIL|recomil:ReCo-MIL|sgpmil:SGPMIL",
"B_spatial_graph": "bayesmil:Bayes-MIL|smmil:SmMIL / Smooth Operator|wikg:WiKG|psamil:PSA-MIL|dtmil:DT-MIL|gtp:GTP|setmil:SETMIL|slidegraph:SlideGraph+|camil_context:CAMIL context aware|longmil:LongMIL|mammil:MamMIL|mamba2dmil:2DMambaMIL|attrimil:AttriMIL|dyhg:DyHG|probsamil:ProbSA-MIL|smmile:SMMILe|pathrwkv:PathRWKV|stablemil:StableMIL",
"C_recipe_plugin": "remix:ReMix|sclwc:SCL-WC|ibmil:IBMIL|mhim:MHIM-MIL|rankmix:RankMix|augdiff:AugDiff|pmil_shapley:PMIL Shapley pseudo-bag|prodiv:ProDiv|psemix:PseMix|aem:AEM|mfcmil:MFC-MIL|mil_dropout:MIL-Dropout|packmil:PackMIL|carmil:CAR-MIL|mfemil:MFE-MIL|mrblock:MR-Block / MR-AB-MIL|nnmil:nnMIL|sdfp:S-DFP|dgmil:DGMIL|rrt:R2T-MIL / RRT-MIL|asmil:ASMIL|c2aug:C2Aug|fcmil:FC-MIL|mammoth:MAMMOTH|ttis:TTIS",
"D_multiscale": "csmil:CS-MIL|h2mil:H2-MIL|msattention:Multi-scale Attention MIL Wibawa|zoommil:ZoomMIL|hagmil:HAG-MIL|mgtrans:MG-Trans|hdmil:HDMIL|ahdmil:AHDMIL|kmgmamba:KMG-Mamba|lrmil:LRMIL",
"E_encoder_training": "campanella_rnn:Campanella MIL-RNN|rmdl:RMDL|center_embeddings:MIL with Center Embeddings|c2c:Cluster-to-Conquer|monai_mil:Dependency-aware MIL / MONAI|sparseconvmil:SparseConvMIL|attention2majority:Attention2majority|pmil_prototypical:PMIL Prototypical|promptmil:Prompt-MIL|icmil_coupled:ICMIL Iteratively Coupled|macnet:MacNet",
"F_survival_omics": "patchgcn:Patch-GCN|varpool:VarPool|advmil:AdvMIL|scmil:SCMIL Sparse Context|deepgraphsurv:DeepGraphSurv|deepattnmisl:DeepAttnMISL|protosurv:ProtoSurv heterogeneous graph|c2mil:C2MIL|gatmamba:GAT-Mamba|triagemil:TRIAGE-MIL|he2rna:HE2RNA|amiml:AMIML|mcat:MCAT|porpoise:PORPOISE|cmta:CMTA|motcat:MOTCat|pathomics:PathOmics|healnet:HEALNet|mmp:MMP|pibd:PIBD|survpath:SurvPath|dispro:DisPro|pamoe:PaMoE",
"G_vision_language": "cate:CATE|vilamil:ViLa-MIL|codmil:CoD-MIL|focus:FOCUS|slip:SLIP|vlsa:VLSA|conmil:ConMIL|convl:ConVL|dyko:DyKo|libramil:Libra-MIL|miladapter:MIL-Adapter|muse:MUSE|taxomil:TaxoMIL",
"H_extra_supervision": "hybrid_supervision:Hybrid Supervision Learning|sammil:SAM-MIL|simil:SI-MIL|dspamil:DSPA-MIL / DSPA-U-MIL|hmil:HMIL|lpgmil:LPGMIL|m4:M4|protomil:ProtoMIL",
"I_continual": "akdpmp:AKD-PMP|comel:CoMEL|qpmilvl:QPMIL-VL|kmpmil:KMP-MIL",
"J_pretrained_slide": "hipt:HIPT|feather:FEATHER / MIL-Lab models|icmil_context:ICMIL In-context",
"K_other_modalities": "loss_attention:Loss-based Attention|micromil:MicroMIL|mggmmil:MGGMMIL autofluorescence",
"L_composite": "lnplmil:LNPL-MIL|nciemil:NcIEMIL|mexd:MExD",
}

# Source leads from the user-provided plan; not all have been verified in this checkout.
SOURCES = {
"mammil": "Vison307/MamMIL",
"recomil": "zsszc/ReCo-MIL",
"momil": "YuqiZhang-Buaa/MoMIL",
"gmmamba": "titizheng/GMMamba",
"mfemil": "AtlasAnalyticsLab/MFE-MIL",
"msmmil": "cool-breeze-and-rain/MSMMIL",
"dgmil": "miccaiif/DGMIL",
"pamoe": "wjx-error/PAMoE",
"pathomics": "Cassie07/PathOmics",
"mmp": "mahmoodlab/MMP",
"h2t": "vqdang/H2T", "abmil": "AMLab-Amsterdam/AttentionDeepMIL", "clam": "mahmoodlab/CLAM",
"dsmil": "binli123/dsmil-wsi", "transmil": "szc19990412/TransMIL", "dtfd": "hrzhang1123/DTFD-MIL",
"frmil": "PhilipChicco/FRMIL", "remix": "TencentAILabHealthcare/ReMix",
"ibmil": "HHHedo/IBMIL", "mhim": "DearCaat/MHIM-MIL", "rrt": "DearCaat/RRT-MIL",
"mambamil": "isyangshu/MambaMIL", "mamba2dmil": "AtlasAnalyticsLab/2DMamba",
"wikg": "WonderLandxD/WiKG", "patchgcn": "mahmoodlab/Patch-GCN", "survpath": "mahmoodlab/SurvPath",
"pibd": "zylbuaa/PIBD", "mexd": "JWZhao-uestc/MExD", "he2rna": "owkin/HE2RNA_code",
"linmil": "charlotterchtr/Lin-MIL", "attrimil": "MedCAI/AttriMIL", "nnmil": "Luoxd1996/nnMIL",
"hipt": "mahmoodlab/HIPT", "porpoise": "mahmoodlab/PORPOISE", "mcat": "mahmoodlab/MCAT",
    "ilra": "jinxixiang/low_rank_wsi", "acmil": "dazhangyu123/ACMIL", "aem": "dazhangyu123/AEM",
}

SOURCES.update({'protosurv':'wjx-error/ProtoSurv','gatmamba':'rina-ding/gat-mamba'})

SOURCES.update({'frmil': 'PhilipChicco/FRMIL', 'ibmil': 'HHHedo/IBMIL', 'rrt': 'DearCaat/RRT-MIL', 'linmil': 'charlotterchtr/Lin-MIL', 'attrimil': 'MedCAI/AttriMIL', 'nnmil': 'Luoxd1996/nnMIL', 'madmil': 'tueimage/MAD-MIL', 'ptcmil': 'ubc-tea/PTCMIL', 'litemil': 'hkussaibi/LiteMIL', 'sgpmil': 'mandlos/SGPMIL', 'pneamil': 'Chen-Jxiang/PNEA-MIL', 'carmil': 'ImaneCR/CAR-MIL', 'asmil': 'Linfeng-Ye/ASMIL', 'fcmil': '7FFDW/FCMIL', 'mrblock': 'BearCleverProud/MR-Block', 'retmil': 'Hongbo-Chu/RetMIL', 'momil': 'YuqiZhang-Buaa/MoMIL', 'iibmil': 'TencentAILabHealthcare/IIB-MIL', 'survpath': 'mahmoodlab/SurvPath', 'porpoise': 'mahmoodlab/PORPOISE', 'panther': 'mahmoodlab/PANTHER', 'chowder': 'owkin/HistoSSLscaling', 'varpool': 'mahmoodlab/varpool'})

SOURCES.update({'focusmil': 'Raymvp/FocusMIL-and-other-max-pooling-methods', 'h2t': 'vqdang/H2T', 'cbmil': 'wwu98934/CBMIL', 'dgrmil': 'ChongQingNoSubway/DGR-MIL'})

SOURCES.update({'psemix': 'liupei101/PseMix', 'remix': 'TencentAILabHealthcare/ReMix', 'prodiv': 'liupei101/PseMix'})

SOURCES.update({"additive_mil": "lingxitong/MIL_BASELINE", "fouriermil": "lingxitong/MIL_BASELINE"})
SOURCES.update({'mmil': 'hustvl/MMIL-Transformer'})
SOURCES.update({'cmta': 'FT-ZHOU-ZZZ/CMTA', 'motcat': 'Innse/MOTCat'})

SOURCES.update({"csmil": "hrlblab/CS-MIL", "hagmil": "BearCleverProud/HAG-MIL", "h2mil": "lin-lcx/H2-MIL"})

SOURCES.update({'smmil': 'Franblueee/SmMIL', 'psamil': 'SharonPeled/PSA-MIL',
                'slidegraph': 'wenqi006/SlideGraph', 'bayesmil': 'ralphc1212/bayes-mil'})

SOURCES.update({'caprmil': 'mandlos/CAPRMIL', 'varmil': 'NKI-AI/dlup-lightning-mil'})
SOURCES.update({'mico': 'junjianli106/MiCo', 'efficientmil': 'chengyingshe/EfficientMIL',
                'agdrmil': 'shuiguoc/AGDR-MIL', 'pamil_prototype': 'Jiashuai-Liu/PAMIL'})

SOURCES.update({'pibd': 'zylbuaa/PIBD', 'scmil': 'yang-ze-kang/SCMIL', 'deepattnmisl': 'uta-smile/DeepAttnMISL', 'advmil': 'liupei101/AdvMIL'})

SOURCES.update({'mil_dropout': 'ChongQingNoSubway/MILDropout', 'packmil': 'FangHeng/PackMIL'})

SOURCES.update({'sdfp': 'thomascong121/S-DFP', 'mfcmil': 'WissingChen/MFC-MIL', 'dyhg': 'Yuxuan-Chen7/DyHG'})

SOURCES.update({'camil_channel': 'maojy0914/CAMIL', 'camil_context': 'olgarithmics/ICLR_CAMIL'})

SOURCES.update({'healnet': 'konst-int-i/healnet'})

SOURCES.update({'cate': 'HKU-MedAI/CATE', 'vilamil': 'Jiangbo-Shi/ViLa-MIL'})

SOURCES.update({'rankmix': 'willpower057/RankMix'})
SOURCES.update({'monai_mil': 'Project-MONAI/MONAI'})
SOURCES.update({'campanella_rnn': 'MSKCC-Computational-Pathology/MIL-nature-medicine-2019'})
SOURCES.update({'stablemil': 'theeeqi/stableMIL', 'longmil': 'invoker-LL/Long-MIL', 'pmil_shapley': 'RenaoYan/PMIL'})
SOURCES.update({'setmil': 'Louis-YuZhao/SETMIL'})
SOURCES.update({'gtp': 'vkola-lab/tmi2022', 'c2aug': 'weiaicunzai/mixup'})
SOURCES.update({'probsamil': 'Franblueee/torchmil'})
SOURCES.update({'micromil': 'kimjongwoo-cell/MicroMIL'})
SOURCES.update({'smmile': 'ZeyuGaoAi/SMMILe', 'loss_attention': 'xsshi2015/Loss-Attention'})

SOURCES.update({'tpmil': 'LitaoYang-Jet/TPMIL', 'ins': 'miccaiif/INS', 'cdpmil': 'HKU-MedAI/cDPMIL', 'mammoth': 'mahmoodlab/MAMMOTH'})

SOURCES.update({'hmil': 'ChengJin-git/HMIL', 'sclwc': 'Xiyue-Wang/SCL-WC'})

SOURCES.update({'msattention': 'mdsatria/MultiAttentionMIL'})
SOURCES.update({'ttis': 'QuIIL/TTIS'})
SOURCES.update({'mamba2dmil': 'AtlasAnalyticsLab/2DMamba'})
SOURCES.update({'simil': 'bmi-imaginelab/SI-MIL'})
SOURCES.update({'pathrwkv': 'Puzzle-Logic/PathRWKV'})

SOURCES.update({'c2mil': 'mimic0127/C2MIL'})

SOURCES.update({'triagemil': 'barathi-1993/TRIAGE-MIL'})
SOURCES.update({'deepgraphsurv': 'uta-smile/DeepGraphSurv'})
SOURCES.update({'codmil': 'Jiangbo-Shi/CoD-MIL'})

SOURCES.update({'dtmil': 'yfzon/DT-MIL', 'amiml':'Boomwwe/AMIML'})
SOURCES.update({'dispro':'Innse/DisPro','slip':'LTS5/SLIP'})

SOURCES.update({'augdiff':'szc19990412/AugDiff','vlsa':'liupei101/VLSA'})
SOURCES.update({'zoommil':'histocartography/zoommil'})
SOURCES.update({'hdmil': 'JiuyangDong/HDMIL', 'ahdmil': 'JiuyangDong/AHDMIL'})
SOURCES.update({'mgtrans': 'Jiangbo-Shi/MG-Trans'})

PROFILES = {
 'gmmamba_mambamil_bimamba_source': ('gmmamba', 'Complete original512D two-BiMamba16state hard-mask IMM, coordinateKMeans10 and8head CSS/class-token graph; source AdamW100max/VAL AUC+F1/epoch>50, original2readers and exact state; explicit pinned MambaMIL BiMamba dependency port; source-native512D bypassed FC1 retained', 'classification'),
 'gmmamba_fm_projection_repaired': ('gmmamba', 'Complete original512D two-BiMamba16state hard-mask IMM, coordinateKMeans10 and8head CSS/class-token graph; source AdamW100max/VAL AUC+F1/epoch>50, original2readers and exact state; explicit pinned MambaMIL BiMamba dependency port; explicit source FC1/GELU/dropout input-projection repair for multiple FM widths', 'classification'),
 'ahdmil_camelyon16_source_repaired': ('ahdmil', 'Full HDMIL DMIN preparation then actual two-expert AHDMIL joint optimization, NumPy attention/hard distillation, alternating Adams and simultaneous crossEMA, source retention-gated VAL selection/reset; explicit RCC lower_ratio and entry-name repairs', 'classification'),
 'ahdmil_nsclc_source_repaired': ('ahdmil', 'Full HDMIL DMIN preparation then actual two-expert AHDMIL joint optimization, NumPy attention/hard distillation, alternating Adams and simultaneous crossEMA, source retention-gated VAL selection/reset; explicit RCC lower_ratio and entry-name repairs', 'classification'),
 'ahdmil_brca_source_repaired': ('ahdmil', 'Full HDMIL DMIN preparation then actual two-expert AHDMIL joint optimization, NumPy attention/hard distillation, alternating Adams and simultaneous crossEMA, source retention-gated VAL selection/reset; explicit RCC lower_ratio and entry-name repairs', 'classification'),
 'ahdmil_rcc_source_repaired': ('ahdmil', 'Full HDMIL DMIN preparation then actual two-expert AHDMIL joint optimization, NumPy attention/hard distillation, alternating Adams and simultaneous crossEMA, source retention-gated VAL selection/reset; explicit RCC lower_ratio and entry-name repairs', 'classification'),
 'hdmil_camelyon16_source': ('hdmil', 'Complete released two-stage DMIN512/256/ChebyKAN/SmoothTop1/Gumbel self-distillation then native RGB16 MobileNetV4 LIPN; TRAIN weighted replacement, two Adams200+200max, source VALmeanCE stages and full-state teacher transfer', 'classification'),
 'hdmil_nsclc_source': ('hdmil', 'Complete released two-stage DMIN512/256/ChebyKAN/SmoothTop1/Gumbel self-distillation then native RGB16 MobileNetV4 LIPN; TRAIN weighted replacement, two Adams200+200max, source VALmeanCE stages and full-state teacher transfer', 'classification'),
 'hdmil_brca_source': ('hdmil', 'Complete released two-stage DMIN512/256/ChebyKAN/SmoothTop1/Gumbel self-distillation then native RGB16 MobileNetV4 LIPN; TRAIN weighted replacement, two Adams200+200max, source VALmeanCE stages and full-state teacher transfer', 'classification'),
 'hdmil_rcc_source': ('hdmil', 'Complete released two-stage DMIN512/256/ChebyKAN/SmoothTop1/Gumbel self-distillation then native RGB16 MobileNetV4 LIPN; TRAIN weighted replacement, two Adams200+200max, source VALmeanCE stages and full-state teacher transfer', 'classification'),
 'mgtrans_source_repaired': ('mgtrans', 'Complete released192width/3heads/512anchor two-scale graph Transformer, actual3layer GENConv, negativeCE semantic consistency and32slide RenyiMI window; original Adam/200maxepochs/VALerror latest-tie selection; explicit device/API/latent-query repairs', 'classification'),
 'zoommil_bright_source_repaired': ('zoommil', 'Complete original three-scale dual-gated/100sample differentiableTopK/Kronecker/sum ZoomMIL; source weighted TRAIN replacement sampler, Adam/plateau meanVALCE/100fixed epochs and weightedF1 selection; explicit saver/device ports, native8readers, cached features only', 'classification'),
 'zoommil_crc_source_repaired': ('zoommil', 'Complete original three-scale ZoomMIL with CRC K16, actual weighted sampler/Adam/plateau and100fixed epochs; missing source VALloss-saver input explicitly repaired; no two-level or onlineFM substitution', 'classification'),
 'augdiff_amil_released_time_reconstruction': ('augdiff', 'Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original amil CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and released_time_reconstruction', 'classification'),
 'augdiff_amil_schedule_repaired_reconstruction': ('augdiff', 'Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original amil CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and schedule_repaired_reconstruction', 'classification'),
 'augdiff_lossattn_released_time_reconstruction': ('augdiff', 'Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original lossattn CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and released_time_reconstruction', 'classification'),
 'augdiff_lossattn_schedule_repaired_reconstruction': ('augdiff', 'Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original lossattn CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and schedule_repaired_reconstruction', 'classification'),
 'augdiff_dsmil_released_time_reconstruction': ('augdiff', 'Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original dsmil CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and released_time_reconstruction', 'classification'),
 'augdiff_dsmil_schedule_repaired_reconstruction': ('augdiff', 'Complete actual full-width7-view DAE/EMA pretraining then frozen RAW DDIM semantic augmentation plus original dsmil CE/LookaheadRAdam/gradacc2/7xTRAIN-VAL/cleanTEST; explicit singleton1D feature-layout reconstruction and schedule_repaired_reconstruction', 'classification'),
 'vlsa_conch_source_repaired': ('vlsa', 'Full source CONCH native512 VLFAN/Text-TaskRes,4-base ordinal interpolation, IFMLE+EMD,32-patient window Adam/10epochs/final checkpoint; explicit TRAIN-only terminal time-edge repair', 'survival'),
 'slip_clip_vit_b16_source_repaired': ('slip','Complete original SLIP actual native512 CLIP-ViT-B16 text/context/tissue attention and C*C contrastive objective, source prompt-only SGD10/final-epoch with explicit unified-loader and precision/device ports','classification'),
 'slip_clip_rn50_source_repaired': ('slip','Complete original SLIP actual native1024 CLIP-RN50 text/context/tissue attention and C*C contrastive objective, source prompt-only SGD10/final-epoch with explicit unified-loader and precision/device ports','classification'),
 'amiml_paper_reference': ('amiml','Complete four TRAIN-only morphology clusters, full native8-channel minmax/decoder/residual/selfattention and explicit V8->C paper-output reconstruction, weighted-logit CE/Adam/300/plateau10/VAL-only two-level selection', 'classification'),
 'amiml_source_repaired': ('amiml','Full AMIML with explicitly repaired class/input dimensions and released probability-CE/Adam/300/plateau5 control, all four clusters and source checkpoint/early-stop semantics; missing original loader reconstructed', 'classification'),
 'dtmil_source_crop_reference_train': ('dtmil', 'complete pinned DT-MIL full spatial/deformable-encoder/ordinary-decoder/query-head, released crop/binary contract with explicit finite-padding port; complete named50epoch AdamW/LR-group/StepLR reference training because public trainer is absent', 'classification'),
 'dtmil_repaired_reference_train': ('dtmil', 'complete pinned DT-MIL and physical cached-view/data augmentation recipe; explicit current-height crop/C-way labels/finite-padding ports, full reference AdamW/LR-groups/StepLR/clipping/firstVAL and exact resume', 'classification'),
 'codmil_source_repaired': ('codmil', 'complete author two-scale chain-of-diagnosis and masked-normal-text objective, physical row-map/text binding, explicit C-way/full-D/T input ports and complete original one-bag Adam/VAL-loss/stopping recipe with exact state resume', 'classification'),
 'ins_frozen_source_repaired': ('ins', 'complete frozen-feature INS route with actual native ResNet18 feature-pseudoimage/query-key/prototype/queue/partial-label/SupCon kernels and full source two-phase SGD+bag-CE recipe; named weak-label, valid-queue, two-class bag-head and scratch-initialization repairs', 'classification'),
 'sclwc_paper_reference': ('sclwc', 'complete explicitly reconstructed paper CDA/PNM/WSCL; native512/256 class-specific attention, active live-anchor cross-slide objective and real TRAIN-only typed banks; full balanced one-bag Adam/cosine/validation stopping and exact resume', 'classification'),
 'sclwc_literal_source_control': ('sclwc', 'literal pinned source model and actual original train_loop, full-D input port; preserves missing PNM and disabled WSCL as an explicitly named incomplete-source control; complete author balanced Adam/cosine/selection loop', 'classification'),
 'deepgraphsurv_agcn': ('deepgraphsurv', 'literal2018 adaptive feature graph attention32/1 and native256/256/128AGCN with complete original mixed-event window trainer, TF1Adam, update-count LR/BN schedules and full checkpoint; full-dimensional frozen-FM input condition and explicit VAL-only selection/inference stream corrections', 'survival'),
 'triagemil_source': ('triagemil', 'complete original MASS quality/four-axis/FPS selection, semantic hierarchical hypergraphs and native full-D six-head/four-layer model; source Cox/AdamW/accumulation/per-epoch AMP/VAL-first-tie/minimum-epoch stopping with exact resume', 'survival'),
 'c2mil_source': ('c2mil', 'full original adaptive TRAIN-reference bias removal, causal graph three-view network, five losses and native batch/Adam/VAL-selection recipe; complete dynamic cache and exact epoch resume', 'survival'),
 'gatmamba_source': ('gatmamba', 'complete original physical GAT/Mamba1.2.0.post1 network, source Cox+all-parameterL1 and full shuffled-patient Adam/plateau/last-tie selection recipe; explicit full-D input port and original inactive edge conditioning', 'survival'),
 'gatmamba_edge_conditioned': ('gatmamba', 'complete native GAT-Mamba recipe with explicit edge_dim16 and singleton normalization corrections; actual physical/subtype/FM-space graph provenance, all registered feature channels', 'survival'),
 'protosurv_source': ('protosurv', 'complete original dual-view graph/prototype network, three objectives and full source survival sampler/augmentation/gc/Adam/cosine/selection; original supported category/time domain', 'survival'),
 'protosurv_repaired': ('protosurv', 'complete ProtoSurv recipe with named full-vocabulary compatibility, local pad RNG, tied-time Breslow and partial-tail corrections; actual graph/category/prototype provenance', 'survival'),

 'pathrwkv_source': ('pathrwkv', 'complete pinned TimeMix/ChannelMix/2D position/MTL model and heterogeneous CE/Huber/source-Cox, source sampling/AdamW/cosine/early stop; original chunk shift resets and Cox broadcasting retained explicitly', 'multitask'),
 'pathrwkv_repaired': ('pathrwkv', 'complete native PathRWKV heterogeneous recipe with explicit observed masks, Breslow-Cox repair and bounded GPU streaming carrying both token shifts and WKV state; actual author CUDA backend or explicitly selected CPU tensor reference', 'multitask'),
 'c2aug_transmil_paper_reference': ('c2aug', 'complete named paper-reference recipe: all three TRAIN-only cross-bag augmentations, original TransMIL, EMA teacher, active bag FIFO InfoNCE, learned prototypes and all-group forward KL; explicit paper/source differences', 'classification'),
 'agdrmil_yaml_source': ('agdrmil', 'complete released YAML hard grading/cosine/low-mask/high+full Sigmoid-CE recipe, native Adam and source shuffle/validation RNG; no invented grader gradient', 'classification'),
 'agdrmil_tcga_source': ('agdrmil', 'complete source model/trainer with explicit README TCGA thresholds/mask/dual-loss coefficient; hard-gradient and stochastic-eval source behavior retained', 'classification'),
 'h2mil_source_compat': ('h2mil', 'complete original two-stage RAConv/IHPool and summed-batch Adam recipe; explicit shape repair, literal four-probability CE and binary metric path', 'classification'),
 'h2mil_repaired': ('h2mil', 'complete original hierarchy/model/optimizer recipe with explicit shape, C-way head and logits-CE repair; hard pooling gradient boundary retained', 'classification'),
 'hipt_lgp_frozen_source': ('hipt', 'complete original local-global pretrained HIPT classifier, actual frozen teacher, balanced sampler and source CE/32-per-bag Adam; default final epoch', 'classification'),
 'hipt_lgp_finetune_source': ('hipt', 'complete original pretrained HIPT classifier with local ViT finetuning, native physical pyramids and balanced source CE/32-per-bag Adam', 'classification'),
 'simil_source': ('simil', 'complete original dual-branch SI-MIL and coupled BCE/KD/attention recipe; literal published TopK derivative, TRAIN-only PathExpert statistics', 'classification'),
 'simil_full_vjp': ('simil', 'complete original SI-MIL model and optimizer with explicitly corrected full Gaussian TopK VJP, TRAIN-only PathExpert statistics', 'classification'),
 'mamba2dmil_source': ('mamba2dmil', 'complete native physical 2D source scan, literal CPU backward; original CE/AdamW/cosine and discrete NLL-per-patient recipe', 'classification'),
 'mamba2dmil_autograd': ('mamba2dmil', 'complete native 2D model and recipes, exact published CPU forward with explicitly corrected automatic derivatives', 'classification'),
 'mamba2dmil_paper': ('mamba2dmil', 'complete native model/recipes, explicit vertical downsweep and derivative corrections independently checked against primary horizontal/vertical equations', 'classification'),
 **{f'ttis_{head}_source': ('ttis', 'original TTIS six-head source CE/LookaheadAdamW/cosine recipe, complete TEST-only physical/semantic FPS and feature-level fusion', 'classification')
    for head in ['abmil', 'clam', 'rrt', 's4', 'transmil', 'wikg']},
 'ttis_clam_full_loss': ('ttis', 'complete original CLAM combined bag/instance objective as explicitly named training repair, followed by original TTIS selection/fusion', 'classification'),
 'msattention_source': ('msattention', 'original three-depth residual attention, full CE/Adam supplementary recipe; explicit frozen-WSI transfer with controlled stochastic eval dropout', 'classification'),
 'msattention_repaired': ('msattention', 'complete original residual attention/CE/Adam recipe with named mode-aware eval dropout repair and independent validation selection', 'classification'),
 'pmil_shapley_abmil_source': ('pmil_shapley', 'native Shapley marginal logits, progressive pseudo bags and fresh multi-round ABMIL, original gradient carry and dual validation selection', 'classification'),
 'pmil_shapley_abmil_repaired': ('pmil_shapley', 'complete PMIL Shapley/plateau/progressive/multi-round recipe with named per-pseudo-gradient reset and constant-score repair', 'classification'),
 'pmil_shapley_clam_sb_source': ('pmil_shapley', 'complete Shapley/progressive/EM-round recipe with actual author CLAM-SB instance constraints and original pseudo-gradient carry', 'classification'),
 'pmil_shapley_clam_sb_repaired': ('pmil_shapley', 'full PMIL CLAM-SB recipe including instance CE, named gradient and constant-score repairs, native round/plateau policy', 'classification'),
 'pmil_shapley_clam_mb_source': ('pmil_shapley', 'complete author CLAM-MB/Shapley recipe with source branch0 candidate search, gradient carry and full multi-round training', 'classification'),
 'pmil_shapley_clam_mb_repaired': ('pmil_shapley', 'full PMIL CLAM-MB with explicit target-branch ranking, per-pseudo-gradient reset and constant-score repairs', 'classification'),
 'mammoth_mean': ('mammoth', 'native multi-head slot routing/shared low-rank experts, feature mean, full weighted AdamW/cosine classification recipe', 'classification'),
 'mammoth_max': ('mammoth', 'native MAMMOTH experts and paper single-winning-slot logits, full weighted AdamW/cosine recipe', 'classification'),
 'mammoth_abmil': ('mammoth', 'native MAMMOTH and author gated attention/classifier, full weighted AdamW/cosine/CE recipe', 'classification'),
 'mammoth_clam': ('mammoth', 'native MAMMOTH and author CLAM instance constraints, binary SmoothTop1 and complete weighted AdamW/cosine recipe', 'classification'),
 'mammoth_dsmil': ('mammoth', 'native MAMMOTH and author critical-instance dual stream, source averaged-logit CE and full weighted AdamW/cosine', 'classification'),
 'mammoth_transmil': ('mammoth', 'native MAMMOTH and author Nystrom/PPEG TransMIL, complete weighted AdamW/cosine/CE recipe', 'classification'),
 'mammoth_transformer': ('mammoth', 'native MAMMOTH and author CLS global transformer/MLP, complete weighted AdamW/cosine/CE recipe', 'classification'),
 'mammoth_ilra_repaired': ('mammoth', 'native MAMMOTH and author ILRA/GAB/NLP; named routing-width128 repair of invalid256/256 YAML, full weighted AdamW/cosine', 'classification'),
 'campanella_rnn': ('campanella_rnn', 'complete trainable ResNet34 hard-MIL selection, validated best CNN, native recurrent final-step CE and source Adam/SGD two-stage recipe on real streamed ROIs', 'classification'),
 'monai_mean_tutorial': ('monai_mil', 'native trainable ResNet50 mean-logit baseline; cumulative ordinal BCE, complete AdamW/cosine RGB tutorial', 'ordinal'),
 'monai_max_tutorial': ('monai_mil', 'native trainable ResNet50 max-logit baseline; cumulative ordinal BCE, complete AdamW/cosine RGB tutorial', 'ordinal'),
 'monai_attention_tutorial': ('monai_mil', 'native trainable ResNet50 attention baseline; cumulative ordinal BCE, complete AdamW/cosine RGB tutorial', 'ordinal'),
 'monai_transformer_tutorial': ('monai_mil', 'full native trainable ResNet50 plus dependency Transformer; ordinal BCE, two AdamW groups and epoch cosine, raw full-level stochastic tiling', 'ordinal'),
 'monai_pyramid_tutorial': ('monai_mil', 'complete native ResNet50 four-stage feature-map pyramid Transformer, ordinal BCE and full two-group AdamW/cosine RGB tutorial', 'ordinal'),

 'stablemil_classification_source': ('stablemil', 'full physical fusion/semantic/local-global/RoPE network and Adam/CE; source cross-epoch accumulation tails, named execution wiring fixes', 'classification'),
 'stablemil_classification_repaired': ('stablemil', 'complete classifier recipe with explicit degenerate/small-bag and tail accumulation repairs', 'classification'),
 'stablemil_survival_source': ('stablemil', 'full hazard network and original clamped NLL/gc/Adam loop with explicit missing launcher/model wiring completion and TRAIN-only bins', 'survival'),
 'stablemil_survival_repaired': ('stablemil', 'complete StableMIL hazard recipe with explicit physical small-bag and tail weighting repairs, patient NLL selection', 'survival'),
 'setmil_source': ('setmil', 'full author physical PPE/TPMF/iRPE, CE/AdamW/StepLR and spatial augmentation; original stale-height crop and all-subset cached views retained', 'classification'),
 'setmil_repaired': ('setmil', 'complete SETMIL recipe, explicit corrected center crop and TRAIN-only cached views; coherent checkpoints and sealed test evaluation', 'classification'),
 'gtp_source': ('gtp', 'author GCN/BN, mincut/orthogonality and 3-blockViT, full Adam/MultiStepLR and batch8/drop_last; source graph/BN conditions retained, explicit frozen-FM input', 'classification'),
 'gtp_repaired': ('gtp', 'full author GTP recipe and true gradient/LRP GraphCAM; explicit zero-edge/singleton-BN/constant-map repairs with otherwise identical source path', 'classification'),
 'probsa_abmil_source': ('probsamil', 'author-library Gaussian attention MC-BCE plus graph KL, cyclical coefficient and Adam/LinearLR; source mean-logit inference; padded sparse KL repaired', 'classification'),
 'probsa_abmil_probmean': ('probsamil', 'author-library full ABMIL ProbSA recipe; paper Eq19 mean-probability inference, sparse/padding KL repairs and fixed query random stream', 'classification'),
 'probsa_transformer_source': ('probsamil', 'author-library two-layer global transformer plus sampled Gaussian attention/graph KL, cyclical Adam recipe; source mean-logit inference', 'classification'),
 'probsa_transformer_probmean': ('probsamil', 'full global-transformer ProbSA author-library recipe; paper Eq19 predictive mixture and explicit sparse/padding/SDPA repairs', 'classification'),
 'micromil_frozen': ('micromil', 'explicit frozen-FM WSI transfer; author DCE, hard-Gumbel representatives and cosine graph, original single-head GAT, full source BCE/Adam and validation F1', 'classification'),
 'loss_attention_classical': ('loss_attention', 'paper A2/A4 classical feature MLP, author shared class/attention head, bag CE plus .1 weighted-instance CE sum, full Adam; explicit WSI feature adaptation and validation-only error/CE selection', 'classification'),
 'loss_attention_mil_source': ('loss_attention', 'MNIST source-head WSI adaptation, shared classifier and bag CE plus 2 weighted-instance CE mean, complete source Adam recipe; validation replaces source test-based selection', 'classification'),
 'loss_attention_temporal': ('loss_attention', 'image-temporal recipe explicitly adapted to WSI bags: shared head, weighted-instance CE mean, aligned TRAIN EMA consistency, original schedules and legacy-Adam state reset every epoch', 'classification'),
 'pamil_prototype_source': ('pamil_prototype', 'author dual branches, source CE-on-softmax and one-center collate clustering, warm/joint/push/last; train instance annotations; explicit bounded atomic push repairs', 'classification'),
 'pamil_prototype_repaired': ('pamil_prototype', 'author full staged recipe with explicit CE-on-logits, local KMeans and class-specific instance mining repairs; train-only extra instance supervision', 'classification'),
 'pamil_prototype_multilabel': ('pamil_prototype', 'author independent sigmoid/BCE route, two-level KMeans and complete warm/joint/annotation-filtered push/last; explicit instance-index/device repairs', 'multilabel'),

'smmile': ('smmile', 'author spatial conv/detection, SP sampling and score dropout, validation-best two-stage refinement/MRF with reset Adam and source validation objective; explicit numerical repairs', 'classification'),
'smmile_binary': ('smmile', 'author one-positive-class binary branch, native basic-plus-augmentation BCE, optional negative-bag consistency, two-stage pseudo-label refinement and label-free instance outputs', 'classification'),
'smmile_multilabel': ('smmile', 'author independent multilabel sigmoid route, original bi-tempered augmentation, combination-balanced sampling, two-stage refinement/MRF and source validation objective', 'multilabel'),

'momil': ('momil', 'author three-order Mamba2, literal channel reverse and padded-token fusion; explicit constructor-name repair and complete CE/Adam source last-or-validation checkpoint policy', 'classification'),
'momil_survival_extension': ('momil', 'explicit author survival-head and generic source NLL trainer composition; train-only intervals and flushed tail accumulation, not a published working MoMIL survival factory', 'survival'),
'recomil_single': ('recomil', 'author single-scale contextual prototypes, MLP reduction, rare gate, orthogonal loss and full warmup/tau/freeze Adam recipe; train-only clustering', 'classification'),
'recomil_graph': ('recomil', 'author single-scale contextual prototypes and graph prototype reduction with complete source recipe', 'classification'),
'recomil_dual': ('recomil', 'author dual-scale MLP prototype branches and learned sigmoid fusion, full separate branch schedules and objectives', 'classification'),
'recomil_dual_graph': ('recomil', 'author dual-scale graph prototype reduction and learned fusion with complete source recipe', 'classification'),
'mammil': ('mammil', 'author Mamba2 topology scans and GENConv, explicit full-layer-loop repair, frozen HNSW/MST/DFS/BFS graphs, source RAdam and CE/NLL', 'classification'),
'mammil_source': ('mammil', 'literal published zip(layers,ppeg) execution; original single-slide graph and RAdam CE/NLL, inactive final scan explicitly retained', 'classification'),
'mammil_patient_union': ('mammil', 'explicit corrected-loop patient survival extension using disjoint physical slide graphs and concatenated traversals; source NLL/RAdam', 'survival'),
'mfemil_mean': ('mfemil', 'author shared adapter, window-masked normalized-feature reconstruction and MeanMIL; joint CE/NLL and Adam', 'classification'),
'mfemil_max': ('mfemil', 'author shared adapter, MFE decoder and MaxMIL; joint CE/NLL and Adam', 'classification'),
'mfemil_abmil': ('mfemil', 'author shared adapter, MFE decoder and DAttention; joint CE/NLL and Adam', 'classification'),
'mfemil_clam_sb': ('mfemil', 'released MFE-CLAM-SB bag-only head with shared adapter and reconstruction; instance loss not used by source MFE recipe', 'classification'),
'mfemil_clam_mb': ('mfemil', 'released MFE-CLAM-MB bag-only head with shared adapter and reconstruction; instance loss not used by source MFE recipe', 'classification'),
'mfemil_transmil': ('mfemil', 'released survival MFE-TransMIL, shared adapter and reconstruction, censored NLL, train-only intervals and Adam', 'survival'),
'msmmil': ('msmmil', 'author AFWMamba three scan branches with shared BiAttn; CE, source RAdam/Lookahead, two-bag accumulation and validation-loss selection', 'classification'),
'mambamil_srmamba': ('mambamil', 'author SR-Mamba transpose-padded two-branch scan; explicit torch/CUDA backend, full bag CE or discrete NLL with training-only bins', 'classification'),
'mambamil_bimamba': ('mambamil', 'author Bi-Mamba forward/backward scan; explicit backend, full bag CE or discrete NLL with patient accumulation', 'classification'),
'mambamil_unidirectional': ('mambamil', 'author unidirectional Mamba ablation; explicit backend, full bag CE or discrete NLL with patient accumulation', 'classification'),
'dgmil': ('dgmil', 'author dynamic extreme-instance projection and negative-cluster Mahalanobis; stable unique ties, train-only inference density, raw-score AUROC', 'classification'),
'dgmil_source_extremes': ('dgmil', 'released heapq first-match extreme selection and full-batch Adam; train-only inference density repair, raw-score AUROC', 'classification'),
'pamoe_transmil': ('pamoe', 'author TransMIL with explicit batch/expert indexing repair; tissue-prior CE, author batch Cox or classification CE, event sampler, AdamW/cyclic cosine', 'classification'),
'pamoe_transmil_source': ('pamoe', 'literal released TransMIL expert slicing; tissue-prior CE, author batch Cox or classification CE, event sampler, AdamW/cyclic cosine', 'classification'),
'pamoe_transformer': ('pamoe', 'author Transformer with explicit batch/expert indexing repair; tissue-prior CE, author batch Cox or classification CE, event sampler, AdamW/cyclic cosine', 'classification'),
'pamoe_transformer_source': ('pamoe', 'literal released Transformer expert slicing; tissue-prior CE, author batch Cox or classification CE, event sampler, AdamW/cyclic cosine', 'classification'),

'pathomics_concat_source': ('pathomics', 'author concat branch, two-stage MSE alignment and censored NLL, two Adam optimizers, explicit source freeze and checkpoint/tail repairs', 'survival'),
'pathomics_image_source': ('pathomics', 'author image branch, two-stage MSE alignment and censored NLL, two Adam optimizers, explicit source freeze and checkpoint/tail repairs', 'survival'),
'pathomics_omics_source': ('pathomics', 'author omics branch, two-stage MSE alignment and censored NLL, two Adam optimizers, explicit source freeze and checkpoint/tail repairs', 'survival'),

'mmp_panther_source': ('mmp', 'author PANTHER prototypes with full-attention fusion, train-only KMeans, pathway SNN, batch Cox/NLL, AdamW warmup cosine and explicit source repairs', 'survival'),
'mmp_panther_uot_source': ('mmp', 'author PANTHER prototypes with differentiable UOT fusion, train-only KMeans, pathway SNN, batch Cox/NLL, AdamW warmup cosine and explicit source repairs', 'survival'),
'mmp_ot_source': ('mmp', 'author OT prototypes with full-attention fusion, train-only KMeans, pathway SNN, batch Cox/NLL, AdamW warmup cosine and explicit source repairs', 'survival'),
'mmp_ot_uot_source': ('mmp', 'author OT prototypes with differentiable UOT fusion, train-only KMeans, pathway SNN, batch Cox/NLL, AdamW warmup cosine and explicit source repairs', 'survival'),
'pibd_source': ('pibd', 'pinned author prototype bottlenecks, PoE, disentangled transformer, joint CLUB fitting, weighted patient minibatches and complete survival/proxy/KL/MI objective', 'survival'),
'advmil_esat': ('advmil', 'pinned author continuous ESAT + RLIP discriminator; two Adam optimizers, censored reconstruction, noise inference and explicit label masking; corrected discriminator BCE', 'survival'),
'advmil_abmil': ('advmil', 'pinned author continuous ABMIL + RLIP discriminator; two Adam optimizers, censored reconstruction, noise inference and explicit label masking; corrected discriminator BCE', 'survival'),
'advmil_esat_source': ('advmil', 'pinned author continuous ESAT + RLIP discriminator; two Adam optimizers, censored reconstruction, noise inference and explicit label masking; literal released discriminator BCE', 'survival'),
'advmil_abmil_source': ('advmil', 'pinned author continuous ABMIL + RLIP discriminator; two Adam optimizers, censored reconstruction, noise inference and explicit label masking; literal released discriminator BCE', 'survival'),

 'hmil_source_repaired': ('hmil', 'author hierarchical coarse/fine attention, SupCon, source curriculum and bag alignment; explicit scalar-head repair from prior author commit', 'classification'),
 'tpmil': ('tpmil', 'author trainable K+1 prototypes, detached attention pseudo-targets, .5 CE + .5 KL and balanced sampling; explicit constant-attention repair', 'classification'),
 'cdpmil_source': ('cdpmil', 'public source path: per-slide Bayesian DP centroids and sampled Gaussian class log densities, CE and Adam/cosine; no variational gamma updates in released trainer', 'classification'),
'rankmix_dsmil': ('rankmix', 'author binary DSMIL, supervised initialization, natural/balanced sampling and dual BCE; repaired ranked patch IDs', 'classification'),
'rankmix_dsmil_source': ('rankmix', 'binary DSMIL RankMix preserving literal author argsort-position selection; explicit source-behavior comparator', 'classification'),
'rankmix_frmil': ('rankmix', 'author binary FRMIL, paired supervised initialization, mixed-bag BCE plus original-pair magnitude; repaired IDs and padded selection', 'classification'),
'rankmix_frmil_source': ('rankmix', 'binary FRMIL RankMix preserving literal author argsort-position/padding behavior; explicit source comparator', 'classification'),
'cate': ('cate', 'author binary CATE-MIL; concept-guided IB, top-10 InfoNCE, KL and detached CFI; explicit weighted objective', 'classification'),
'vilamil': ('vilamil', 'author dual-scale prototype/context decoders and learned CLIP prompts; corrected EOT position, frozen text encoder per paper, CE', 'classification'),
'vilamil_source': ('vilamil', 'literal public ViLa prompt EOT and trainable text encoder; dual-scale decoders and CE; separately named source behavior', 'classification'),
'he2rna': ('he2rna', 'author tile Conv1D network; spatial supertiles, shared random top-k, train-mean bias, MSE regression and project/gene Pearson selection', 'regression'),
'healnet': ('healnet', 'author shared latent multimodal fusion, weighted censored NLL, L1 and per-batch Adam/OneCycle; patient inputs and train-only preprocessing', 'survival'),
'camil_channel': ('camil_channel', 'author four Nystrom/MCAB blocks and gated pooling; CE, author RAdam/Lookahead and two-bag gradient accumulation', 'classification'),
'camil_context': ('camil_context', 'TensorFlow equation port: spatial-cosine neighbor graph, literal Nystrom axes, squared-gate global/local fusion; Keras Adam and validation-loss plateau', 'classification'),
'dyhg': ('dyhg', 'author Gumbel soft hypergraph incidence, node-edge-node messages and gated pooling; CE, Adam and isolated stochastic inference', 'classification'),
'mfcmil_abmil': ('mfcmil', 'author ABMIL, multiresolution filters, frequency projections and high/low memory interventions; CE and step warmup/cosine', 'classification'),
'mfcmil_transmil': ('mfcmil', 'author Nystrom TransMIL with frequency, multiresolution and memory intervention; CE and step warmup/cosine', 'classification'),
'sdfp_abmil': ('sdfp', 'author S-DFP with raw-feature attention; shared prompt then slide-aware joint training, CE and step cosine', 'classification'),
'sdfp_gated': ('sdfp', 'author S-DFP with raw-feature gated attention; shared prompt then slide-aware joint training, CE and step cosine', 'classification'),
'sdfp_dsmil': ('sdfp', 'author S-DFP/DSMIL; two-stage prompts plus all-instance CE and bag CE, joint Adam and step cosine', 'classification'),
'sdfp_transmil': ('sdfp', 'author S-DFP/Nystrom TransMIL; two-stage joint prompting, CE and step cosine', 'classification'),
'mil_dropout_abmil': ('mil_dropout', 'author non-gated ABMIL with three MIL-Dropout layers; bag CE, AdamW and checkpointed Lookahead', 'classification'),
'mil_dropout_gated': ('mil_dropout', 'author gated ABMIL with three MIL-Dropout layers; bag CE, AdamW and checkpointed Lookahead', 'classification'),
'mil_dropout_dsmil': ('mil_dropout', 'author MIL-Dropout DSMIL bag output; CE, AdamW and epoch cosine; critical selection remains discrete', 'classification'),
'mil_dropout_transmil': ('mil_dropout', 'author MIL-Dropout TransMIL/Nystrom/PPEG; CE, RAdam and checkpointed Lookahead', 'classification'),
'packmil_abmil': ('packmil', 'author sequence packs, residual hyperslides and ADS; CE plus focal auxiliary objective, Adam/cosine', 'classification'),
'packmil_survival': ('packmil', 'author ABMIL packs/ADS and hyperslide censor targets; train-only patient event bins, main and mixed censored NLL', 'survival'),
'scmil': ('scmil', 'author SoftFilter and sparse context attention; continuous mixture censor likelihood; explicit CPU clustering and patient slide-mixture adaptation', 'survival'),
'deepattnmisl': ('deepattnmisl', 'author phenotype Conv/pooling/attention; batch patient-mean Cox plus L1; explicit patient KMeans or provided labels', 'survival'),
'mico': ('mico', 'author cluster routes/reducers; train-only FAISS initialization; CE; author Adam decay grouping', 'classification'),
'mico_survival': ('mico', 'author per-slide context and patient joint pooling; train-only FAISS centers and event bins; discrete NLL', 'survival'),
'efficientmil_gru': ('efficientmil', 'author APS plus bidirectional GRU; dual bag/max-instance BCE, L2, Adam and cosine schedule', 'classification'),
'efficientmil_lstm': ('efficientmil', 'author APS plus bidirectional LSTM; dual bag/max-instance BCE, L2, Adam and cosine schedule', 'classification'),
'bayesmil_vis': ('bayesmil', 'author variational weights and logistic-normal attention; CE + weight regularizer; MC mean probability prediction', 'classification'),
'bayesmil_sdpr': ('bayesmil', 'Bayes-MIL with literal author class-dependent patch regularizer, checkpointed plateau schedule and MC inference', 'classification'),
'bayesmil_apcrf': ('bayesmil', 'full spatial APCRF plus SDPR, real canvas and corrected raster stride; author regularizer and MC probability average', 'classification'),
'caprmil_mean': ('caprmil', 'author soft context tokens, token attention and de-slicing; mean pooling, CE and AdamW/cosine', 'classification'),
'caprmil_attention': ('caprmil', 'author CAPRMIL context blocks with learned attention pooling; CE and AdamW/cosine', 'classification'),
'caprmil_gated': ('caprmil', 'author CAPRMIL context blocks with gated attention pooling; CE and AdamW/cosine', 'classification'),
'varmil': ('varmil', 'author raw-feature attention-weighted mean/variance concatenation and CE; frozen-feature VarMIL, not SimCLR pretraining', 'classification'),
'smmil_early': ('smmil', 'author early feature smoothing on spatial adjacency, BCE and epoch warmup', 'classification'),
'smmil_mid': ('smmil', 'author mid attention-projection smoothing on spatial adjacency, BCE and epoch warmup', 'classification'),
'smmil_late': ('smmil', 'author late attention-logit smoothing on spatial adjacency, BCE and epoch warmup', 'classification'),
'smmil_transformer': ('smmil', 'author SmTransformer plus early Sm attention pool, BCE and epoch warmup', 'classification'),
'psamil': ('psamil', '2026 author spatial decay attention, active head-diversity objective, class balancing and separate decay learning rate', 'classification'),
'slidegraph': ('slidegraph', 'author hierarchical feature-spatial clustering, EdgeConv, binary pair ranking and train-only Platt scaling; single validation checkpoint', 'classification'),
"csmil": ("csmil", "author raw cross-scale scoring, masked bag attention, per-slide fine-scale KMeans and cluster sampling; binary NLL with summed accumulation", "classification"),
"hagmil_ce": ("hagmil", "author three-stage four-layer IAM models, detached hierarchical selection, CE instance clustering and per-level optimizer updates; explicit CE profile", "classification"),
"cmta": ("cmta", "pinned author cross-modal encoders, translators and decoders; discrete NLL + detached-target path/omics L1 alignment", "survival"),
"motcat": ("motcat", "pinned author OT co-attention, dual Transformers and fusion; full-bag discrete NLL; explicit entropy UOT regularizer", "survival"),
"motcat_micro": ("motcat", "author shuffled micro-bag partition, mean per-micro NLL and mean risk; streamed backward; fixed-seed isolated inference", "survival"),
"cbmil": ("cbmil", "author adaptive cluster selection, patch/phenotype fusion, CLAM instance loss, two-view NT-Xent and asymmetric feature mixing; explicit FM-to-512 adapter", "classification"),
"sgpmil": ("sgpmil", "author sparse variational GP attention; Monte Carlo mean logits CE + inducing KL / training bags; mean probability prediction; AdamW", "classification"),
"mmil": ("mmil", "author multi-level group attention and message-token communication; random group masking; CE; fixed-seed unmasked inference", "classification"),
"mmil_coords": ("mmil", "author physical-coordinate KMeans grouping, group masking and message-token communication; CE", "classification"),
"ptcmil": ("ptcmil", "author prompt clustering, local Nystrom attention and learned token merging; CE + PTC orthogonality; checkpointed within-epoch moving prompts", "classification"),
"survpath": ("survpath", "pinned author pathway/histology multimodal attention; train-only molecular normalization and event-time bins; censored discrete NLL", "survival"),
"mcat": ("mcat", "pinned author six-signature co-attention, dual Transformers and fusion; train-only normalization/bins; discrete NLL", "survival"),
"porpoise": ("porpoise", "pinned author gated WSI attention + omics SNN + gated bilinear fusion; train-only normalization/bins; discrete NLL", "survival"),
"patchgcn": ("patchgcn", "pinned author residual GENConv and concatenated depths; exact physical kNN within slides; patient disjoint union; discrete NLL", "survival"),
"psemix_abmil": ("psemix", "ProtoDiv phenotype refinement, fixed checkpointed partitions, pseudo-bag mixing and content-weighted soft BCE; PULS ABMIL base", "classification"),
"prodiv_abmil": ("prodiv", "ProtoDiv pseudo-bag sampling via PseMix authors implementation; PULS ABMIL base and BCE", "classification"),
"remix_abmil": ("remix", "train-bag spherical KMeans and covariance shifts; same-class joint replacement/append/interpolation/shift; author ReLU ABMIL + BCE", "classification"),
"frmil": ("frmil", "pinned MICCAI model; binary class-paired episodes, train-maximum padding, bag CE + max-instance BCE + feature magnitude", "classification"),
"focusmil": ("focusmil", "binary variational max pooling; reparameterized latent and active KL; deterministic posterior-mean inference", "classification"),
"dgrmil": ("dgrmil", "pinned author model; class-conditional EMA centers, warmup, one-hot BCE, log-determinant diversity and triplet alignment", "classification"),
"rrt": ("rrt", "pinned author RRT-MIL: regional MSA, cross-region MSA and EPEG; CE; current refactored source", "classification"),
'chowder': ('chowder', 'Owkin later implementation adaptation: classwise extreme scores and shared sigmoid MLP', 'classification'),
'varpool': ('varpool', 'projected attention variance pooling with mean branch and CE classification head', 'classification'),
'varpool_cox': ('varpool', 'projected attention variance pooling with full-cohort Cox objective', 'survival'),
'h2t': ('h2t', 'train-only prototype mining, normalized per-cluster means, linear classifier (PANTHER authors H2T comparator)', 'classification'),
'panther': ('panther', 'train-only prototype mining, diagonal NIW MAP-EM allcat representation and linear classifier', 'classification'),
'ibmil': ('ibmil', 'supervised warmup, training-bag spherical clustering, interventional attention, CE; sklearn kmeans backend', 'classification'),

'additive_mil': ('additive_mil', 'attention-weighted instance prediction then exact additive class contributions', "classification"),
'linmil': ('linmil', 'ReLU-kernel linear attention, CLS pooling, CE', "classification"),
'fouriermil': ('fouriermil', 'adaptive Fourier filtering blocks and CLS pooling, CE', "classification"),
'retmil': ('retmil', 'local chunk retention, global retention and gated pooling, CE', "classification"),
'pneamil': ('pneamil', 'class-specific nonnegative evidence, max aggregation and active L1 sparsity', "classification"),
'mrblock': ('mrblock', 'gated attention with frozen affine maps and nonlinear low-rank residuals', "classification"),
'carmil': ('carmil', 'factual CE + counterfactual effect CE + attention regularization', "classification"),
'fcmil': ('fcmil', 'spatial/frequency attention + drop/replace causal regularization', "classification"),
'asmil': ('asmil', 'bag CE + normalized-sigmoid EMA anchor consistency; repaired EMA updates', "classification"),
'attrimil': ('attrimil', 'frozen-feature variant: attribution pooling + physical-neighbour spatial constraint + checkpointed cross-slide ranking queues', "classification"),
'nnmil': ('nnmil', 'classification recipe: train-only bag-size planning, balanced batches, feature subspaces, masked padding and probability ensemble', "classification"),
'madmil': ('madmil', 'split-feature independent gated multihead attention with concatenated pooled heads', "classification"),
'litemil': ('litemil', 'author cross-attention queries; validation-only model selection; no test-weighted fold ensemble', "classification"),

"mhim_abmil": ("mhim", "v1 attention equations; explicit supervised warm-start, teacher EMA, high/low/random masking and feature distillation", "classification"),
"ilra": ("ilra", "low-rank compression/expansion, nonlocal pooling; CE; one pooling seed", "classification"),
"wikg": ("wikg", "chunked feature-derived top-k graph, knowledge attention and bi-interaction", "classification"),
"acmil_ga": ("acmil", "official GA equations: multi-branch supervision, stochastic top-k masking and diversity", "classification"),
"aem_abmil": ("aem", "ABMIL + full-bag negative attention entropy regularization", "classification"),
"mean_linear": ("mean_mil", "raw-feature mean + train-only standardization + logistic regression", "classification"),
"mean_mil": ("mean_mil", "nonlinear instance logits, mean pooling", "classification"),
"max_mil": ("max_mil", "nonlinear instance logits, classwise max pooling", "classification"),
"abmil": ("abmil", "WSI feature adaptation, non-gated attention, CE", "classification"),
"gated_abmil": ("abmil", "WSI feature adaptation, gated attention, CE", "classification"),
"clam_sb": ("clam", "vendored official model, CE instance clustering + bag loss", "classification"),
"clam_mb": ("clam", "vendored official multibranch model, instance clustering + bag loss", "classification"),
"dsmil": ("dsmil", "dual-stream BCE; sigmoid stream ensemble normalized for exclusive classes", "classification"),
"transmil": ("transmil", "MIL_BASELINE Nystrom + PPEG adaptation; CE", "classification"),
"dtfd": ("dtfd", "two optimizers, group CE, AFS/MaxS/MaxMinS distillation; deterministic evaluation grouping", "classification"),
"abmil_cox": ("abmil", "PULS adaptation: full-cohort Cox Breslow, patient bags", "survival"),
"gated_abmil_cox": ("abmil", "PULS adaptation: full-cohort Cox Breslow, gated attention", "survival"),
}


# Complete public-reference catalog metadata, preserved from the validated runtime registry.
PRESERVED_REFERENCE_PROFILE_METADATA = {'pamil_policy_paper_reference': {'id': 'pamil_policy_paper_reference', 'family': 'pamil_policy', 'recipe': 'Complete explicit paper-reference native512 SFFR/fused history/TCM, LIIS topM or GMSS/GHSS, WSL/STL/stopgradSIA/actualTRAINreward/PPO/AdaMax300/fullstate', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'single_slide_bag', 'explicit_feature_row_order'], 'class_count': None, 'external_source': 'pamil_policy', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}, 'pamil_policy_released_sampling_reference_training': {'id': 'pamil_policy_released_sampling_reference_training', 'family': 'pamil_policy', 'recipe': 'Actual released LIIS weighted replacement/padding/floorT/retainfirst/rawhistory control, with complete named reconstructed paper training/objectives/reward/PPO', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'single_slide_bag', 'explicit_feature_row_order'], 'class_count': None, 'external_source': 'pamil_policy', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}, 'focus_source': {'id': 'focus_source', 'family': 'focus', 'recipe': 'Original FOCUS native full trainable CONCH text/prompt, complete visual selection/compression/8head attention, one-bag CE, VAL-error firstbest', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'single_slide_bag', 'focus_assets'], 'class_count': None, 'external_source': 'focus', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run'}}
PROFILES.update({'pamil_policy_paper_reference': ('pamil_policy', 'Complete explicit paper-reference native512 SFFR/fused history/TCM, LIIS topM or GMSS/GHSS, WSL/STL/stopgradSIA/actualTRAINreward/PPO/AdaMax300/fullstate', 'classification'), 'pamil_policy_released_sampling_reference_training': ('pamil_policy', 'Actual released LIIS weighted replacement/padding/floorT/retainfirst/rawhistory control, with complete named reconstructed paper training/objectives/reward/PPO', 'classification'), 'focus_source': ('focus', 'Original FOCUS native full trainable CONCH text/prompt, complete visual selection/compression/8head attention, one-bag CE, VAL-error firstbest', 'classification')})
SOURCES.update({'pamil_policy': 'titizheng/PAMIL', 'focus': 'dddavid4real/FOCUS'})

PROFILES.update({'dispro_source_repaired':('dispro','Full original native UniPro-P30 + UniPro-G50 + MultiPro50; explicit actual-Parameter/CLS/cosine/mask/presence repairs, outcome-free VAL-selected teachers and full-state continuation','survival')})


# Complete fixed source adapters must survive regeneration in an empty target.
RECENT_SOURCE_PROFILES = [{'id': 'lrmil_source_repaired',
  'family': 'lrmil',
  'recipe': 'Complete native CONCH768/12-layer patch KD, actual VAL-selected HRteacher and LRstudent source '
            'CE/bagKL/softKL/hardCE; independent source phase resets, selected-encoder features and all3Adam '
            'states. Explicit layout/teacher-tuple ports and declared200/200/200 reference caps.',
  'tasks': ['classification'],
  'required_inputs': ['native_CONCH768_hr_intermediates',
                      'measured_RGB256_5x',
                      'actual_pretrained_CONCH_weights',
                      'single_slide_bag'],
  'class_count': None,
  'external_source': 'lrmil',
  'gpu_validation': 'not_run',
  'real_wsi_validation': 'not_run',
  'status': 'implemented_unvalidated',
  'paper_reproduction': 'not_run'},
 {'id': 'kmgmamba_source',
  'family': 'kmgmamba',
  'recipe': 'Full original96width/9neighbor/top-degree prototype graph, cross-scale4state selective '
            'scan,128attention,three weighted CE/SVM losses,Adam100max/VALAUC latest ties; original binary '
            'heads',
  'tasks': ['classification'],
  'required_inputs': ['patch_features',
                      'single_slide_bag',
                      'kmgmamba_measured_dual_scale',
                      'per_slide_coords'],
  'class_count': 2,
  'external_source': 'kmgmamba',
  'gpu_validation': 'not_run',
  'real_wsi_validation': 'not_run',
  'status': 'implemented_unvalidated',
  'paper_reproduction': 'not_run'},
 {'id': 'kmgmamba_multiclass_repaired',
  'family': 'kmgmamba',
  'recipe': 'Full original96width/9neighbor/top-degree prototype graph, cross-scale4state selective '
            'scan,128attention,three weighted CE/SVM losses,Adam100max/VALAUC latest ties; explicit C-class '
            'attention/prototype head repair',
  'tasks': ['classification'],
  'required_inputs': ['patch_features',
                      'single_slide_bag',
                      'kmgmamba_measured_dual_scale',
                      'per_slide_coords'],
  'class_count': None,
  'external_source': 'kmgmamba',
  'gpu_validation': 'not_run',
  'real_wsi_validation': 'not_run',
  'status': 'implemented_unvalidated',
  'paper_reproduction': 'not_run'}]
RECENT_SOURCE_METHODS = {'kmgmamba': {'id': 'kmgmamba',
              'name': 'KMG-Mamba',
              'group': 'D_multiscale',
              'scope_origin': 'user_164_admission_list',
              'source_url': 'https://github.com/dmhdmhdmh/KMG-Mamba',
              'source_status': 'git_commit_locked',
              'source_commit': '4b5075bb2d3eddf46f9bc4f46175d250b134e27c',
              'license': 'not_declared_in_pinned_repository; private verified-source loading without '
                         'redistribution',
              'profiles': ['kmgmamba_source', 'kmgmamba_multiclass_repaired'],
              'status': 'adapted',
              'required_inputs': ['patch_features',
                                  'single_slide_bag',
                                  'kmgmamba_measured_dual_scale',
                                  'per_slide_coords'],
              'paper_reproduction': 'not_run',
              'real_wsi_validation': 'not_run',
              'requirements_review': 'Complete source architecture,three losses,original100max/AUC tie '
                                     'stopping and full optimizer/RNG recovery implemented; numerical and '
                                     'training validation pending',
              'blocker': None},
 'lrmil': {'id': 'lrmil',
           'name': 'LRMIL',
           'group': 'D_multiscale',
           'scope_origin': 'user_164_admission_list',
           'source_url': 'https://github.com/hvcl/LRMIL',
           'source_status': 'git_commit_locked',
           'source_commit': '66ea1c5a447ed2a830bb1af6739ab9261e61000d',
           'license': 'Apache-2.0; native CONCH dependency retains its own terms',
           'profiles': ['lrmil_source_repaired'],
           'status': 'adapted',
           'required_inputs': ['measured_low_RGB256_5x',
                               'native_CONCH768_high_features_and_layer0_1_10',
                               'measured_cross_resolution_coordinates',
                               'actual_pretrained_CONCH_visual_checkpoint'],
           'paper_reproduction': 'not_run',
           'real_wsi_validation': 'not_run',
           'requirements_review': 'Complete3phase source trainer and actual native pretrained '
                                  'encoder/input/CLI written; training validation in progress, no formal '
                                  'admission.',
           'blocker': None}}

RECENT_SOURCE_PROFILES.extend([{'id': 'sammil_attn_source_control', 'family': 'sammil', 'recipe': 'Complete released512/128 attention SAM-MIL, all five pseudo-bag CE and area consistency objectives, fixed deepcopy teacher, Adam/cosine200 source recipe; explicit source loss-overwrite/teacher-gradient/mask-index control', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'single_slide_bag', 'sammil_measured_regions'], 'class_count': 2, 'external_source': 'sammil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'status': 'implemented_unvalidated', 'paper_reproduction': 'not_run'}, {'id': 'sammil_attn_objective_repaired', 'family': 'sammil', 'recipe': 'Complete released512/128 attention SAM-MIL, all five pseudo-bag CE and area consistency objectives, fixed deepcopy teacher, Adam/cosine200 source recipe; explicit global CE/student-consistency/compact region-index repairs', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'single_slide_bag', 'sammil_measured_regions'], 'class_count': None, 'external_source': 'sammil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'status': 'implemented_unvalidated', 'paper_reproduction': 'not_run'}])
RECENT_SOURCE_METHODS['sammil'] = {'id': 'sammil', 'name': 'SAM-MIL', 'group': 'H_extra_supervision', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/FangHeng/SAM-MIL', 'source_status': 'git_commit_locked', 'source_commit': 'ccff640825321a12658daa0a85c03390bc9d367b', 'license': 'Apache-2.0', 'profiles': ['sammil_attn_source_control', 'sammil_attn_objective_repaired'], 'status': 'adapted', 'required_inputs': ['patch_features', 'single_slide_bag', 'sammil_measured_regions'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'full_released_recipe_and_named_objective_repairs', 'blocker': None}

RECENT_SOURCE_PROFILES.extend([{'id': 'c2c_cluster_source', 'family': 'c2c', 'recipe': 'Full end-to-end original pretrained ResNet18/256/64/32attention; each-epoch actual FAISS8group and up-to8patch/group sampling; CE+.01patchCE+.01uniform-KL, all-patch128streaming/source25epoch Adam and latest VAL-accuracy tie; original Notebook BN with exact Torch1.5 semantics', 'tasks': ['classification'], 'required_inputs': ['original_rgb_patch_bags', 'actual_original_resnet18_weights', 'single_slide_bag', 'c2c_historical_assets'], 'class_count': 2, 'external_source': 'c2c', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'status': 'implemented_unvalidated', 'paper_reproduction': 'not_run'}, {'id': 'c2c_cluster_default_bn_source', 'family': 'c2c', 'recipe': 'Full end-to-end original pretrained ResNet18/256/64/32attention; each-epoch actual FAISS8group and up-to8patch/group sampling; CE+.01patchCE+.01uniform-KL, all-patch128streaming/source25epoch Adam and latest VAL-accuracy tie; explicit original constructor-default BN condition', 'tasks': ['classification'], 'required_inputs': ['original_rgb_patch_bags', 'actual_original_resnet18_weights', 'single_slide_bag', 'c2c_historical_assets'], 'class_count': 2, 'external_source': 'c2c', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'status': 'implemented_unvalidated', 'paper_reproduction': 'not_run'}, {'id': 'c2c_cluster_multiclass_repaired', 'family': 'c2c', 'recipe': 'Full end-to-end original pretrained ResNet18/256/64/32attention; each-epoch actual FAISS8group and up-to8patch/group sampling; CE+.01patchCE+.01uniform-KL, all-patch128streaming/source25epoch Adam and latest VAL-accuracy tie; original Notebook BN with exact Torch1.5 semantics; explicit C-way weak patch label repair', 'tasks': ['classification'], 'required_inputs': ['original_rgb_patch_bags', 'actual_original_resnet18_weights', 'single_slide_bag', 'c2c_historical_assets'], 'class_count': None, 'external_source': 'c2c', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'status': 'implemented_unvalidated', 'paper_reproduction': 'not_run'}])
RECENT_SOURCE_METHODS['c2c'] = {'id': 'c2c', 'name': 'Cluster-to-Conquer', 'group': 'E_encoder_training', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/YashSharma/C2C', 'source_status': 'git_commit_locked', 'source_commit': 'd53b6350372117bf3e5251044a856e0251d810b2', 'license': 'MIT', 'profiles': ['c2c_cluster_source', 'c2c_cluster_default_bn_source', 'c2c_cluster_multiclass_repaired'], 'status': 'adapted', 'required_inputs': ['original_rgb_patch_bags', 'actual_original_resnet18_weights', 'single_slide_bag'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'complete_source_reclustering_sampling_and_three_losses_with_historical_BN_compatibility', 'blocker': None}

RECENT_SOURCE_PROFILES.extend([{'id': 'sparseconvmil_raw_source_c32', 'family': 'sparseconvmil', 'recipe': 'Full original end-to-end ResNet18, two3x3 c32 original compiled submanifold sparse convolutions, active-only mean/32classifier,200epoch Adam/with-replacement tile sampling; source spatial transforms and final weights; explicit seeded evaluation extension', 'tasks': ['classification'], 'required_inputs': ['original_rgb_patch_bags', 'actual_raw_tile_coordinates', 'compiled_sparseconvnet', 'single_slide_bag'], 'class_count': None, 'external_source': 'sparseconvmil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'status': 'implemented_unvalidated', 'paper_reproduction': 'not_run'}, {'id': 'sparseconvmil_raw_source_c128', 'family': 'sparseconvmil', 'recipe': 'Full original end-to-end ResNet18, two3x3 c128 original compiled submanifold sparse convolutions, active-only mean/32classifier,200epoch Adam/with-replacement tile sampling; source spatial transforms and final weights; explicit seeded evaluation extension', 'tasks': ['classification'], 'required_inputs': ['original_rgb_patch_bags', 'actual_raw_tile_coordinates', 'compiled_sparseconvnet', 'single_slide_bag'], 'class_count': None, 'external_source': 'sparseconvmil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'status': 'implemented_unvalidated', 'paper_reproduction': 'not_run'}])
RECENT_SOURCE_METHODS['sparseconvmil'] = {'id': 'sparseconvmil', 'name': 'SparseConvMIL', 'group': 'E_encoder_training', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/MarvinLer/SparseConvMIL', 'source_status': 'git_commit_locked', 'source_commit': 'd7dd705bc3dd4ebfb6a36f0589698400c9fa4c15', 'license': 'AGPL-3.0; external code fetched separately, not vendored', 'profiles': ['sparseconvmil_raw_source_c32', 'sparseconvmil_raw_source_c128'], 'status': 'adapted', 'required_inputs': ['original_rgb_patch_bags', 'actual_raw_tile_coordinates', 'compiled_sparseconvnet', 'single_slide_bag'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'complete_public_source_recipe_and_paper_differences_recorded', 'blocker': None}


RECENT_SOURCE_PROFILES.extend([{'id': 'conmil_pos_source', 'family': 'conmil', 'recipe': 'Complete native512/256/128 gated semantic attention,learned alpha and original cubic B-spline KAN; actual frozen CONCH positive/negative text; TRAIN-only4096cap,source50epoch Adam/cosine and strict VAL accuracy; original binary target and selection branch', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_conch_visual_text_space', 'actual_encoded_positive_queries', 'train_only_prompt_review_provenance', 'single_slide_bag'], 'class_count': 2, 'external_source': 'conmil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'conmil_posneg_source', 'family': 'conmil', 'recipe': 'Complete native512/256/128 gated semantic attention,learned alpha and original cubic B-spline KAN; actual frozen CONCH positive/negative text; TRAIN-only4096cap,source50epoch Adam/cosine and strict VAL accuracy; original binary target and selection branch', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_conch_visual_text_space', 'actual_encoded_positive_queries', 'train_only_prompt_review_provenance', 'single_slide_bag', 'actual_encoded_negative_queries'], 'class_count': 2, 'external_source': 'conmil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'conmil_posneg_multiclass_repaired', 'family': 'conmil', 'recipe': 'Complete native512/256/128 gated semantic attention,learned alpha and original cubic B-spline KAN; actual frozen CONCH positive/negative text; TRAIN-only4096cap,source50epoch Adam/cosine and strict VAL accuracy; explicit C-way labels/metrics/first-checkpoint repair', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_conch_visual_text_space', 'actual_encoded_positive_queries', 'train_only_prompt_review_provenance', 'single_slide_bag', 'actual_encoded_negative_queries'], 'class_count': None, 'external_source': 'conmil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}])
RECENT_SOURCE_METHODS['conmil'] = {'id': 'conmil', 'name': 'ConMIL', 'group': 'G_vision_language', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/anxuanhan/ConMIL', 'source_status': 'git_commit_locked', 'source_commit': 'aabb5386d15a6002a3a39437ea46c0d99d29a2f3', 'license': 'MIT; original source fetched privately', 'profiles': ['conmil_pos_source', 'conmil_posneg_source', 'conmil_posneg_multiclass_repaired'], 'status': 'adapted', 'required_inputs': ['actual_encoded_negative_queries', 'actual_encoded_positive_queries', 'native_conch_visual_text_space', 'patch_features', 'single_slide_bag', 'train_only_prompt_review_provenance'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'actual_full_source_KAN_attention_train_and_original_CONCH_prompt_encoding', 'blocker': None}



RECENT_SOURCE_PROFILES.extend([{'id': 'miladapter_zsmil_bgap_zs', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/BGAP/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_bgap_random', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/BGAP/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_bgmp_zs', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/BGMP/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_bgmp_random', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/BGMP/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_abmil_zs', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/ABMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_abmil_random', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/ABMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_transmil_zs', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/TransMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_transmil_random', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/TransMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_wikgmil_zs', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/WIKGMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_wikgmil_random', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/WIKGMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_ilramil_zs', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/ILRAMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_ilramil_random', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/ILRAMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_rrtmil_zs', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/RRTMIL/ZS; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_zsmil_rrtmil_random', 'family': 'miladapter', 'recipe': 'Actual complete author ZSMIL/RRTMIL/random; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_taskres_bgap', 'family': 'miladapter', 'recipe': 'Actual complete author TaskRes/BGAP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_taskres_bgmp', 'family': 'miladapter', 'recipe': 'Actual complete author TaskRes/BGMP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_taskres_abmil', 'family': 'miladapter', 'recipe': 'Actual complete author TaskRes/ABMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_taskres_transmil', 'family': 'miladapter', 'recipe': 'Actual complete author TaskRes/TransMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_clipadapter_bgap', 'family': 'miladapter', 'recipe': 'Actual complete author CLIPAdapter/BGAP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_clipadapter_bgmp', 'family': 'miladapter', 'recipe': 'Actual complete author CLIPAdapter/BGMP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_clipadapter_abmil', 'family': 'miladapter', 'recipe': 'Actual complete author CLIPAdapter/ABMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_clipadapter_transmil', 'family': 'miladapter', 'recipe': 'Actual complete author CLIPAdapter/TransMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_tipadapter_bgap', 'family': 'miladapter', 'recipe': 'Actual complete author TIPAdapter/BGAP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_tipadapter_bgmp', 'family': 'miladapter', 'recipe': 'Actual complete author TIPAdapter/BGMP/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_tipadapter_abmil', 'family': 'miladapter', 'recipe': 'Actual complete author TIPAdapter/ABMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'miladapter_tipadapter_transmil', 'family': 'miladapter', 'recipe': 'Actual complete author TIPAdapter/TransMIL/None; full512CONCH/768KEEP native aggregators, exact actual author public text prototypes, classwise k-shot TRAIN support, full source20epoch AdamW/cosine/CEsum/final weights. Undefined ZS temperature and CLIP/TIP ABMIL tuple/KEEP native input-width repairs explicitly declared; original no-independent-test sampling extended to patient-disjoint TRAIN pool.', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'native_author_conch_or_keep_text_prototypes', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'class_count': None, 'external_source': 'miladapter', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}])
RECENT_SOURCE_METHODS['miladapter'] = {'id': 'miladapter', 'name': 'MIL-Adapter', 'group': 'G_vision_language', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/cvblab/MIL-Adapter', 'source_status': 'git_commit_locked', 'source_commit': 'ade11bb19c3f09a35da9e8255a87a079217aef48', 'license': 'No LICENSE file at fixed commit; privately fetched, no vendoring', 'profiles': ['miladapter_zsmil_bgap_zs', 'miladapter_zsmil_bgap_random', 'miladapter_zsmil_bgmp_zs', 'miladapter_zsmil_bgmp_random', 'miladapter_zsmil_abmil_zs', 'miladapter_zsmil_abmil_random', 'miladapter_zsmil_transmil_zs', 'miladapter_zsmil_transmil_random', 'miladapter_zsmil_wikgmil_zs', 'miladapter_zsmil_wikgmil_random', 'miladapter_zsmil_ilramil_zs', 'miladapter_zsmil_ilramil_random', 'miladapter_zsmil_rrtmil_zs', 'miladapter_zsmil_rrtmil_random', 'miladapter_taskres_bgap', 'miladapter_taskres_bgmp', 'miladapter_taskres_abmil', 'miladapter_taskres_transmil', 'miladapter_clipadapter_bgap', 'miladapter_clipadapter_bgmp', 'miladapter_clipadapter_abmil', 'miladapter_clipadapter_transmil', 'miladapter_tipadapter_bgap', 'miladapter_tipadapter_bgmp', 'miladapter_tipadapter_abmil', 'miladapter_tipadapter_transmil'], 'status': 'adapted', 'required_inputs': ['native_author_conch_or_keep_text_prototypes', 'patch_features', 'patient_disjoint_train_support_pool', 'single_slide_bag'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'full_author_26_actual_routes_and_all_native_aggregators_source20epoch_AdamW_cosine_CEsum_persistent_python_order_source_seed42_boundary_TIP_trainable_support_cache_and_final_only; actual_imported_prototype_generation_details_unknown; explicit_source_bug_and_patient_disjoint_protocol_ports', 'blocker': None}

RECENT_SOURCE_PROFILES.extend([{'id': 'center_embeddings_bag_classifier_repaired', 'family': 'center_embeddings', 'recipe': 'Full source original ResNet34/FPN/32x256RGB bagCNN/shared512embedding, actual original ImageNet weights and torchvision0.7 preprocessing; exhaustive per-epoch source TRAIN and VAL candidate scores/top32/ordered bags, complete instance CE + bag CE + center loss, separate two-group Adam and center Adam, literal source staircase and20epoch last-tie VAL balanced-ACC recipe. Explicit undefined bag_feat -> actual normalized bag_embed repair; bag_classifier inference.', 'tasks': ['classification'], 'required_inputs': ['original_raw256_rgb_patch_bags', 'actual_level0_tile_centers', 'historical_imagenet_resnet34', 'single_slide_bag'], 'class_count': 2, 'external_source': 'center_embeddings', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'center_embeddings_soft_assignment_repaired', 'family': 'center_embeddings', 'recipe': 'Full source original ResNet34/FPN/32x256RGB bagCNN/shared512embedding, actual original ImageNet weights and torchvision0.7 preprocessing; exhaustive per-epoch source TRAIN and VAL candidate scores/top32/ordered bags, complete instance CE + bag CE + center loss, separate two-group Adam and center Adam, literal source staircase and20epoch last-tie VAL balanced-ACC recipe. Explicit undefined bag_feat -> actual normalized bag_embed repair; soft_assignment inference.', 'tasks': ['classification'], 'required_inputs': ['original_raw256_rgb_patch_bags', 'actual_level0_tile_centers', 'historical_imagenet_resnet34', 'single_slide_bag'], 'class_count': 2, 'external_source': 'center_embeddings', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}])
RECENT_SOURCE_METHODS['center_embeddings'] = {'id': 'center_embeddings', 'name': 'MIL with Center Embeddings', 'group': 'E_encoder_training', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/PhilipChicco/MICCAI2020mil', 'source_status': 'git_commit_locked', 'source_commit': '871b1ecad3f0480c526536b11fafe4d759273a25', 'license': 'GPL-3.0; privately fetched original source, not vendored', 'profiles': ['center_embeddings_bag_classifier_repaired', 'center_embeddings_soft_assignment_repaired'], 'status': 'adapted', 'required_inputs': ['original_raw256_rgb_patch_bags', 'actual_level0_tile_centers', 'historical_imagenet_resnet34', 'single_slide_bag'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'complete_original_end_to_end_raw_R34_FPN_bagCNN_three_loss_dual_Adam_source_exhaustive_positive_top32_and_original0.7augmentations; source_undefined_bag_feat_repair; fullstate_and_original_default_vs_center_soft_assignment_inference; exact_frozen_tile_extraction_provenance_and_worker0_recovery_port', 'blocker': None}


RECENT_SOURCE_PROFILES.extend([{'id': 'convl_yaml_fp16_source_repaired', 'family': 'convl', 'recipe': 'Full original nativeCONCH512/768hidden12layer contextual text and16dynamic prompts, shared256projection/fourheadCGA/visual+concept heads; source NLL+.1sigmoidKL-.5instanceClustering-.3conceptClustering, complete patient bags, Lookahead/RAdam200max/gradacc2 including odd-tail divisor2, NLL-only VAL stopping8, full optimizer/scaler/RNG recovery; original CUDAFP16 mixed precision', 'tasks': ['survival'], 'required_inputs': ['patch_features', 'native_conch512_space', 'actual_pretrained_conch_text', 'original_convl_concepts', 'patient_survival_bins', 'complete_patient_bags'], 'class_count': None, 'external_source': 'convl', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'convl_yaml_fp32_reference', 'family': 'convl', 'recipe': 'Full original nativeCONCH512/768hidden12layer contextual text and16dynamic prompts, shared256projection/fourheadCGA/visual+concept heads; source NLL+.1sigmoidKL-.5instanceClustering-.3conceptClustering, complete patient bags, Lookahead/RAdam200max/gradacc2 including odd-tail divisor2, NLL-only VAL stopping8, full optimizer/scaler/RNG recovery; explicit full-native FP32 precision reference', 'tasks': ['survival'], 'required_inputs': ['patch_features', 'native_conch512_space', 'actual_pretrained_conch_text', 'original_convl_concepts', 'patient_survival_bins', 'complete_patient_bags'], 'class_count': None, 'external_source': 'convl', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}])
RECENT_SOURCE_METHODS['convl'] = {'id': 'convl', 'name': 'ConVL', 'group': 'G_vision_language', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/junjianli106/ConVL', 'source_status': 'git_commit_locked', 'source_commit': 'f1017a7ea0f51ff8d436b1334f937b85aa86c46b', 'license': 'not_declared_at_fixed_commit; privately fetched original source, not vendored', 'profiles': ['convl_yaml_fp16_source_repaired', 'convl_yaml_fp32_reference'], 'status': 'adapted', 'required_inputs': ['patch_features', 'native_conch512_space', 'actual_pretrained_conch_text', 'original_convl_concepts', 'patient_survival_bins', 'complete_patient_bags'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Complete actual pretrained full text/dynamicprompt/CGA/source objectives and original patient200epoch/AMP16/gradacc2/LookaheadRAdam/NLL-only VAL selection, separate explicit FP32 reference; actual original external bin labels supported; named patient-index, modern optimizer state/import/device ports, TRAIN-only quartile extension separately declared', 'blocker': None}


SOURCES['m4'] = 'Bigyehahaha/M4'
RECENT_SOURCE_PROFILES.append({'id':'m4_multilabel_source','family':'m4',
    'recipe':'Full original5MP-AMIL experts/512D/four distinct128D gate slices per endpoint/32hidden sigmoid towers, CPE and3/5/7hierarchical convolutions, all full slide bags; source summed task probability BCE, Adam1e-4/wd4e-5/32epoch cosine0.1, original shuffled TRAIN+VAL loaders and strict VALmeanAUROC selection, complete Adam/scheduler/RNG restore',
    'tasks':['multilabel'],'required_inputs':['patch_features','complete_binary_endpoint_labels','patient_grouped_split'],
    'class_count':None,'external_source':'m4','gpu_validation':'not_run','real_wsi_validation':'not_run',
    'paper_reproduction':'not_run','status':'implemented_unvalidated'})
RECENT_SOURCE_METHODS['m4']={'id':'m4','name':'M4','group':'H_extra_supervision','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/Bigyehahaha/M4','source_status':'git_commit_locked',
    'source_commit':'2b0f92ef872fcd9997f1ae6b4ba6565fc53eb332','license':'MIT copyright2024yezhang',
    'profiles':['m4_multilabel_source'],'status':'adapted',
    'required_inputs':['patch_features','complete_binary_endpoint_labels','patient_grouped_split'],
    'paper_reproduction':'not_run','real_wsi_validation':'not_run',
    'requirements_review':'Complete original model/multi-task32epoch training/main cosine/source sampler/strict VALmeanAUROC, explicit complete measured binary labels and FM input-dimension ports; actual twoepoch source weight/Adam/scheduler/RNG equality and actual disk restore prechecks passed. Native/whole/final-wheel gates pending.',
    'blocker':None}

# Explicit reviewed metadata for methods added after the historical M4 scope.
RECENT_SOURCE_PROFILES.extend([{'id': 'libramil_source_p4', 'family': 'libramil', 'recipe': 'Complete original512D/8head/46RCCstructural descriptions or explicit full external descriptions, learned structural/bag prompts and4visual prototypes, source20iteration SOT and CE; actual pretrained CONCH frozen language encoding, full source AdamW.01decay/100warmup/cosine/20epochs, literal accuracy-AUC selection branch and complete state', 'tasks': ['classification'], 'required_inputs': ['native_CONCH_features', 'ordered_language_descriptions', 'patient_grouped_split'], 'class_count': None, 'external_source': 'libramil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'libramil_readme_p6', 'family': 'libramil', 'recipe': 'Complete original512D/8head/46RCCstructural descriptions or explicit full external descriptions, learned structural/bag prompts and6visual prototypes, source20iteration SOT and CE; actual pretrained CONCH frozen language encoding, full source AdamW.01decay/100warmup/cosine/20epochs, literal accuracy-AUC selection branch and complete state', 'tasks': ['classification'], 'required_inputs': ['native_CONCH_features', 'ordered_language_descriptions', 'patient_grouped_split'], 'class_count': None, 'external_source': 'libramil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}, {'id': 'muse_released_source', 'family': 'muse', 'recipe': 'Complete released MUSE full12layer768hidden/512projection text,16prompts,600original knowledge descriptions,8Gaussian noisy experts/top2/8headattention and20separate retrieved-view CE updates per bag; original recursive initialization, source sorted least-cosine retrieval/logits sum, complete diagnostic RNG replay without private I/O; constantAdam1e-4/wd1e-5/source200max/patience20/epoch>80 and strictVALerror', 'tasks': ['classification'], 'required_inputs': ['complete_source_feature_bags', 'actual_pretrained_CONCH', 'original600description_knowledge', 'patient_grouped_split'], 'class_count': 2, 'external_source': 'muse', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run', 'status': 'implemented_unvalidated'}])
RECENT_SOURCE_METHODS.update({'iibmil': {'id': 'iibmil', 'name': 'IIB-MIL', 'group': 'A_frozen', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/TencentAILabHealthcare/IIB-MIL', 'source_status': 'full_components_verified_training_recipe_pending', 'source_commit': 'a354ec6ea2b9ea2159e6784d54a6afb355c04b50', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': [], 'status': 'pending_adapter', 'required_inputs': ['patch_features'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Seven actual full native1280D/500instance source checks passed across all3original configurations, complete label confidence/prototype machinery and explicit1024D input interface. Full source training epoch is absent; no runnable method registered before recipe/data/resume/native/whole gates are resolved.', 'blocker': 'Fixed source has prototype/model, PartialLoss confidence EMA and inference, but no complete public trainer; prototype warmup, loss weights, padding mask and confidence-update schedule still need evidence before faithful adaptation. See docs/IIBMIL_AUDIT.md for exact paths and unresolved training contract.'}, 'gmmamba': {'id': 'gmmamba', 'name': 'GMMamba', 'group': 'A_frozen', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/titizheng/GMMamba', 'source_status': 'complete_source_adapter_under_verification', 'source_commit': '70576f1aaeef0d7041d6860799ade1ac905b85aa', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': ['gmmamba_mambamil_bimamba_source', 'gmmamba_fm_projection_repaired'], 'status': 'adapted', 'required_inputs': ['patch_features', 'per_slide_coords', 'spatial_metadata', 'single_slide_bag'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Ten full native512D/257patch/10group original component checks passed. Complete original training/selection/recovery adapter written, execution validation in progress. Missing original BiMamba dependency explicitly bound to pinned MambaMIL implementation; source hard-mask non-gradient, unused FC1, native512 and optional FM-projection repair disclosed.', 'blocker': None}, 'libramil': {'id': 'libramil', 'name': 'Libra-MIL', 'group': 'G_vision_language', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/zfy07/Libra-MIL', 'source_status': 'git_commit_locked', 'source_commit': 'f29921970bfa22d364e08ea1768b984b47e07a8d', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': ['libramil_source_p4', 'libramil_readme_p6'], 'status': 'adapted', 'required_inputs': ['native_CONCH_features', 'ordered_language_descriptions', 'patient_grouped_split'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Complete full original architecture, actual pretrained46+3RCC language embedding, original20epoch CPU references both4/6prototypes passed; unifiedCLI/recovery/full regression gates underway.', 'blocker': None}, 'muse': {'id': 'muse', 'name': 'MUSE', 'group': 'G_vision_language', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/JiahaoXu-god/CVPR2026_MUSE', 'source_status': 'git_commit_locked', 'source_commit': '0bd0e1abb402e772ed71cf208cfbd4b870175a00', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': ['muse_released_source'], 'status': 'adapted', 'required_inputs': ['complete_source_feature_bags', 'actual_pretrained_CONCH', 'original600description_knowledge', 'patient_grouped_split'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Complete original architecture and20view recipe integrated; actual88,985,107parameter/full600description/native257patch20update source precheck and exact651diagnosticRNG oracle passed. Full unified/data/source-reference/recovery/native gates still required; released-code/paper differences explicitly documented.', 'blocker': None}})
REVIEWED_METHOD_OVERRIDES={'iibmil': {'id': 'iibmil', 'name': 'IIB-MIL', 'group': 'A_frozen', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/TencentAILabHealthcare/IIB-MIL', 'source_status': 'full_components_verified_training_recipe_pending', 'source_commit': 'a354ec6ea2b9ea2159e6784d54a6afb355c04b50', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': [], 'status': 'pending_adapter', 'required_inputs': ['patch_features'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Seven actual full native1280D/500instance source checks passed across all3original configurations, complete label confidence/prototype machinery and explicit1024D input interface. Full source training epoch is absent; no runnable method registered before recipe/data/resume/native/whole gates are resolved.', 'blocker': 'Fixed source has prototype/model, PartialLoss confidence EMA and inference, but no complete public trainer; prototype warmup, loss weights, padding mask and confidence-update schedule still need evidence before faithful adaptation. See docs/IIBMIL_AUDIT.md for exact paths and unresolved training contract.'}, 'gmmamba': {'id': 'gmmamba', 'name': 'GMMamba', 'group': 'A_frozen', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/titizheng/GMMamba', 'source_status': 'complete_source_adapter_under_verification', 'source_commit': '70576f1aaeef0d7041d6860799ade1ac905b85aa', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': ['gmmamba_mambamil_bimamba_source', 'gmmamba_fm_projection_repaired'], 'status': 'adapted', 'required_inputs': ['patch_features', 'per_slide_coords', 'spatial_metadata', 'single_slide_bag'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Ten full native512D/257patch/10group original component checks passed. Complete original training/selection/recovery adapter written, execution validation in progress. Missing original BiMamba dependency explicitly bound to pinned MambaMIL implementation; source hard-mask non-gradient, unused FC1, native512 and optional FM-projection repair disclosed.', 'blocker': None}, 'libramil': {'id': 'libramil', 'name': 'Libra-MIL', 'group': 'G_vision_language', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/zfy07/Libra-MIL', 'source_status': 'git_commit_locked', 'source_commit': 'f29921970bfa22d364e08ea1768b984b47e07a8d', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': ['libramil_source_p4', 'libramil_readme_p6'], 'status': 'adapted', 'required_inputs': ['native_CONCH_features', 'ordered_language_descriptions', 'patient_grouped_split'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Complete full original architecture, actual pretrained46+3RCC language embedding, original20epoch CPU references both4/6prototypes passed; unifiedCLI/recovery/full regression gates underway.', 'blocker': None}, 'muse': {'id': 'muse', 'name': 'MUSE', 'group': 'G_vision_language', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/JiahaoXu-god/CVPR2026_MUSE', 'source_status': 'git_commit_locked', 'source_commit': '0bd0e1abb402e772ed71cf208cfbd4b870175a00', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': ['muse_released_source'], 'status': 'adapted', 'required_inputs': ['complete_source_feature_bags', 'actual_pretrained_CONCH', 'original600description_knowledge', 'patient_grouped_split'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Complete original architecture and20view recipe integrated; actual88,985,107parameter/full600description/native257patch20update source precheck and exact651diagnosticRNG oracle passed. Full unified/data/source-reference/recovery/native gates still required; released-code/paper differences explicitly documented.', 'blocker': None}}

REVIEWED_METHOD_OVERRIDES.update({'dyko': {'id': 'dyko', 'name': 'DyKo', 'group': 'G_vision_language', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/junjianli106/DyKo', 'source_status': 'pinned_source_components_verified_full_native_pending', 'source_commit': 'a8b43645840cf283c334b07c42cf4ddf230f644a', 'license': 'no_LICENSE_in_pinned_tree_private_loading_only', 'profiles': [], 'status': 'pending_adapter', 'required_inputs': ['native_aligned768D_patch_features', 'actual_pretrained_TITAN_text_tower', 'released_cohort_concept_bank_and_class_prompts', 'patient_safe_source_split_and_fewshot_manifest', 'native_GPU_FAISS_or_explicit_CPU_reference'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'source_components_30_verified_full_adapter_pending', 'blocker': 'Authorized original TITAN assets and full source/native/CLI training evidence required; no substitute language model; no executable profile registered'}})

RECENT_SOURCE_PROFILES.extend([{'id': 'taxomil_gpt2_all_losses_key_repaired', 'family': 'taxomil', 'recipe': 'Full original124M-pretrained GPT2 dualhead generative TaxoMIL,1024D gatedMIL,all four source losses,3-cycle schedule,AdamW1e-05/wd.01,100max/patience20/VAL exactmatch; explicit c_ital/f_ital lookup repair; released CLI learning rate', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'hierarchical_labels', 'taxomil_original_gpt2_assets', 'single_slide_bag'], 'class_count': None, 'external_source': 'taxomil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'status': 'implemented_unvalidated', 'paper_reproduction': 'not_run'}, {'id': 'taxomil_gpt2_paper_lr_key_repaired', 'family': 'taxomil', 'recipe': 'Full original124M-pretrained GPT2 dualhead generative TaxoMIL,1024D gatedMIL,all four source losses,3-cycle schedule,AdamW0.0001/wd.01,100max/patience20/VAL exactmatch; explicit c_ital/f_ital lookup repair; paperAppendixD learning rate only, source loss discrepancies disclosed', 'tasks': ['classification'], 'required_inputs': ['patch_features', 'hierarchical_labels', 'taxomil_original_gpt2_assets', 'single_slide_bag'], 'class_count': None, 'external_source': 'taxomil', 'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'status': 'implemented_unvalidated', 'paper_reproduction': 'not_run'}])


from milplus.nciemil_spec import PROFILES as NCI_PROFILES,METHOD as NCI_METHOD
RECENT_SOURCE_PROFILES.extend(json.loads(json.dumps(NCI_PROFILES)))
from milplus.attention2majority_spec import PROFILES as A2M_PROFILES,METHOD as A2M_METHOD
RECENT_SOURCE_PROFILES.extend(json.loads(json.dumps(A2M_PROFILES)))
RECENT_SOURCE_METHODS['attention2majority']=json.loads(json.dumps(A2M_METHOD))


from milplus.icmil_context_spec import PROFILES as CONTEXT_PROFILES,METHOD as CONTEXT_METHOD
RECENT_SOURCE_PROFILES.extend(json.loads(json.dumps(CONTEXT_PROFILES)))
RECENT_SOURCE_METHODS['icmil_context']=json.loads(json.dumps(CONTEXT_METHOD))
SOURCES['icmil_context']='injurise/ICMIL'


from milplus.icmil_coupled_spec import PROFILES as COUPLED_PROFILES,METHOD as COUPLED_METHOD
RECENT_SOURCE_PROFILES.extend(json.loads(json.dumps(COUPLED_PROFILES)))
RECENT_SOURCE_METHODS['icmil_coupled']=json.loads(json.dumps(COUPLED_METHOD))
SOURCES['icmil_coupled']='Dootmaan/ICMIL'


from milplus.pmil_prototypical_spec import PROFILES as PMIL_PROFILES,METHOD as PMIL_METHOD
RECENT_SOURCE_PROFILES.extend(json.loads(json.dumps(PMIL_PROFILES)))
RECENT_SOURCE_METHODS['pmil_prototypical']=json.loads(json.dumps(PMIL_METHOD))
SOURCES['pmil_prototypical']='zihou98/PMIL'


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "src/milplus/resources"
    target.mkdir(exist_ok=True, parents=True)
    lock_path = target / "sources.lock.json"
    locks = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    records = []
    blockers = {
        'longmil': 'Pinned full-history source only releases v1 ALiBi2D/xformers; NeurIPS v2 local/chunked attention and 2x2 pooling remain coming-soon, no full trainer. Do not relabel v1 as complete v2. See docs/PENDING_SOURCE_REVIEW.md.',
        'smmile': 'Physical NIC grid, frozen input-bound SLIC assets, three author network variants and BCE/refinement/MRF numerical checks are implemented. Full two-stage trainer, multilabel task/metrics contract, exact resume and instance artifact export remain pending. No runnable-method count from kernels alone. See docs/SMMILE.md.',
        'mammil': 'Dedicated Mamba2 SSD backend and author HNSW/MST traversals, original RAdam training. Source and explicit corrected layer loops, native one-WSI survival and named patient-union extension. See docs/MAMMIL.md.',
        'gmmamba': 'Released code imports an absent, unpinned custom BiMamba; IMM input projection is unused and hard selector gradients/small groups need auditing. Full grouping, CSS and AdamW training must be adapted with explicit repairs. See docs/STATE_SPACE_REVIEW_QUEUE.md.',
        'advmil': 'Author generator/discriminator, region-level projection and staged adversarial updates identified. Public discriminator bce option uses log(sigmoid(fake))-1 rather than standard fake BCE; region ordering and supervised/semi-supervised contracts need explicit adaptation and objective verification.',
        'agdrmil': 'Author attention grader is used only through boolean indices and receives no training gradient. Heads apply Sigmoid before CE; constant-score normalization is undefined. Requires a documented original-versus-repaired recipe, not a silent claim of learned attention grading.',
        'pamil_prototype': 'Complete staged adapter with extra TRAIN instance supervision, source and repaired classification routes and multilabel BCE. See docs/PAMIL.md.',
        'h2mil': 'Pinned graph trainer reuses incompatible LayerNorm widths, hardcodes four outputs and applies softmax before CE. Pool fitness parameters feed only hard indices and have no loss gradient; unused pooling modules and graph input protocol require a documented full-method repair. See docs/PENDING_SOURCE_REVIEW.md.',
        'sclwc': 'Pinned author trainer never forwards epoch, disabling contrastive loss; even when enabled, the contrastive queue features are detached so no network gradient flows. Needs a paper-grounded repair before admission; see docs/HIERARCHICAL_MIL.md.',
        'ins': 'Author source pinned. Frozen-feature trainer and RGB joint contrastive/prototype trainer are distinct; full confidence bank, momentum encoder, queue and pseudo-target schedule still require adaptation. See docs/PROTOTYPE_DP.md.',
        'iibmil': 'Fixed source has prototype/model, PartialLoss confidence EMA and inference, but no complete public trainer; prototype warmup, loss weights, padding mask and confidence-update schedule still need evidence before faithful adaptation. See docs/IIBMIL_AUDIT.md for exact paths and unresolved training contract.',
        'cbmil': 'Fixed source has a missing models.mil import and requires cluster selection, CLAM instance loss and contrastive/mixed-bag training; not a CE-only adapter.',
        'ptcmil': 'Prompt/cluster token merger, prompt-context updates and PTC loss require adaptation; current source contains CUDA-specific allocation and TCFormer dependencies.',
        'sgpmil': 'Sparse Gaussian-process objective and variational parameters need a dedicated GPyTorch trainer; a forward-only wrapper is insufficient.',
        'momil': 'Mamba2 backend and MoE route must be verified; the upstream third-party torch convolution fallback is not admitted as the original model.',
        'mambamil': 'Requires the correct SRMamba/BiMamba CUDA backend, method ordering and compatible environment; no generic Mamba fallback.',
        'mamba2dmil': 'Requires actual spatial inputs and official 2D scan implementation; pseudo-square simplified baseline is not admitted.',
        'survpath': 'Author source is pinned; pathway expression input, multimodal attention and survival discretization/loss need dedicated task schema and trainer.',
        'porpoise': 'Author source is pinned; omics alignment, fusion and original survival recipe remain; image-only Cox is not counted as PORPOISE.',
    }
    for group, raw in GROUPS.items():
        for item in raw.split("|"):
            mid, name = item.split(":", 1)
            profiles = [k for k, v in PROFILES.items() if v[0] == mid]
            source = locks.get(mid, {})
            official = source.get("official_source", source)
            records.append({"id": mid, "name": name, "group": group, "scope_origin": "user_164_admission_list",
                "source_url": "https://github.com/" + SOURCES[mid] if mid in SOURCES else None,
                "source_status": "git_commit_locked" if official.get("commit") else "source_review_pending",
                "source_commit": official.get("commit"), "license": official.get('license', 'not_yet_verified'),
                "profiles": profiles, "status": "adapted" if profiles else "pending_adapter",
                "required_inputs": ["patch_features"] if group.startswith("A_") else ["per_method_review_required"],
                "paper_reproduction": "not_run", "real_wsi_validation": "not_run",
                "requirements_review": "adapter_contract_recorded" if profiles else "pending_per_method_review",
                "blocker": None if profiles else blockers.get(mid,
                    "Original source, complete training recipe, inputs and validation must be verified; no fallback implementation")})
    assert len(records) == 164
    ahdmil = next(r for r in records if r['id'] == 'ahdmil')
    ahdmil['requirements_review'] = 'complete fixed-source asymmetric joint training with true selected DMIN pretraining, two native MobileNetV4 experts, both Adam updates per bag, all-class retention, parameter-only crossEMA, eligibility/reset and full restart state; native verification pending'
    ahdmil['required_inputs'] = ['patch_features', 'single_slide_bag', 'hdmil_paired_low_rgb', 'per_slide_coords']
    hdmil = next(r for r in records if r['id'] == 'hdmil')
    hdmil['requirements_review'] = 'complete released DMIN and LIPN, original512/256 ChebyKAN instance/bag heads and RGB16 MobileNetV4; all loss terms, two Adams and source selection/stage transfer; source first2class/global-normalization quirks explicitly preserved; native and real WSI validation pending'
    hdmil['required_inputs'] = ['patch_features', 'single_slide_bag', 'hdmil_paired_low_rgb', 'per_slide_coords']
    mgtrans = next(r for r in records if r['id'] == 'mgtrans')
    mgtrans['requirements_review'] = 'complete_released_two_scale_PAM_shared_Transformer_three_layer_GCN_semantic_negativeCE_and_32slide_Renyi_MI_window; explicit_physical_alignment_fullstate_resume_native200max_Adam_latest_VALerror_tie; source_graph_detach_and_unused12layer_config_disclosed'
    zoommil = next(r for r in records if r['id'] == 'zoommil')
    zoommil['requirements_review'] = 'complete_released_three_scale_dual_gated_100sample_differentiableTopK_Kronecker_and_sum_graph; actual_full_weighted_sampler_Adam_plateau_meanVALCE_100fixed_epochs_and_fullstate_resume; explicit_saver_device_ports_native8readers; paper2level_and_onlineFM_not_claimed'
    zoommil['required_inputs'] = ['patch_features','single_slide_bag','zoommil_three_scale_hierarchy']
    next(r for r in records if r['id'] == 'augdiff')['requirements_review'] = (
        'complete_original7view_fullwidth_DAE_EMA_pretraining_and_frozen_RAW_DDIM3heads_7xTRAIN_VAL_CE_'
        'LookaheadRAdam_accum2_AMP_cleanTEST_full_state; featureUNet_DataInterface_missing_explicit_'
        'singleton1D_reconstruction_and_time_control_vs_schedule_repair; original_historical_weights_scores_not_claimed')
    codmil = next(r for r in records if r['id'] == 'codmil')
    codmil['requirements_review'] = 'complete_original_physical_two_scale_text_masking_recipe_C_D_T_source_ports_explicit_full_VAL_objective_and_state_resume'
    codmil['required_inputs'] = ['actual_coarse_and_fine_features', 'real_coords_and_bound_parent_map', 'ordered_diagnostic_and_normal_text_embeddings']
    next(r for r in records if r['id'] == 'ins')['requirements_review'] = 'complete_native_frozen_feature_ResNet18_route_with_all_IWSCL_prototype_confidence_and_bag_objectives; actual_source_phase_sampler_SGD_schedule_and_full_state_resume; pseudoimage_condition_Cway_bag_head_bag_only_negative_labels_valid_queue_and_paper_scratch_init_explicit'
    next(r for r in records if r['id'] == 'sclwc')['requirements_review'] = 'complete_named_paper_CDA_PNM_live_anchor_WSCL_reconstruction_and_separate_literal_incomplete_source_control; explicit_PNM_reference_class_and_role_TRAIN_only_typed_banks_full_balanced_Adam_cosine_selection_exact_resume; all_undisclosed_choices_recorded'
    smmile = next(r for r in records if r['id'] == 'smmile')
    smmile['required_inputs'] = ['patch_features', 'real_physical_coords', 'patch_stride_and_units',
                                 'input_bound_superpixel_manifest', 'slide_or_multilabel_targets']
    smmile['requirements_review'] = 'full_recipe_adapter_contract_recorded'
    pamil = next(r for r in records if r['id'] == 'pamil_prototype')
    pamil['requirements_review'] = 'full_staged_recipe_extra_instance_supervision_contract_recorded'
    loss_attention = next(r for r in records if r['id'] == 'loss_attention')
    loss_attention['requirements_review'] = 'source_and_paper_full_recipes_explicit_wsi_feature_adaptation'
    next(r for r in records if r['id'] == 'micromil')['requirements_review'] = 'full_recipe_explicit_frozen_fm_transfer_with_source_repairs'
    next(r for r in records if r['id'] == 'monai_mil')['requirements_review'] = 'complete_published_raw_image_ordinal_tutorial; paper_pseudolabel_extension_not_claimed'
    next(r for r in records if r['id'] == 'campanella_rnn')['requirements_review'] = 'complete_source_two_phase_raw_roi_recipe_with_best_selector_and_exact_resume'
    next(r for r in records if r['id'] == 'mammoth')['requirements_review'] = 'native_experts_eight_classification_heads_complete_paper_based_recipe_explicit_source_differences'
    next(r for r in records if r['id'] == 'pmil_shapley')['requirements_review'] = 'complete_native_shapley_progressive_multi_round_recipe_with_source_and_named_repairs'
    next(r for r in records if r['id'] == 'msattention')['requirements_review'] = 'full_supplementary_training_and_original_three_depth_attention_explicit_frozen_wsi_transfer; private_tumour_pipeline_not_claimed'
    next(r for r in records if r['id'] == 'ttis')['requirements_review'] = 'all_six_native_heads_complete_optimizer_and_test_only_selection; explicit_spatial_semantic_assets_source_space_policy_and_CLAM_full_loss_variant'
    next(r for r in records if r['id'] == 'mamba2dmil')['requirements_review'] = 'complete_original_physical_grid_model_and_classification_survival_recipes; literal_CPU_and_named_forward_derivative_repairs; single_WSI_patient_and_TRAIN_bin_contract'
    next(r for r in records if r['id'] == 'simil')['requirements_review'] = 'full_original_dual_branch_and_coupled_training; row_aligned_raw_PathExpert_inputs_TRAIN_only_quantiles; literal_and_full_Gaussian_VJP_routes'
    next(r for r in records if r['id'] == 'hipt')['requirements_review'] = 'complete_original_pretrained_classification_model_and_source_recipe; actual_teacher_and_native_patch_space_bound_to_physical_pyramids; original_survival_Python_unreleased_not_substituted'
    next(r for r in records if r['id']=='triagemil')['requirements_review']='complete_original_MASS_quality_four_axis_FPS_semantic_hypergraphs_native_model_and_full_source_Cox_AdamW_AMP_accumulation_no_event_skips_first_tie_VAL_minimum_epoch_stop; actual_coordinate_and_FM_bound_assets_exact_resume'
    next(r for r in records if r['id']=='c2mil')['requirements_review']='complete_original_adaptive_bias_removal_causal_graph_five_losses_and_source_batch_merge_Adam_schedule_VAL_last_tie_recipe; actual_whole_slide_thumbnails_TRAIN_only1500patch_reference_physical_graphs_full_FM_channels_and_dynamic_cache_resume'
    next(r for r in records if r['id']=='h2mil')['requirements_review']='complete_original_RAConv_IHPool_and_summed_batch_recipe; actual_three_level_feature_geometry; named_shape_head_CE_repairs; released_hard_fitness_gradient_and_paper_source_boundaries_explicit'
    next(r for r in records if r['id']=='agdrmil')['requirements_review']='complete_native512_dual_branch_model_and_source_recipe; explicit_yaml_vs_TCGA_presets; original_hard_grader_gradient_boundary_and_Sigmoid_CE_retained; source_validation_RNG_and_degenerate_input_contract'
    next(r for r in records if r['id']=='c2aug')['requirements_review']='complete_named_paper_reference; native_source_TransMIL_and_paper_Eq2_to9; official_supplement_defaults; all_augmentations_active_bag_FIFO_all_group_forward_KL_learnable_prototypes_EMA_and_exact_resume; source_deficits_and_undisclosed_choices_explicit'
    next(r for r in records if r['id']=='gatmamba')['requirements_review']='complete_original_GAT_Mamba1.2.0.post1_and_source_Cox_L1_Adam_shuffled_VAL_plateau_last_tie_recipe; actual_six_class_classifier_provenance_physical_patient_graphs_full_FM_input_port_and_named_edge_conditioning_correction; exact_resume'
    next(r for r in records if r['id']=='protosurv')['requirements_review']='complete_original_graph_tissue_prototype_fusion_and_three_objectives; actual_FM_and_physical_graph_assets; full_source_sampling_augmentation_gc_Adam_cosine_selection_and_exact_resume; source_and_named_repairs_explicit'
    next(r for r in records if r['id']=='pathrwkv')['requirements_review']='complete_original_TimeMix_ChannelMix_task_projection_heads_and_heterogeneous_recipe; explicit_real_yx_geometry_native_CUDA_or_named_tensor_backend; source_and_named_Cox_missingness_streaming_repairs'
    next(r for r in records if r['id']=='amiml')['requirements_review']='complete_four_TRAIN_only_morphology_clusters_full_native8_encoder_minmax_decoder_residual_selfattention; explicit_missing_V8toC_output_and_loader_reconstruction; source_probability_CE_vs_paper_logit_reference; four_independent_Adam_plateau_stopping_seedstreams_patient_bags_and_VAL_only_two_level_selection_exact_resume'
    next(r for r in records if r['id']=='dispro').update(
        required_inputs=['patient_bags','explicit_observed_WSI_and_RNA_flags','ordered_gene_pathways','actual_pretrained_BioBERT'],
        requirements_review='complete_original_UniPro_P_UniPro_G_MultiPro_stages; actual_full12layer768_BioBERT_and_pathway_networks; repaired_actualParameter_CLS_cosine_mask_and_presence; outcome_free_VAL_teachers_TRAIN_bins_and_strict_stage_resume; default130epoch_native_budget_and_final_admission_gates_pending',blocker=None)
    assert len({r["id"] for r in records}) == 164
    profiles = [{"id": k, "family": v[0], "recipe": v[1],
        "tasks": ["classification"] if k in {"smmile", "smmile_binary"} else ["classification"] if k == "momil" or k.startswith("recomil_") else ["survival"] if k in {"momil_survival_extension", "mammil_patient_union", "mfemil_transmil"} else ["classification", "survival"] if k.startswith("mfemil_") or k in {"mammil", "mammil_source"} else ["classification"] if k in {"msmmil", "dgmil", "dgmil_source_extremes"} else ["classification", "survival"] if k.startswith(("pamoe_", "mambamil_")) else [v[2], "ordinal"] if v[2] == "classification" and not k.startswith("packmil_") else [v[2]],
        "required_inputs": ["patch_features"] + (
            ['smmile_assets', 'per_slide_coords', 'single_slide_bag'] if k in {'smmile','smmile_binary','smmile_multilabel'} else ['single_slide_bag'] if k == 'momil' else ['single_slide_bag', 'dual_scale'] if k.startswith('recomil_dual') else ['single_slide_bag'] if k.startswith('recomil_') else ['per_slide_coords'] if k == 'mammil_patient_union' else ['per_slide_coords', 'single_slide_bag'] if k in {'mammil', 'mammil_source'} else ['survival_coords'] if k.startswith('mfemil_') else ['tissue_prototypes'] if k.startswith('pamoe_') else ['regions4x4', 'per_slide_coords'] if k.startswith('advmil_') else ['hierarchical_labels', 'single_slide_bag'] if k == 'hmil_source_repaired' else ['single_slide_bag'] if k in {'msmmil', 'tpmil', 'cdpmil_source'} or k.startswith('rankmix_') else ['concept_anchors', 'single_slide_bag'] if k == 'cate' else ['learned_dual_scale_prompts', 'single_slide_bag'] if k.startswith('vilamil') else ['regression_targets', 'coords_for_supertiles'] if k == 'he2rna' else ['coords', 'single_slide_bag'] if k == 'camil_context' else ['single_slide_bag'] if k.startswith(('sdfp_', 'mfcmil_')) or k == 'camil_channel' else ['per_slide_coords', 'spatial_metadata'] if k == 'scmil' else ['omics', 'pathways'] if k.startswith(('pamoe_', 'pathomics_', 'mmp_')) or k in {'pibd_source','survpath','mcat','cmta','motcat','motcat_micro'} else ['omics'] if k in {'porpoise', 'healnet'} else
            ['coords', 'spatial_metadata', 'single_slide_bag', 'slide_canvas'] if k == 'bayesmil_apcrf' else
            ['coords', 'spatial_metadata', 'single_slide_bag'] if k.startswith('smmil_') or k in {'psamil', 'slidegraph'} else
            ['single_slide_bag', 'aligned_centers'] if k == 'csmil' else ['single_slide_bag', 'quadtree'] if k == 'hagmil_ce' else ['spatial_graph'] if k == 'patchgcn' else ['single_slide_bag'] if k in {'cbmil', 'mico'} else
            ['coords'] if k in {'attrimil', 'mmil_coords'} else []),
        "class_count": 2 if k == "smmile_binary" else 2 if k in {"mammil", "mammil_source", "mammil_patient_union", "msmmil", "dgmil", "dgmil_source_extremes"} or k.startswith(('recomil_', 'rankmix_', 'smmil_', 'bayesmil_')) or k in {'cate','frmil','focusmil','dgrmil','csmil','slidegraph'} else None,
        "external_source": v[0] if k in {"smmile", "smmile_binary", "smmile_multilabel"} else v[0] if k in {"momil", "momil_survival_extension", "mammil", "mammil_source", "mammil_patient_union", "msmmil", "dgmil", "dgmil_source_extremes"} or k.startswith(('recomil_', 'mfemil_', 'mambamil_', 'pamoe_', 'pathomics_', 'mmp_', 'advmil_', 'rankmix_', 'sdfp_', 'mfcmil_', 'mil_dropout_', 'packmil_', 'smmil_', 'bayesmil_', 'caprmil_', 'efficientmil_')) or k in {'pibd_source','hmil_source_repaired','tpmil','cdpmil_source','cate','vilamil','vilamil_source','he2rna','healnet','camil_channel','camil_context','dyhg','scmil','deepattnmisl','mico','mico_survival','varmil','psamil','slidegraph','rrt','frmil','dgrmil','survpath','mcat','porpoise','patchgcn','sgpmil','mmil','mmil_coords','ptcmil','cbmil','cmta','motcat','motcat_micro','csmil','hagmil_ce'} else None,
        "gpu_validation": "not_run", "real_wsi_validation": "not_run",
        "status": "implemented_unvalidated", "paper_reproduction": "not_run"} for k, v in PROFILES.items()]
    for profile in profiles:
        if profile['family']=='gmmamba':
            profile.update(tasks=['classification'],required_inputs=['patch_features','per_slide_coords','spatial_metadata','single_slide_bag'],class_count=None,external_source='gmmamba')
        if profile['family'] == 'ahdmil':
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'single_slide_bag', 'hdmil_paired_low_rgb', 'per_slide_coords'], class_count=3 if profile['id'] == 'ahdmil_rcc_source_repaired' else 2, external_source='ahdmil')
        if profile['family'] == 'hdmil':
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'single_slide_bag', 'hdmil_paired_low_rgb', 'per_slide_coords'], class_count=3 if profile['id'] == 'hdmil_rcc_source' else 2, external_source='hdmil')
        if profile['family'] == 'mgtrans':
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'single_slide_bag', 'mgtrans_dual_scale_alignment', 'per_slide_coords'], class_count=None, external_source='mgtrans')
        if profile['family']=='zoommil':
            profile.update(tasks=['classification'],required_inputs=['patch_features','single_slide_bag','zoommil_three_scale_hierarchy'],class_count=None,external_source='zoommil')
        if profile['family']=='augdiff':
            profile.update(tasks=['classification'],required_inputs=['patch_features','single_slide_bag','augdiff_trained_dae'],class_count=None,external_source='augdiff')
        if profile['family']=='vlsa':
            profile.update(tasks=['survival'],required_inputs=['patch_features','patient_bags','vlsa_native_conch_survival'],class_count=None,external_source='vlsa')
        if profile['id'].startswith('slip_'):
            profile.update(tasks=['classification'],required_inputs=['patch_features','single_slide_bag','slip_native_clip_prompts'],class_count=None,external_source='slip')
        if profile['id'].startswith('dispro_'):
            profile.update(tasks=['survival'],required_inputs=['dispro_assets','patient_bags','actual_pretrained_BioBERT'],
                           class_count=None,external_source='dispro')
        if profile['id'] in PRESERVED_REFERENCE_PROFILE_METADATA:
            profile.update(PRESERVED_REFERENCE_PROFILE_METADATA[profile['id']])
        if profile['id'].startswith('amiml_'):
            profile.update(tasks=['classification'], required_inputs=['patch_features','patient_bags','four_morphology_clusters'], class_count=2 if profile['id']=='amiml_source_repaired' else None, external_source='amiml')
        if profile['id'].startswith('dtmil_'):
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'coords', 'spatial_metadata', 'single_slide_bag'], class_count=2 if profile['id']=='dtmil_source_crop_reference_train' else None, external_source='dtmil')
        if profile['id'].startswith('codmil_'):
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'single_slide_bag', 'codmil_assets'],
                           class_count=None, external_source='codmil')
        if profile['id'].startswith('ins_frozen_'):
            profile.update(tasks=['classification'], required_inputs=['patch_features','single_slide_bag','full_feature_dimension_divisible_by_32'],
                           class_count=2, external_source='ins', input_condition='trainable_native_ResNet18_on_full_frozen_feature_pseudoimages')
        if profile['id'].startswith('sclwc_'):
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'single_slide_bag', 'explicit_pnm_reference_class_and_role'],
                           class_count=2 if profile['id'].endswith('source_control') else None, external_source='sclwc',
                           complete_paper_objectives=profile['id'] == 'sclwc_paper_reference')
        if profile['id'].startswith('deepgraphsurv_'):
            profile.update(tasks=['survival'],required_inputs=['patch_features','deepgraphsurv_assets'],class_count=None,external_source='deepgraphsurv')
        if profile['id'].startswith('triagemil_'):
            profile.update(tasks=['survival'],required_inputs=['patch_features','triagemil_assets','per_slide_coords','spatial_metadata','single_slide_bag'],class_count=None,external_source='triagemil')
        if profile['id'].startswith('c2mil_'):
            profile.update(tasks=['survival'],required_inputs=['patch_features','c2mil_assets','per_slide_coords','spatial_metadata','single_slide_bag'],class_count=None,external_source='c2mil')
        if profile['id'].startswith('gatmamba_'):
            profile.update(tasks=['survival'],required_inputs=['patch_features','gatmamba_assets','per_slide_coords','spatial_metadata'],class_count=None,external_source='gatmamba')
        if profile['id'].startswith('protosurv_'):
            profile.update(tasks=['survival'],required_inputs=['patch_features','protosurv_assets','per_slide_coords','spatial_metadata','single_slide_bag'],class_count=None,external_source='protosurv')
        if profile['id'].startswith('pathrwkv_'):
            profile.update(tasks=['classification','ordinal','regression','survival','multitask'],required_inputs=['patch_features','per_slide_coords','spatial_metadata','single_slide_bag'],class_count=None,external_source='pathrwkv')
        if profile['id'].startswith('c2aug_'):
            profile.update(tasks=['classification'],required_inputs=['patch_features','single_slide_bag'],external_source='c2aug')
        if profile['id'].startswith('agdrmil_'):
            profile.update(tasks=['classification'],required_inputs=['patch_features','single_slide_bag'],external_source='agdrmil')
        if profile['id'].startswith('h2mil_'):
            profile.update(tasks=['classification'],required_inputs=['patch_features','h2mil_hierarchy','per_slide_coords','single_slide_bag','actual_thumbnail_features','actual_coarse_features'],class_count=2 if profile['id']=='h2mil_source_compat' else None,external_source='h2mil')
        if profile['id'].startswith('hipt_lgp_'):
            profile.update(tasks=['classification'],required_inputs=['patch_features','hipt_pyramid','per_slide_coords','single_slide_bag','native_patch_embedding_space','pretrained_region_teacher'],class_count=None,external_source='hipt')
        if profile['id'].startswith('simil_'):
            profile.update(tasks=['classification'],required_inputs=['patch_features','pathexpert_assets','per_slide_coords','single_slide_bag'],class_count=2,external_source='simil')
        if profile['id'].startswith('mamba2dmil_'):
            profile.update(tasks=['classification', 'survival'], required_inputs=['patch_features', 'per_slide_coords', 'spatial_metadata', 'single_slide_bag', 'physical_mamba2d_grid'], class_count=None, external_source='mamba2dmil')
        if profile['id'].startswith('ttis_'):
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'single_slide_bag', 'ttis_assets', 'test_physical_coords', 'frozen_semantic_vectors'], class_count=None, external_source='ttis')
        if profile['id'].startswith('msattention_'):
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'single_slide_bag'], class_count=None, external_source='msattention')
        if profile['id'].startswith('pmil_shapley_'):
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'single_slide_bag'], class_count=None, external_source='pmil_shapley')
        if profile['id'].startswith('mammoth_'):
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'single_slide_bag'], class_count=None, external_source='mammoth')
        if profile['id'] == 'campanella_rnn':
            profile.update(tasks=['classification'], required_inputs=['raw_regions', 'single_slide_bag'], class_count=2, external_source='campanella_rnn')
        if profile['id'].startswith('monai_'):
            profile.update(tasks=['ordinal'], required_inputs=['raw_images', 'single_slide_bag'], class_count=None, external_source='monai_mil')
        if profile['id'].startswith('stablemil_'):
            profile.update(tasks=['survival'] if 'survival' in profile['id'] else ['classification'], required_inputs=['patch_features', 'per_slide_coords', 'spatial_metadata', 'single_slide_bag'], class_count=None, external_source='stablemil')
        if profile['id'] in {'setmil_source', 'setmil_repaired'}:
            profile.update(tasks=['classification'], required_inputs=['patch_features', 'coords', 'spatial_metadata', 'single_slide_bag'], class_count=None, external_source='setmil')
        if profile['id'] in {'gtp_source', 'gtp_repaired'}:
            profile.update(tasks=['classification'], required_inputs=['coords', 'spatial_metadata', 'single_slide_bag'], class_count=None, external_source='gtp')
        if profile['id'].startswith('probsa_'):
            profile.update(tasks=['classification'], required_inputs=['coords', 'spatial_metadata', 'single_slide_bag'], class_count=2, external_source='probsamil')
        if profile['id'] == 'micromil_frozen':
            profile.update(tasks=['classification'], required_inputs=['patch_features'], class_count=2, external_source='micromil')
        if profile['id'].startswith('loss_attention_'):
            profile['tasks'] = ['classification']
            profile['required_inputs'] = ['patch_features', 'single_slide_bag']
            profile['external_source'] = 'loss_attention'
        if profile['id'].startswith('pamil_prototype_'):
            profile['tasks'] = ['multilabel'] if profile['id'].endswith('multilabel') else ['classification']
            profile['required_inputs'] = ['patch_features', 'training_instance_labels', 'single_slide_bag']
            profile['external_source'] = 'pamil_prototype'
    assert not ({p['id'] for p in profiles} & {p['id'] for p in RECENT_SOURCE_PROFILES})
    profiles.extend(json.loads(json.dumps(RECENT_SOURCE_PROFILES)))
    for index, record in enumerate(records):
        if record['id'] in RECENT_SOURCE_METHODS:
            records[index] = json.loads(json.dumps(RECENT_SOURCE_METHODS[record['id']]))
    for record in records:
        implemented = [profile for profile in profiles if profile['family'] == record['id']]
        if implemented:
            record['required_inputs'] = sorted({item for profile in implemented for item in profile['required_inputs']})
    for index,record in enumerate(records):
        if record["id"] in REVIEWED_METHOD_OVERRIDES:
            records[index]=json.loads(json.dumps(REVIEWED_METHOD_OVERRIDES[record["id"]]))
    evidence_path = root / "docs/validation/profiles_cpu.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        from milplus.provenance import software_identity
        current = evidence.get('software') == software_identity()
        passed = {r["id"] for r in evidence["profiles"] if r["state"] == "passed"} if current else set()
        for p in profiles:
            if p["id"] in passed:
                p["status"] = "cpu_synthetic_validated"
            p["gpu_validation"] = "not_run"
            p["real_wsi_validation"] = "not_run"
    gpu_path = root/'docs/validation/pathrwkv_cuda.json'
    if gpu_path.exists():
        from milplus.provenance import software_identity
        gpu = json.loads(gpu_path.read_text())
        current = software_identity()
        same_code = all(gpu.get('software', {}).get(k) == current[k]
                        for k in ['source_sha256', 'sources_lock_sha256'])
        # CUDA and CPU package identities intentionally remain separate. Only
        # the actual matching runtime/lock and four completed native CLI runs
        # can promote these profiles; a queued job or kernel-only probe cannot.
        runs = gpu.get('cli_runs', [])
        expected = {(p, d) for p in ['pathrwkv_source','pathrwkv_repaired'] for d in [1024,1536]}
        if (gpu.get('state') == 'passed' and same_code and len(runs) == 4 and
                {(r.get('profile'), r.get('feature_dimension')) for r in runs} == expected and
                all(r.get('hidden') == 768 and r.get('optimizer_updates', 0) > 0 for r in runs)):
            for p in profiles:
                if p['family'] == 'pathrwkv':p['gpu_validation'] = 'h200_native_cuda_synthetic_validated'
    # Complete generative TaxoMIL; keep fatal source-key repair and LR conditions explicit.
    next(row for row in records if row["id"]=="taxomil").update({'id': 'taxomil', 'name': 'TaxoMIL', 'group': 'G_vision_language', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/QuIIL/TaxoMIL', 'source_status': 'complete_source_adapter_under_verification', 'source_commit': '5e095f0c212b3a30c3f6c6c81b3ca1a18e0c676e', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': ['taxomil_gpt2_all_losses_key_repaired', 'taxomil_gpt2_paper_lr_key_repaired'], 'status': 'adapted', 'required_inputs': ['patch_features', 'hierarchical_labels', 'taxomil_original_gpt2_assets', 'single_slide_bag'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Actual full217M-class graph/source literal model and original data/collator/train/VAL oracles verified; all gradients/moments/RNG/fullstate restored. Source all-loss lookup defect explicitly repaired; projected-vs-raw sibling objective and paper LR differences disclosed. Full native budgets and public CLI validation pending.', 'blocker': None})
    next(row for row in records if row["id"]=="nciemil").update({'id': 'nciemil', 'name': 'NcIEMIL', 'group': 'L_composite', 'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/polyethylene16/NcIEMIL', 'source_status': 'original_aggregator_components_checked_filter_pipeline_pending', 'source_commit': '780d8b4c04ef6266e552bfd4c47b76470f7510fd', 'license': 'no_LICENSE_in_pinned_tree_private_source_loading', 'profiles': [], 'status': 'pending_adapter', 'required_inputs': ['per_method_review_required'], 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run', 'requirements_review': 'Full native1024D/four-head/K128and512 source graph and every gradient/RNG checked; source interleavedKV and explicit independentKV variant separated. Paper Eq3max-instanceBCE and Eq4highK/lowK components implemented. All21 default-branch commits/8unique code blobs reviewed; no Swin/raw filtering stage found. Full filter pretraining/cache/training/publicCLI remain required.', 'blocker': 'Implement complete task-trained Swin-Tiny filtering/extraction, provenance-bound highK/lowK selected features and original100epoch SGD recipe with complete recovery before admitting a runnable method. Component validation alone is not full-method admission.'})
    next(row for row in records if row['id']=='nciemil').update(json.loads(json.dumps(NCI_METHOD)))
    from milplus.comel_spec import PROFILES as COMEL_PROFILES, METHOD as COMEL_METHOD
    assert not ({p['id'] for p in profiles}&{p['id'] for p in COMEL_PROFILES})
    profiles.extend(json.loads(json.dumps(COMEL_PROFILES)))
    records[next(i for i,r in enumerate(records) if r['id']=='comel')]=json.loads(json.dumps(COMEL_METHOD))
    from milplus.akdpmp_spec import PROFILES as AKD_PROFILES, METHOD as AKD_METHOD
    assert not ({p['id'] for p in profiles}&{p['id'] for p in AKD_PROFILES})
    profiles.extend(json.loads(json.dumps(AKD_PROFILES)))
    records[next(i for i,r in enumerate(records) if r['id']=='akdpmp')]=json.loads(json.dumps(AKD_METHOD))
    from milplus.qpmilvl_spec import PROFILES as QPMIL_PROFILES, METHOD as QPMIL_METHOD
    assert not ({p['id'] for p in profiles}&{p['id'] for p in QPMIL_PROFILES})
    profiles.extend(json.loads(json.dumps(QPMIL_PROFILES)))
    records[next(i for i,r in enumerate(records) if r['id']=='qpmilvl')]=json.loads(json.dumps(QPMIL_METHOD))
    from milplus.kmpmil_spec import PROFILES as KMP_PROFILES, METHOD as KMP_METHOD
    assert not ({p['id'] for p in profiles}&{p['id'] for p in KMP_PROFILES})
    profiles.extend(json.loads(json.dumps(KMP_PROFILES)))
    records[next(i for i,r in enumerate(records) if r['id']=='kmpmil')]=json.loads(json.dumps(KMP_METHOD))
    from milplus.protomil_spec import PROFILES as PROTO_PROFILES, METHOD as PROTO_METHOD
    assert not ({p['id'] for p in profiles}&{p['id'] for p in PROTO_PROFILES})
    profiles.extend(json.loads(json.dumps(PROTO_PROFILES)))
    records[next(i for i,r in enumerate(records) if r['id']=='protomil')]=json.loads(json.dumps(PROTO_METHOD))
    from milplus.macnet_spec import PROFILES as MAC_PROFILES, METHOD as MAC_METHOD
    assert not ({p['id'] for p in profiles}&{p['id'] for p in MAC_PROFILES})
    profiles.extend(json.loads(json.dumps(MAC_PROFILES)))
    records[next(i for i,r in enumerate(records) if r['id']=='macnet')]=json.loads(json.dumps(MAC_METHOD))
    from milplus.dspamil_spec import METHOD as DSPA_METHOD
    records[next(i for i,row in enumerate(records) if row['id']=='dspamil')]=json.loads(json.dumps(DSPA_METHOD))
    from milplus.feather_spec import METHOD as FEATHER_METHOD
    records[next(i for i,row in enumerate(records) if row['id']=='feather')]=json.loads(json.dumps(FEATHER_METHOD))
    from milplus.dyko_spec import METHOD as DYKO_METHOD
    records[next(i for i,row in enumerate(records) if row['id']=='dyko')]=json.loads(json.dumps(DYKO_METHOD))
    from milplus.mggmmil_spec import METHOD as MGGM_METHOD, PROFILES as MGGM_PROFILES
    assert not ({p["id"] for p in profiles}&{p["id"] for p in MGGM_PROFILES})
    profiles.extend(json.loads(json.dumps(MGGM_PROFILES)))
    records[next(i for i,row in enumerate(records) if row['id']=='mggmmil')]=json.loads(json.dumps(MGGM_METHOD))
    from milplus.hybrid_supervision_spec import METHOD as HYBRID_METHOD, PROFILES as HYBRID_PROFILES
    assert not ({p['id'] for p in profiles}&{p['id'] for p in HYBRID_PROFILES})
    profiles.extend(json.loads(json.dumps(HYBRID_PROFILES)))
    records[next(i for i,row in enumerate(records) if row['id']=='hybrid_supervision')]=json.loads(json.dumps(HYBRID_METHOD))
    from milplus.lnplmil_spec import METHOD as LNPL_METHOD, PROFILES as LNPL_PROFILES
    assert not ({p['id'] for p in profiles}&{p['id'] for p in LNPL_PROFILES})
    profiles.extend(json.loads(json.dumps(LNPL_PROFILES)))
    records[next(i for i,row in enumerate(records) if row['id']=='lnplmil')]=json.loads(json.dumps(LNPL_METHOD))
    from milplus.longmil_spec import METHOD as LONG_METHOD, PROFILES as LONG_PROFILES
    assert not ({p['id'] for p in profiles}&{p['id'] for p in LONG_PROFILES})
    profiles.extend(json.loads(json.dumps(LONG_PROFILES)))
    records[next(i for i,row in enumerate(records) if row['id']=='longmil')]=json.loads(json.dumps(LONG_METHOD))
    from milplus.iibmil_spec import METHOD as IIB_METHOD, PROFILES as IIB_PROFILES
    assert not ({p['id'] for p in profiles}&{p['id'] for p in IIB_PROFILES})
    profiles.extend(json.loads(json.dumps(IIB_PROFILES)))
    records[next(i for i,row in enumerate(records) if row['id']=='iibmil')]=json.loads(json.dumps(IIB_METHOD))
    from milplus.promptmil_spec import METHOD as PROMPT_METHOD, PROFILES as PROMPT_PROFILES
    assert not ({p['id']for p in profiles}&{p['id']for p in PROMPT_PROFILES})
    profiles.extend(json.loads(json.dumps(PROMPT_PROFILES)))
    records[next(i for i,row in enumerate(records)if row['id']=='promptmil')]=json.loads(json.dumps(PROMPT_METHOD))
    from milplus.lpgmil_spec import METHOD as LPG_METHOD, PROFILES as LPG_PROFILES
    assert not ({p["id"] for p in profiles}&{p["id"] for p in LPG_PROFILES})
    profiles.extend(json.loads(json.dumps(LPG_PROFILES)))
    records[next(i for i,row in enumerate(records) if row["id"]=="lpgmil")]=json.loads(json.dumps(LPG_METHOD))
    from milplus.mexd_spec import METHOD as MEXD_METHOD, PROFILES as MEXD_PROFILES
    assert not ({p["id"] for p in profiles}&{p["id"] for p in MEXD_PROFILES})
    profiles.extend(json.loads(json.dumps(MEXD_PROFILES)))
    records[next(i for i,row in enumerate(records) if row["id"]=="mexd")]=json.loads(json.dumps(MEXD_METHOD))
    # Preserve the previously frozen350-profile prefix and the two subsequently
    # appended RMDL conditions, including their source-before-paper ordering.
    profiles.sort(key=lambda row:row["id"])
    from milplus.rmdl_spec import METHOD as RMDL_METHOD, PROFILES as RMDL_PROFILES
    assert not ({p["id"] for p in profiles}&{p["id"] for p in RMDL_PROFILES})
    profiles.extend(json.loads(json.dumps(RMDL_PROFILES)))
    records[next(i for i,row in enumerate(records) if row["id"]=="rmdl")]=json.loads(json.dumps(RMDL_METHOD))
    from milplus.pneamil_spec import PROFILES as PNEAMIL_AUTHOR_PROFILES
    assert not ({p["id"] for p in profiles} & {p["id"] for p in PNEAMIL_AUTHOR_PROFILES})
    profiles.extend(json.loads(json.dumps(PNEAMIL_AUTHOR_PROFILES)))
    pneamil_record = next(row for row in records if row["id"] == "pneamil")
    pneamil_record["profiles"] = ["pneamil"] + [p["id"] for p in PNEAMIL_AUTHOR_PROFILES]
    pneamil_record.update(
        status="adapted_under_verification",
        source_status="complete_original_author_recipe_candidate_under_verification",
        license="GPL-3.0 root license at the fixed author commit",
        requirements_review=(
            "Existing controlled MIL_BASELINE port retained. Two complete author-definition "
            "recipes now preserve64axes/batch8/1024replacement patches/Adam1e-4/"
            "max1000epochs/relative0.0005/patience10. Literal TRAIN-selector and full-VAL "
            "repair are separately named; selected-checkpoint snapshot isolation is repaired. "
            "Complete source/wheel/native-budget and release validation remain pending."
        ),
    )
    from milplus.asmil_spec import PROFILES as ASMIL_AUTHOR_PROFILES
    assert not ({p["id"] for p in profiles} & {p["id"] for p in ASMIL_AUTHOR_PROFILES})
    profiles.extend(json.loads(json.dumps(ASMIL_AUTHOR_PROFILES)))
    asmil_record = next(row for row in records if row["id"] == "asmil")
    asmil_record["profiles"] = ["asmil"] + [p["id"] for p in ASMIL_AUTHOR_PROFILES]
    asmil_record.update(
        status="adapted_under_verification",
        source_status="complete_original_author_recipe_candidate_under_verification",
        requirements_review=(
            "Existing controlled port retained. Original ACMIL_MYMHA and independently "
            "initialized train-mode teacher loaded from pinned definitions. Full50/100epoch "
            "AdamW/cosine/clip5/full-bag recipes, literal NaN-only teacher and separately "
            "named finite-EMA repair. Actual micro-F1+macro-AUROC VAL selector, fixed-patient/"
            "seed/locked-TEST repairs, teacher/optimizer/RNG/sampler recovery are explicit. "
            "Complete source/wheel/native and release validation remain pending."
        ),
    )
    from milplus.attrimil_spec import PROFILES as ATTRIMIL_AUTHOR_PROFILES
    assert not ({p["id"] for p in profiles} & {p["id"] for p in ATTRIMIL_AUTHOR_PROFILES})
    profiles.extend(json.loads(json.dumps(ATTRIMIL_AUTHOR_PROFILES)))
    attrimil_record = next(row for row in records if row["id"] == "attrimil")
    attrimil_record["profiles"] = ["attrimil"] + [p["id"] for p in ATTRIMIL_AUTHOR_PROFILES]
    attrimil_record.update(
        status="adapted_under_verification",
        source_status="complete_original_author_recipe_repaired_candidate_under_verification",
        license="Apache-2.0 at fixed original author commit",
        requirements_review=(
            "Existing controlled port retained. Complete original AttriMIL architecture, SGD2e-4/"
            "momentum.9/wd1e-5, spatial1/rank5, real9-neighbor geometry, class-balanced replacement "
            "sampling and epoch-local4-entry queues. Two max200epoch original main/function "
            "early-stop conditions; strict VAL CE selection, complete SGD/queue/RNG/sampler "
            "recovery. Spatial-axis, undefined logging, device allocation and broken original "
            "summary repairs disclosed. Full source/wheel/native/release gates remain pending."
        ),
    )
    from milplus.carmil_spec import PROFILES as CARMIL_AUTHOR_PROFILES
    assert not ({p["id"] for p in profiles} & {p["id"] for p in CARMIL_AUTHOR_PROFILES})
    profiles.extend(json.loads(json.dumps(CARMIL_AUTHOR_PROFILES)))
    carmil_record = next(row for row in records if row["id"] == "carmil")
    carmil_record["profiles"] = ["carmil"] + [p["id"] for p in CARMIL_AUTHOR_PROFILES]
    carmil_record.update(
        status="adapted_under_verification",
        source_status="complete_original_author_recipe_repaired_candidate_under_verification",
        requirements_review=(
            "Controlled MIL_BASELINE port retained. All four original CAR backbones and L1/COS "
            "loss definitions; original100epoch Adam/cosine recipe and pre-TRAIN VAL AUC "
            "selector, including untrained epoch0 and no post-final selector. Native DSMIL "
            "instance loss and original TransMIL flattened attention retained explicitly. "
            "Private shuffle-generator/RNG/Adam/scheduler/minibatch recovery, optional "
            "original checkpoint binding; NumPy2/CPU timing/patient/locked-TEST adaptations "
            "disclosed. Full reference/source/wheel/release gates remain pending."
        ),
    )
    from milplus.mrblock_spec import PROFILES as MRBLOCK_AUTHOR_PROFILES
    assert not ({p["id"] for p in profiles} & {p["id"] for p in MRBLOCK_AUTHOR_PROFILES})
    profiles.extend(json.loads(json.dumps(MRBLOCK_AUTHOR_PROFILES)))
    mrblock_record = next(row for row in records if row["id"] == "mrblock")
    mrblock_record["profiles"] = ["mrblock"] + [p["id"] for p in MRBLOCK_AUTHOR_PROFILES]
    mrblock_record.update(
        status="adapted_under_verification", license="MIT_original_fixed_tree_read",
        source_status="pinned_original_models_and_complete_trainer_candidate",
        requirements_review=(
            "All five native MR models and five corresponding baseline controls; full200epoch "
            "AdamW/LinearLR per-update recipe, optional original VAL-CE early stop or last "
            "weights. Native merge flags/modes, nonlinear-training/linear-eval residual behavior, "
            "stochastic CATE VAL and predicted-label DGR center updates retained. Exact "
            "model/optimizer/scheduler/RNG/minibatch recovery and fixed TEST adaptation written. "
            "CATE requires actual aligned512D concept assets. Full source/wheel/native budget "
            "and current release gates remain pending."
        ),
    )
    from milplus.nnmil_spec import PROFILES as NNMIL_AUTHOR_PROFILES
    assert not ({p["id"] for p in profiles} & {p["id"] for p in NNMIL_AUTHOR_PROFILES})
    profiles.extend(json.loads(json.dumps(NNMIL_AUTHOR_PROFILES)))
    nnmil_record = next(row for row in records if row["id"] == "nnmil")
    nnmil_record["profiles"] = ["nnmil"] + [p["id"] for p in NNMIL_AUTHOR_PROFILES]
    nnmil_record.update(
        status="adapted_under_verification",
        source_status="pinned_original_classification_trainer_TRAIN_only_planning_repair_candidate",
        requirements_review=(
            "Literal SimpleMIL including full-dimension randperm and ddof1 uncertainty; "
            "native adaptive16..48batch/100epoch/10patience/bfloat16/AdamW/two-group/post-update "
            "cosine recipe. Four explicit AUC/BACC/F1/Kappa classification selectors, original "
            "workers4 and native sampler replay for exact recovery. TRAIN-only feature/label "
            "planning is a disclosed repair; TEST remains locked. Source/wheel/full native "
            "classification gates, separate survival/regression and current release gates "
            "remain pending. No original source redistribution without verified terms."
        ),
    )
    from milplus.fcmil_spec import PROFILES as FCMIL_ORIGINAL_PROFILES
    assert not ({p["id"] for p in profiles} & {p["id"] for p in FCMIL_ORIGINAL_PROFILES})
    profiles.extend(json.loads(json.dumps(FCMIL_ORIGINAL_PROFILES)))
    fcmil_record = next(row for row in records if row["id"] == "fcmil")
    fcmil_record["profiles"] = ["fcmil"] + [p["id"] for p in FCMIL_ORIGINAL_PROFILES]
    fcmil_record.update(
        status="adapted_under_verification", license="Apache-2.0_original_fixed_tree_read",
        source_status="pinned_original_model_loss_with_explicit_benchmark_recipe_and_repairs",
        requirements_review=(
            "Original1024->512 FCMIL/FAA/DAB, quadratic spatial and frequency attention, "
            "unaltered FFT patch order, literal frequency-attention top/bottom interventions "
            "and detached soft-target causal loss. Native Adam wd1e-5, plateau scheduler "
            "and strict VAL AUC selector retained. Missing author configuration is not "
            "invented:100epochs/lr2e-4/batch1/workers0 are explicit benchmark choices. "
            "Writer interface and missing real VAL AUC repaired. Full source/reference/"
            "wheel/recovery/current-release gates remain pending."
        ),
    )
    from milplus.fouriermil_spec import PROFILES as FOURIER_AUTHOR_PROFILES
    assert not ({p["id"] for p in profiles} & {p["id"] for p in FOURIER_AUTHOR_PROFILES})
    profiles.extend(json.loads(json.dumps(FOURIER_AUTHOR_PROFILES)))
    fourier_record = next(row for row in records if row["id"] == "fouriermil")
    fourier_record["profiles"] = ["fouriermil"] + [p["id"] for p in FOURIER_AUTHOR_PROFILES]
    fourier_record.update(
        status="adapted_under_verification",
        source_status="original_source_recovered_separate_native_recipe_candidate",
        requirements_review=(
            "Legacy MIL_BASELINE implementation and its source identity remain available. "
            "Separately named original-source profiles preserve native1024/512/2blocks, "
            "initialization, zero dropout, Lookahead-RAdam, eight workers, accumulation2, "
            "maximum200epochs, literal VAL loss and patience10. Native CUDA FP16 and "
            "explicit CPU32 reference are separate. Current-framework semantics and "
            "named fast/slow/cache optimizer checkpoint repair are disclosed. Complete "
            "source/wheel/native and current-release gates remain pending."
        ),
    )
    # Preserve the existing396-profile prefix; code availability is not research acceptance.
    from milplus.dyko_spec import METHOD as DYKO_METHOD, PROFILES as DYKO_PROFILES
    from milplus.dspamil_spec import METHOD as DSPA_METHOD, PROFILES as DSPA_PROFILES
    from milplus.feather_spec import METHOD as FEATHER_METHOD, PROFILES as FEATHER_PROFILES
    for method, additions in [(DYKO_METHOD, DYKO_PROFILES), (DSPA_METHOD, DSPA_PROFILES),
                              (FEATHER_METHOD, FEATHER_PROFILES)]:
        assert not ({p['id'] for p in profiles} & {p['id'] for p in additions})
        profiles.extend(json.loads(json.dumps(additions)))
        records[next(i for i, row in enumerate(records) if row['id'] == method['id'])] = json.loads(json.dumps(method))
    for name, value in [("methods.json", records), ("profiles.json", profiles)]:
        (target/name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    # Each admitted record has its own reviewable specification, including every pending method.
    specs = root / 'method_specs'
    specs.mkdir(exist_ok=True)
    for record in records:
        entry = {**record, 'execution_profiles': [p for p in profiles if p['family'] == record['id']]}
        (specs/(record['id']+'.json')).write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n')


if __name__ == "__main__":
    main()
