import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def parser():
    p = argparse.ArgumentParser(prog="milplus", description="MIL_BASELINE_PULS: explicit methods, patient-safe experiments")
    s = p.add_subparsers(dest="command", required=True)
    q = s.add_parser("methods")
    q.add_argument("--all", action="store_true", help="Include all 164 admission candidates")
    q.add_argument("--json", action="store_true")
    q.add_argument("--runnable", action="store_true", help="Only implemented execution profiles")
    q = s.add_parser("coverage", help="Count admission records separately from executable profiles")
    q.add_argument("--json", action="store_true")
    q = s.add_parser('prepare-icmil-context', help='Explicit fixed bag selection and native <=25D or TRAIN-only PCA25 reference')
    q.add_argument('--data',required=True);q.add_argument('--output',required=True)
    q.add_argument('--bag-size',type=int,required=True)
    q.add_argument('--projection',choices=['native','train_pca25_reference'],required=True)
    q.add_argument('--seed',type=int,default=2026);q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('prepare-pmil-prototypical-assets', help='Checksum original ImageNet ResNet34 weights on networked CPU')
    q.add_argument('--output',required=True)
    q = s.add_parser('demo-pmil-prototypical', help='Explicitly synthetic full raw PMIL software bags')
    q.add_argument('--output',required=True);q.add_argument('--weights',required=True);q.add_argument('--patches',type=int,default=5)
    q = s.add_parser('pack-pmil-prototypical', help='Pack actual RGB256 images and unique patch IDs without resizing')
    q.add_argument('--table',required=True);q.add_argument('--output',required=True)
    q = s.add_parser('prepare-pmil-prototypical', help='Bind actual PMIL RGB input, initialization and patient partitions')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('prepare-icmil-coupled-assets', help='Download and checksum actual original ImageNet weights on a networked CPU host')
    q.add_argument('--output',required=True)
    q = s.add_parser('demo-icmil-coupled', help='Explicitly synthetic RGB256 full-pipeline software bags')
    q.add_argument('--output',required=True);q.add_argument('--weights',required=True);q.add_argument('--patches',type=int,default=5)
    q = s.add_parser('pack-icmil-coupled', help='Pack literal PIL RGB256 patches, sorted original paths and IDs; no resizing')
    q.add_argument('--table',required=True);q.add_argument('--output',required=True)
    q = s.add_parser('prepare-icmil-coupled', help='Bind actual RGB256 images, original weights and existing patient partitions')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('demo-comel', help='Explicitly synthetic complete continual binary tasks')
    q.add_argument('--output',required=True);q.add_argument('--tasks',type=int,default=3)
    q.add_argument('--patches',type=int,default=17);q.add_argument('--dimension',type=int,default=1024)
    q = s.add_parser('prepare-comel', help='Bind ordered real tasks, patient partitions and patch identities')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('demo-macnet',help='Explicit synthetic raw RGB and actual guide roles for original full Swin/GFK MacNet')
    q.add_argument('--output',required=True);q.add_argument('--weights',required=True)
    q.add_argument('--condition',choices=['public','lscc81','cscc64'],default='public');q.add_argument('--public-unlabelled',type=int,default=32)
    q = s.add_parser('prepare-macnet',help='Bind original raw RGB, actual annotated guide origins and original22K Swin weights')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('pack-macnet',help='Losslessly pack real ordered RGB and explicit unlabelled/prior-tumor roles without resizing or sampling')
    q.add_argument('--table',required=True);q.add_argument('--output',required=True)
    q = s.add_parser('demo-mexd',help='Explicit synthetic full bags for complete MoE plus500step conditional diffusion')
    q.add_argument('--output',required=True);q.add_argument('--cohort',choices=['camelyon16','tcga_nsclc','bracs','class_port'],default='tcga_nsclc')
    q.add_argument('--dimension',type=int,default=768);q.add_argument('--patches',type=int,default=1025)
    q.add_argument('--classes',type=int);q.add_argument('--background-class0',action='store_true')
    q = s.add_parser('demo-lpgmil',help='Synthetic actual feature bags, TRAIN nucleus counts and six hierarchical prototypes')
    q.add_argument('--output',required=True);q.add_argument('--patches',type=int,default=1025)
    q.add_argument('--dimension',type=int,default=1024);q.add_argument('--classes',type=int,default=2)
    q = s.add_parser('prepare-lpgmil',help='Build TRAIN-only lymphocyte-dense hierarchical prototypes with explicit reference thresholds')
    for key in ['data','counts','detector-provenance','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--min-fraction',type=float,required=True);q.add_argument('--min-nuclei',type=int,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('prepare-promptmil-assets',help='Download and checksum original author cohort DINO on a networked CPU host')
    q.add_argument('--output',required=True);q.add_argument('--condition',choices=['brca','bright','crc'],required=True)
    q = s.add_parser('demo-promptmil',help='Explicit synthetic full224RGB bags with original author DINO; not clinical data')
    q.add_argument('--output',required=True);q.add_argument('--weights',required=True)
    q.add_argument('--condition',choices=['brca','bright','crc'],default='brca')
    q.add_argument('--patches',type=int,default=1025);q.add_argument('--classes',type=int)
    q = s.add_parser('prepare-promptmil',help='Bind originalRGB224/IDs, patient splits and original cohort DINO checkpoint')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('pack-promptmil',help='Pack actual decoded224RGB and original IDs without crop/resize')
    q.add_argument('--table',required=True);q.add_argument('--output',required=True)
    q = s.add_parser('demo-iibmil',help='Synthetic actual frozen-feature bags with full500-row sampling; no clinical result')
    q.add_argument('--output',required=True);q.add_argument('--classes',type=int,default=2)
    q.add_argument('--dimension',type=int,default=1280)
    q = s.add_parser('demo-longmil',help='Explicit synthetic actual2D feature bags; no clinical or paper-score claim')
    q.add_argument('--output',required=True);q.add_argument('--patches',type=int,default=1025)
    q.add_argument('--dimension',type=int,default=384);q.add_argument('--survival',action='store_true')
    q.add_argument('--alibi-path')
    q = s.add_parser('demo-lnplmil',help='Explicit synthetic RGB224/TRAIN-LPA data; no clinical or paper-score claim')
    q.add_argument('--output',required=True);q.add_argument('--assets',required=True);q.add_argument('--patches',type=int,default=401)
    q = s.add_parser('prepare-lnplmil',help='Bind actual RGB224, observed TRAIN patch labels, patient splits and original ImageNet ResNet18')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('demo-hybrid-supervision',help='Synthetic full512RGB/fine masks; never clinical data')
    q.add_argument('--output',required=True);q.add_argument('--weights',required=True);q.add_argument('--patches',type=int,default=16)
    q = s.add_parser('prepare-hybrid-supervision',help='Bind actual512RGB/TRAINpixelmasks/DLA34 and patient splits')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('demo-protomil',help='Explicit synthetic512D CONCH-shaped features and actual synthetic RGB/mask probes; not clinical data')
    q.add_argument('--output',required=True);q.add_argument('--patches',type=int,default=17);q.add_argument('--probe-per-class',type=int,default=5)
    q = s.add_parser('prepare-protomil',help='Bind TRAIN-only SAE populations, actual ordered CONCH features, region-annotated probes and concept review')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('demo-kmpmil',help='Explicit synthetic original four-task full CONCH KMP/PFC/TC fixture')
    for key in ['output','weights','sha256']:q.add_argument('--'+key,required=True)
    q.add_argument('--patches',type=int,default=17);q.add_argument('--train-per-class',type=int,default=9)
    q = s.add_parser('prepare-kmpmil',help='Bind actual patients/tasks/patch IDs/CONCH/class texts/semantic descriptors')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('demo-qpmilvl',help='Explicit synthetic original four-task full CONCH prompt fixture')
    for key in ['output','weights','sha256']:q.add_argument('--'+key,required=True)
    q.add_argument('--patches',type=int,default=17);q.add_argument('--train-per-class',type=int,default=8)
    q = s.add_parser('prepare-qpmilvl',help='Bind actual patients/tasks/patch IDs/CONCH/text ensemble')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('demo-akdpmp',help='Explicit synthetic complete author continual recipes')
    q.add_argument('--output',required=True);q.add_argument('--preset',choices=['tcga','skin'],default='tcga')
    q.add_argument('--patches',type=int,default=41)
    q = s.add_parser('prepare-akdpmp',help='Bind actual ordered tasks, patient partitions, FM and original patch IDs')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser("setup", help="Fetch pinned author sources for an adapter")
    q.add_argument("--method", required=True)
    q.add_argument("--plan", action="store_true", help="Show complete pinned source and existing public-asset preparation without downloading or writing")
    q = s.add_parser('prepare-dspamil', help='Bind native UNI, explicit concept alignment and TRAIN-only measured pattern targets for the DSPA2025 reference')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('prepare-feather-assets', help='Verify a locally available original FEATHER24K checkpoint without downloading restricted assets')
    q.add_argument('--encoder',choices=['conch_v15','uni','uni_v2'],required=True)
    q.add_argument('--checkpoint',required=True);q.add_argument('--output',required=True)
    q = s.add_parser('prepare-feather', help='Bind native FEATHER24K visual features, original pretrained checkpoint and patient partitions')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('prepare-dyko-assets', help='Bind user-supplied original TITAN model/config/tokenizer to published file identities')
    q.add_argument('--titan-directory',required=True);q.add_argument('--output',required=True)
    q = s.add_parser('prepare-dyko', help='Bind actual native768D aligned features, source concepts and class/patient identities')
    for key in ['data','schema','output']:q.add_argument('--'+key,required=True)
    q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('demo-rmdl', help='Create explicitly synthetic full1899 spatial slides and TRAIN-only pixel masks')
    q.add_argument('--output',required=True);q.add_argument('--weights',required=True)
    q = s.add_parser('demo-attention2majority', help='Create explicitly synthetic native RGB software-validation bags')
    q.add_argument('--output',required=True);q.add_argument('--weights',required=True)
    q.add_argument('--classes',type=int,choices=[2,3],default=2);q.add_argument('--patches',type=int,default=17)
    q = s.add_parser('prepare-attention2majority', help='Build the Appendix-I TRAIN pool from complete original Keras ImageNet features')
    for key in ['data','output','template-slide','normal-reference']:
        q.add_argument('--'+key,required=True)
    q.add_argument('--component-space',choices=['unmixing_rows','mixing_columns'],required=True)
    q.add_argument('--fm');q.add_argument('--split');q.add_argument('--tensorflow-python')
    q.add_argument('--seed',type=int,default=2026);q.add_argument('--threads',type=int,default=2)
    q.add_argument('--device',default='cpu')
    q = s.add_parser('prepare-nciemil', help='Train full Swin-Tiny on TRAIN bags and bind all-patch highK/lowK selected features')
    for key in ['data','weights','output','pretrain-config']:
        q.add_argument('--'+key,required=True)
    q.add_argument('--k',type=int,choices=[128,512],required=True)
    q.add_argument('--device',default='cpu')
    q.add_argument('--resume',action='store_true')
    q = s.add_parser('prepare-muse', help='Bind full original600-description knowledge and complete source feature bags')
    for key in ['data', 'metadata', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-libramil', help='Bind complete aligned512D bags and actual pretrained ordered language descriptions')
    for key in ['data', 'metadata', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-convl', help='Bind native CONCH patient survival, complete author concepts and explicit original or TRAIN-derived bins')
    for key in ['data', 'metadata', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('pack-center-embeddings', help='Read original WSI center coordinates into exact256RGB source tiles')
    q.add_argument('--table', required=True); q.add_argument('--output', required=True)
    q.add_argument('--level', type=int, default=2); q.add_argument('--mult', type=float, default=1)
    q = s.add_parser('prepare-center-embeddings', help='Bind original raw256RGB, measured centers and historical assets')
    for key in ['data', 'schema', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('import-miladapter-prototypes', help='Import exact public author text prototypes, without claiming regeneration')
    q.add_argument('--output', required=True)
    q.add_argument('--project', choices=['BRCA', 'NSCLC', 'RCC'], default='BRCA')
    q.add_argument('--encoder', choices=['CONCH', 'KEEP'], default='CONCH')
    q = s.add_parser('prepare-miladapter', help='Bind author text prototypes and patient-disjoint TRAIN support pool')
    for key in ['data', 'metadata', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-conmil-text', help='Encode actual original CONCH positive/negative prompts with full causal CLS text tower')
    for key in ['prompts', 'weights', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--device', default='cpu')
    q = s.add_parser('prepare-conmil', help='Bind native visual/text features, labels and TRAIN-only prompt review')
    for key in ['data', 'metadata', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('pack-sparseconvmil', help='Pack actual original x_y RGB tile folders with no resizing')
    q.add_argument('--table', required=True); q.add_argument('--output', required=True)
    q = s.add_parser('prepare-sparseconvmil', help='Bind original RGB tiles and actual spatial coordinates')
    for key in ['data', 'schema', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('pack-c2c', help='Pack actual original path/wsi RGB tiles without resizing or normalization')
    q.add_argument('--table', required=True); q.add_argument('--output', required=True)
    q = s.add_parser('prepare-c2c', help='Bind original raw RGB patch bags and original historical ResNet18 assets')
    for key in ['data', 'schema', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-sammil', help='Bind actual SAM memberships and original areas to patch features')
    for key in ['data', 'index', 'schema', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-lrmil', help='Bind measured LR RGB and all native CONCH teacher intermediate layers')
    for key in ['data', 'index', 'schema', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-hdmil', help='Bind measured low-RGB patches to high-resolution frozen feature rows')
    for key in ['data', 'index', 'schema', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-kmgmamba', help='Bind genuine measured two-scale KMG-Mamba features')
    for key in ['data', 'index', 'schema', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-mgtrans' , help='Bind measured dual-scale features and original physical coverage')
    for key in ['data', 'index', 'schema', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-zoommil', help='Bind actual three-scale FM features and complete source recursive spatial addresses')
    for key in ['data', 'index', 'schema', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('pretrain-augdiff', help='Train the complete source-width DAE on registered TRAIN-only seven image views')
    for key in ['data', 'views-manifest', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--epochs', type=int, required=True, help='Explicit DAE budget; public paper/YAML do not specify epochs')
    q.add_argument('--batch-size', type=int, default=1200)
    q.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    q.add_argument('--device', default='cpu'); q.add_argument('--seed', type=int, default=2026)
    q.add_argument('--threads', type=int, default=1)
    q.add_argument('--stop-after', type=int); q.add_argument('--resume')
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-augdiff', help='Bind selected RAW DAE weights from a completed full-width pretraining run')
    for key in ['data', 'views-manifest', 'pretrain-run', 'output']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-h2mil',help='Bind actual thumbnail/coarse/fine feature pyramids, physical grids and real parent relations')
    for key in ['data','index','schema','output']:
        q.add_argument('--'+key,required=True)
    q.add_argument('--fm')
    q.add_argument('--split')
    q = s.add_parser('prepare-hipt-weights',help='Export the actual published HIPT teacher backbone to a small weights-only file')
    q.add_argument('--checkpoint',required=True)
    q.add_argument('--output',required=True)
    q = s.add_parser('prepare-hipt',help='Bind supplied complete physical 16x16 HIPT regions and native teacher weights')
    for key in ['data','index','schema','weights','output']:
        q.add_argument('--'+key,required=True)
    q.add_argument('--fm')
    q.add_argument('--split')
    q = s.add_parser('prepare-simil', help='Bind supplied patch-aligned raw PathExpert descriptors; no descriptor fabrication')
    for key in ['data','index','schema','output']:
        q.add_argument('--'+key,required=True)
    q.add_argument('--fm')
    q.add_argument('--split')
    q = s.add_parser('prepare-ttis', help='Bind exact TEST physical clusters and frozen semantic vectors')
    for key in ['data', 'output', 'semantic-metadata']:
        q.add_argument('--'+key, required=True)
    q.add_argument('--cluster-backend', required=True, choices=['cuml', 'sklearn_cpu'])
    q.add_argument('--clusters', type=int, default=256)
    q.add_argument('--n-init', type=int, default=10)
    q.add_argument('--cluster-seed', type=int, default=2025)
    q.add_argument('--coordinate-frame', choices=['level0_pixels', 'micrometers'], default='level0_pixels')
    q.add_argument('--guidance-index', help='Optional row-aligned secondary semantic visual feature CSV')
    q.add_argument('--fm')
    q.add_argument('--split')
    q = s.add_parser('prepare-mammil-graphs', help='Prepare frozen MamMIL topology assets on a CPU host')
    q.add_argument('--data', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--fm')
    q.add_argument('--split')
    q.add_argument('--graph-cache')
    q.add_argument('--neighbors', type=int, default=8)
    q.add_argument('--graph-seed', type=int, default=2024)
    q = s.add_parser('prepare-smmile', help='Build input-bound SMMILe superpixels from real patch coordinates')
    q.add_argument('--data', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--fm')
    q.add_argument('--split')
    q.add_argument('--patch-stride', type=float, required=True)
    q.add_argument('--coordinate-frame', choices=['level0_pixels', 'micrometers'], default='level0_pixels')
    q.add_argument('--patches-per-superpixel', type=int, default=16)
    q.add_argument('--compactness', type=float, default=50.)
    q.add_argument('--provenance', required=True)
    q.add_argument('--max-grid-cells', type=int, default=1_000_000)
    q.add_argument('--max-dense-bytes', type=int, default=512*1024**2)
    q = s.add_parser('prepare-pamil', help='Bind supplied TRAIN instance labels to the selected feature/coordinate order')
    q.add_argument('--data', required=True)
    q.add_argument('--annotations', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--provenance', required=True)
    q.add_argument('--fm')
    q.add_argument('--split')
    q = s.add_parser('prepare-text', help='Export local CATE concepts or RN50 text state and tokenize descriptors')
    q.add_argument('--kind', choices=['concept_anchors', 'learned_dual_scale_prompts'], required=True)
    q.add_argument('--features', required=True, help='Safe tensor .pt file; concepts list or CLIP state_dict')
    q.add_argument('--metadata', required=True)
    q.add_argument('--output', required=True)
    q = s.add_parser('prepare-tissues', help='Build PAMoE tissue means from aligned training-only patch assignments')
    for flag in ('data', 'assignments', 'output', 'provenance'):
        q.add_argument('--'+flag, required=True)
    q.add_argument('--names', nargs='+', required=True)
    q.add_argument('--fm')
    q.add_argument('--split')
    q = s.add_parser('prepare-focus', help='Bind native pretrained CONCH, actual ordered descriptions and stable feature rows')
    for name in ['data', 'text-metadata', 'output']:
        q.add_argument('--'+name, required=True)
    q.add_argument('--fm'); q.add_argument('--split')
    q = s.add_parser('prepare-codmil', help='Bind actual coarse/fine physical maps and ordered diagnosis/normal text embeddings')
    for name in ['data', 'index', 'schema', 'text-metadata', 'output']:
        q.add_argument('--'+name, required=True)
    q.add_argument('--fm')
    q.add_argument('--split')
    q = s.add_parser('prepare-deepgraphsurv', help='Freeze original sampled feature-space graphs and literal intrinsic Laplacians')
    q.add_argument('--data', required=True);q.add_argument('--schema', required=True)
    q.add_argument('--output', required=True);q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('prepare-triagemil', help='Freeze actual MASS selection and source semantic hierarchical hypergraphs')
    q.add_argument('--data', required=True);q.add_argument('--schema', required=True)
    q.add_argument('--output', required=True);q.add_argument('--fm');q.add_argument('--split')
    q = s.add_parser('prepare-c2mil', help='Bind actual WSI thumbnails, physical graphs and TRAIN1500patch reference pool')
    q.add_argument('--data', required=True);q.add_argument('--index', required=True)
    q.add_argument('--schema', required=True);q.add_argument('--output', required=True)
    q.add_argument('--fm');q.add_argument('--split');q.add_argument('--sampling-rng-state')
    q = s.add_parser('prepare-gatmamba', help='Freeze actual physical8NN/subtype/feature-distance patient graphs')
    for name in ['data','index','schema','output']:
        q.add_argument('--'+name, required=True)
    q.add_argument('--fm')
    q.add_argument('--split')
    q = s.add_parser('prepare-protosurv', help='Bind real coordinates and tissue assignments to frozen source spatial graphs')
    for name in ['data','index','schema','output']:
        q.add_argument('--'+name, required=True)
    q.add_argument('--prototypes')
    q.add_argument('--prototype-provenance')
    q.add_argument('--fm')
    q.add_argument('--split')
    q=s.add_parser('prepare-slip-assets',help='Download and checksum the actual original CLIP text assets on a networked CPU host')
    q.add_argument('--output',required=True)
    q.add_argument('--architecture',action='append',choices=['clip_vit_b16','clip_rn50'])
    q=s.add_parser('prepare-taxomil-assets',help='Download and checksum the complete original GPT2 decoder/tokenizer on a networked CPU host')
    q.add_argument('--output',required=True)
    q=s.add_parser('prepare-dispro-assets',help='Download and checksum the fixed complete public BioBERT revision on a networked CPU host')
    q.add_argument('--output',required=True)
    q = s.add_parser("inspect")
    q.add_argument("--method", required=True)
    q = s.add_parser("demo")
    q.add_argument('--taxomil',action='store_true',help='Synthetic full1024D bags, real coarse/fine taxonomy and full original GPT2')
    q.add_argument('--taxomil-assets')
    q.add_argument('--taxomil-dataset',choices=['BRACS','PANDA'],default='BRACS')
    q.add_argument('--gmmamba',action='store_true',help='Full native512D synthetic spatial bags for complete GMMamba')
    q.add_argument('--gmmamba-dimension',type=int,default=512)
    q.add_argument('--muse', action='store_true', help='Full native-dimensional synthetic MUSE input and real knowledge banks')
    q.add_argument('--muse-weights')
    q.add_argument('--muse-cohort', default='tcga_brca')
    q.add_argument('--muse-dimension', type=int, default=512)
    q.add_argument('--libramil', action='store_true', help='Synthetic native512D classification bags with full original pretrained language descriptions')
    q.add_argument('--libramil-weights')
    q.add_argument('--m4',action='store_true',help='Synthetic complete multi-endpoint bags for full original M4')
    q.add_argument('--m4-dimension',type=int,default=2048)
    q.add_argument('--m4-endpoints',type=int,default=10)
    q.add_argument('--m4-patches',type=int)
    q.add_argument('--convl', action='store_true', help='Synthetic native512 patient bags with actual full pretrained CONCH text and original ConVL concepts')
    q.add_argument('--convl-weights', help='Actual original CONCH checkpoint, required for the ConVL fixture')
    q.add_argument('--convl-cohort', default='BRCA', help='Original source concept cohort, e.g. BRCA or COADREAD')
    q.add_argument('--center-embeddings', action='store_true', help='Synthetic full original32x256RGB/FPN/center-loss input')
    q.add_argument('--center-assets', help='Exact original ResNet34 and torchvision0.7 asset root')
    q.add_argument('--miladapter', action='store_true', help='Synthetic full512D bags with actual public author text prototypes')
    q.add_argument('--miladapter-project', choices=['BRCA', 'NSCLC', 'RCC'], default='BRCA')
    q.add_argument('--miladapter-encoder', choices=['CONCH', 'KEEP'], default='CONCH')
    q.add_argument('--conmil', action='store_true', help='Explicit synthetic native512 visual bags with actual encoded CONCH text')
    q.add_argument('--conmil-text', help='Existing actual prepare-conmil-text output text.json for this synthetic fixture')
    q.add_argument('--sparseconvmil', action='store_true', help='Explicit synthetic spatial RGB bags for the full source SparseConvMIL network')
    q.add_argument('--c2c', action='store_true', help='Explicit synthetic raw RGB patch bags with actual original ResNet18 weights')
    q.add_argument('--sammil', action='store_true', help='Explicit synthetic SAM-membership fixture; no actual segmentation')
    q.add_argument('--lrmil', action='store_true', help='Synthetic measured LR RGB and three native-width HR layers; actual CONCH initialization')
    q.add_argument('--hdmil', action='store_true', help='Synthetic matched RGB16/feature256 inputs; no real WSI')
    q.add_argument('--kmgmamba', action='store_true', help='Synthetic complete KMG-Mamba two-scale fixture; no real WSI')
    q.add_argument('--mgtrans', action='store_true', help='Synthetic complete two-scale MG-Trans source inputs; no real WSI/FM extraction')
    q.add_argument('--zoommil', action='store_true', help='Synthetic actual complete three-scale source hierarchy; no real FM extraction')
    q.add_argument('--zoommil-geometry', choices=['bright','crc'], default='bright')
    q.add_argument('--augdiff', action='store_true', help='Synthetic full-dimensional seven-view DAE preparation fixture; not real image augmentation')
    q.add_argument('--augdiff-dimension', type=int, default=512)
    q.add_argument('--augdiff-instances', type=int, default=3)
    q.add_argument('--vlsa',action='store_true',help='Synthetic native512 VLSA patient survival bags with actual pretrained CONCH text')
    q.add_argument('--vlsa-weights',help='Actual original CONCH checkpoint path, required with --vlsa')
    q.add_argument('--vlsa-query-key',default='tcga_brca_0')
    q.add_argument('--slip',action='store_true',help='Synthetic native-width SLIP features and ordered class/tissue prompts')
    q.add_argument('--slip-architecture',choices=['clip_vit_b16','clip_rn50'],default='clip_vit_b16')
    q.add_argument('--dispro',action='store_true',help='Explicit incomplete patient WSI/RNA survival fixture with all331 public pathway sizes')
    q.add_argument('--amiml', action='store_true', help='Synthetic full four-morphology patient bags for AMIML')
    q.add_argument('--dtmil', action='store_true', help='Synthetic actual physical-grid DT-MIL feature maps')
    q.add_argument('--pamil-policy', action='store_true', help='Synthetic stable-order feature bags for native PAMIL dynamic-policy training')
    q.add_argument('--focus', action='store_true', help='Synthetic ordered bags/prompts with actual full pretrained CONCH text; requires MILPLUS_FOCUS_WEIGHTS')
    q.add_argument('--codmil', action='store_true', help='Synthetic complete physical CoD scales, row map and class/normal text fixture')
    q.add_argument('--ins', action='store_true', help='Synthetic binary slide bags with full1024-dimensional INS feature input')
    q.add_argument('--sclwc', action='store_true', help='Synthetic slide classification with explicit normal-tissue class roles')
    q.add_argument('--deepgraphsurv', action='store_true', help='Synthetic original adaptive feature-graph survival fixture')
    q.add_argument('--triagemil', action='store_true', help='Synthetic real-coordinate MASS and semantic hypergraph survival fixture')
    q.add_argument('--c2mil', action='store_true', help='Synthetic graph survival with explicit TRAIN references and thumbnail fixtures')
    q.add_argument('--gatmamba',action='store_true',help='Synthetic multi-slide GAT-Mamba survival graphs with explicit subtype fixtures')
    q.add_argument('--protosurv',action='store_true',help='Synthetic physical survival graphs and explicit tissue-category assets')
    q.add_argument('--h2mil',action='store_true',help='Synthetic three-level physical H2-MIL grids with separate Gaussian image-level features')
    q.add_argument('--hipt',action='store_true',help='Synthetic complete native HIPT physical pyramids, with an actual supplied teacher backbone')
    q.add_argument('--raw-images', action='store_true', help='Synthetic complete RGB image levels and ordinal grades for native MONAI training')
    q.add_argument('--raw-regions', action='store_true', help='Synthetic tiled WSI and real ROI grids for full Campanella two-stage training')
    q.add_argument('--backbone-weights', help='Actual HIPT teacher (--hipt), or official ResNet50/ResNet34 ImageNet1K_V1 checkpoint (--raw-images/--raw-regions)')
    q.add_argument("--output", required=True)
    q.add_argument("--patients", type=int, default=36)
    q.add_argument("--classes", type=int, default=2)
    q.add_argument("--survival", action="store_true")
    q.add_argument("--regression", action="store_true")
    q.add_argument('--vision-language', choices=['concept_anchors', 'learned_dual_scale_prompts'])
    q.add_argument('--smmile', choices=['binary', 'multiclass', 'multilabel'])
    q.add_argument("--omics", action="store_true", help="Include synthetic patient molecular vectors and six signatures")
    q.add_argument('--multiscale', choices=['aligned_centers', 'quadtree'])
    q = s.add_parser("run-matrix")
    q.add_argument("--plan", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--devices", nargs="+", default=["cpu"])
    q.add_argument("--config")
    q.add_argument("--epochs", type=int)
    q = s.add_parser("report")
    q.add_argument("--runs", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--subset", choices=["val", "test"], default="test")
    for cmd in ("check", "run"):
        q = s.add_parser(cmd)
        q.add_argument("--data", required=True)
        q.add_argument("--fm")
        q.add_argument("--split")
        q.add_argument("--method", required=cmd == "run")
        if cmd == "check":
            q.add_argument("--deep", action="store_true")
        else:
            q.add_argument("--output")
            q.add_argument("--seed", type=int, default=2026)
            q.add_argument("--device", default="cpu")
            q.add_argument("--config")
            q.add_argument("--epochs", type=int)
            q.add_argument("--resume", action="store_true")
            q.add_argument("--stop-after",type=int,help="Interrupt after actual method progress units (ProtoMIL: SAE WSI pieces/concept step/MIL epochs), preserving recovery state")
    q = s.add_parser("evaluate")
    q.add_argument("--run", required=True)
    q.add_argument("--subset", choices=["val", "test"], default="test")
    q.add_argument("--device", default="cpu")
    q = s.add_parser("compare")
    q.add_argument("--left", required=True)
    q.add_argument("--right", required=True)
    q.add_argument("--classes", required=True, type=int)
    q.add_argument("--metric", default="macro_auc")
    q.add_argument("--unit", choices=["patient", "slide"], default="patient")
    q.add_argument("--repetitions", type=int, default=2000)
    q.add_argument("--output", required=True)
    q = s.add_parser("matrix")
    q.add_argument("--data", required=True)
    q.add_argument("--methods", nargs="+", required=True)
    q.add_argument("--fms", nargs="+", required=True)
    q.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    q.add_argument("--output", required=True)
    q = s.add_parser("prepare-tcga")
    q.add_argument("--inventory", required=True)
    q.add_argument("--project", choices=["TCGA-BRCA", "TCGA-COAD"], required=True)
    q.add_argument("--labels", nargs="+", required=True)
    for subset in ("train", "val", "test"):
        q.add_argument(f"--{subset}-csv", required=True)
    q.add_argument("--output", required=True)
    q = s.add_parser("index-features")
    for option in ("data", "feature-dir", "fm", "weights", "preprocessing"):
        q.add_argument("--" + option, required=True)
    q.add_argument("--dimension", type=int, required=True)
    q.add_argument("--coords-dir")
    q = s.add_parser("trident")
    for option in ("repo", "wsi-dir", "output", "fm"):
        q.add_argument("--" + option, required=True)
    q.add_argument("--python", default=sys.executable)
    q.add_argument("--mag", type=float, default=20)
    q.add_argument("--patch-size", type=int, default=256)
    q.add_argument("--checkpoint")
    q.add_argument("--custom-list")
    q.add_argument("--execute", action="store_true")
    q = s.add_parser("fetch-source")
    q.add_argument("--source", required=True)
    q.add_argument("--destination", required=True)
    q = s.add_parser("hcluster")
    for option in ("repository", "data", "method", "output", "image", "script"):
        q.add_argument("--" + option, required=True)
    q.add_argument("--project", default="ailab-medpeak")
    q.add_argument("--group", default="medai4s_gpu_pool")
    q.add_argument("--memory-mib", type=int, default=32768)
    q.add_argument("--cpu", type=int, default=8)
    q.add_argument("--gpu", type=int, default=1)
    q.add_argument("--submit", action="store_true")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        from .akdpmp_runtime import maybe_restart
        restarted = maybe_restart(a, argv)
        if restarted is not None:
            return restarted
        result = dispatch(a)
        if result is not None:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str, allow_nan=False))
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"milplus: {e}", file=sys.stderr)
        raise SystemExit(2)


def dispatch(a):
    if a.command=="demo-mexd":
        from .mexd_demo import create_demo
        return {"data":str(create_demo(a.output,a.cohort,a.dimension,a.patches,a.classes,a.background_class0))}
    if a.command=="demo-lpgmil":
        from .lpgmil_demo import create_demo
        return {"data":str(create_demo(a.output,a.patches,a.dimension,a.classes))}
    if a.command=="prepare-lpgmil":
        from .lpgmil_data import prepare
        from .data import Bundle
        return {"manifest":str(prepare(Bundle(a.data,a.fm,a.split),a.counts,a.detector_provenance,a.output,a.min_fraction,a.min_nuclei))}
    if a.command=='prepare-promptmil-assets':
        from .promptmil_assets import fetch
        return {'weights':str(fetch(a.output,a.condition))}
    if a.command=='demo-promptmil':
        from .promptmil_demo import create_demo
        return {'data':str(create_demo(a.output,a.weights,a.condition,a.patches,a.classes))}
    if a.command=='prepare-promptmil':
        from .promptmil_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='pack-promptmil':
        from .promptmil_data import pack
        return {'index':str(pack(a.table,a.output))}
    if a.command=='demo-iibmil':
        from .iibmil_demo import create_demo
        return {'data':str(create_demo(a.output,a.classes,a.dimension))}
    if a.command=='demo-longmil':
        from .longmil_demo import create_demo
        return {'data':str(create_demo(a.output,a.patches,a.dimension,a.survival,a.alibi_path))}
    if a.command=='demo-lnplmil':
        from .lnplmil_demo import create_demo
        return {'data':str(create_demo(a.output,a.assets,a.patches))}
    if a.command=='prepare-lnplmil':
        from .lnplmil_raw import prepare_bundle
        from .data import Bundle
        return {'manifest':str(prepare_bundle(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='demo-hybrid-supervision':
        from .hybrid_supervision_demo import create_demo
        return {'data':str(create_demo(a.output,a.weights,a.patches))}
    if a.command=='prepare-hybrid-supervision':
        from .hybrid_supervision_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='demo-macnet':
        from .macnet_demo import create_demo
        return {'data':str(create_demo(a.output,a.weights,a.condition,a.public_unlabelled))}
    if a.command=='prepare-macnet':
        from .macnet_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='pack-macnet':
        from .macnet_pack import pack
        return {'index':str(pack(a.table,a.output))}
    if a.command=='demo-protomil':
        from .protomil_demo import create_demo
        return {'data':str(create_demo(a.output,a.patches,a.probe_per_class))}
    if a.command=='prepare-protomil':
        from .protomil_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='demo-kmpmil':
        from .kmpmil_demo import create_demo
        return {'data':str(create_demo(a.output,a.weights,a.sha256,a.patches,a.train_per_class))}
    if a.command=='prepare-kmpmil':
        from .kmpmil_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='demo-qpmilvl':
        from .qpmilvl_demo import create_demo
        return {'data':str(create_demo(a.output,a.weights,a.sha256,a.patches,a.train_per_class))}
    if a.command=='prepare-qpmilvl':
        from .qpmilvl_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='demo-akdpmp':
        from .akdpmp_demo import create_demo
        return {'data':str(create_demo(a.output,a.preset,a.patches))}
    if a.command=='prepare-akdpmp':
        from .akdpmp_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='demo-comel':
        from .comel_demo import create_demo
        return {'data':str(create_demo(a.output,a.tasks,a.patches,a.dimension))}
    if a.command=='prepare-comel':
        from .comel_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='prepare-pmil-prototypical-assets':
        from .pmil_prototypical_assets import fetch
        return {'weights':str(fetch(a.output))}
    if a.command=='demo-pmil-prototypical':
        from .pmil_prototypical_demo import create_demo
        return {'data':str(create_demo(a.output,a.weights,a.patches))}
    if a.command=='pack-pmil-prototypical':
        from .icmil_coupled_pack import pack
        return {'index':str(pack(a.table,a.output))}
    if a.command=='prepare-pmil-prototypical':
        from .pmil_prototypical_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='prepare-icmil-coupled-assets':
        from .icmil_coupled_assets import fetch
        return {'weights':str(fetch(a.output))}
    if a.command=='demo-icmil-coupled':
        from .icmil_coupled_demo import create_demo
        return {'data':str(create_demo(a.output,a.weights,a.patches))}
    if a.command=='pack-icmil-coupled':
        from .icmil_coupled_pack import pack
        return {'index':str(pack(a.table,a.output))}
    if a.command=='prepare-icmil-coupled':
        from .icmil_coupled_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    if a.command=='prepare-icmil-context':
        from .icmil_context_prepare import prepare
        return {'data':str(prepare(a.data,a.output,a.bag_size,a.projection,a.seed,a.fm,a.split))}
    if a.command=='prepare-taxomil-assets':
        from .taxomil_assets import fetch
        return {'assets':str(fetch(a.output))}
    if a.command == 'prepare-muse':
        from .muse_data import prepare
        from .data import Bundle
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),a.metadata,a.output))}
    if a.command == 'prepare-libramil':
        from .libramil_data import prepare
        from .data import Bundle
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), a.metadata, a.output))}
    if a.command == 'prepare-convl':
        from .convl_data import prepare
        from .data import Bundle
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), a.metadata, a.output))}
    if a.command == 'pack-center-embeddings':
        from .center_embeddings_pack import pack
        return {'image_index': str(pack(a.table, a.output, a.level, a.mult))}
    if a.command == 'prepare-center-embeddings':
        from .center_embeddings_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), read_json(a.schema), a.output))}
    if a.command == 'import-miladapter-prototypes':
        from .miladapter_data import import_author_prototypes
        return {'text_assets': str(import_author_prototypes(a.output, a.project, a.encoder))}
    if a.command == 'prepare-miladapter':
        from .miladapter_data import prepare
        from .data import Bundle
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), a.metadata, a.output))}
    if a.command == 'prepare-conmil-text':
        from .conmil_text import prepare_text
        return {'text_assets': str(prepare_text(a.prompts, a.weights, a.output, a.device))}
    if a.command == 'prepare-conmil':
        from .conmil_data import prepare
        from .data import Bundle
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), a.metadata, a.output))}
    if a.command == 'pack-sparseconvmil':
        from .sparseconvmil_pack import pack
        return {'image_index': str(pack(a.table, a.output))}
    if a.command == 'prepare-sparseconvmil':
        from .sparseconvmil_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), read_json(a.schema), a.output))}
    if a.command == 'pack-c2c':
        from .c2c_pack import pack
        return {'image_index': str(pack(a.table, a.output))}
    if a.command == 'prepare-c2c':
        from .c2c_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), read_json(a.schema), a.output))}
    if a.command == 'prepare-sammil':
        from .sammil_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), a.index, read_json(a.schema), a.output))}
    if a.command == 'prepare-lrmil':
        from .lrmil_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), a.index, read_json(a.schema), a.output))}
    if a.command == 'prepare-hdmil':
        from .hdmil_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), a.index, read_json(a.schema), a.output))}
    if a.command == 'prepare-kmgmamba':
        from .kmgmamba_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), a.index, read_json(a.schema), a.output))}
    if a.command == 'prepare-mgtrans':
        from .mgtrans_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest': str(prepare(Bundle(a.data, a.fm, a.split), a.index, read_json(a.schema), a.output))}
    if a.command == 'prepare-zoommil':
        from .zoommil_data import prepare
        from .data import Bundle
        from .io import read_json
        return {'manifest': str(prepare(Bundle(a.data,a.fm,a.split),a.index,read_json(a.schema),a.output))}
    if a.command == 'pretrain-augdiff':
        import torch

        from .augdiff_data import AugDiffInputs
        from .augdiff_pretrain import DiffusionTrainer
        from .data import Bundle
        if a.threads < 1:
            raise ValueError('AugDiff CPU thread count must be positive')
        torch.set_num_threads(a.threads)
        assets = AugDiffInputs(Bundle(a.data, a.fm, a.split), a.views_manifest)
        trainer = DiffusionTrainer(assets, a.epochs, a.device, a.batch_size, a.precision, a.seed)
        return trainer.fit(a.output, a.stop_after, a.resume)
    if a.command == 'prepare-augdiff':
        from .augdiff_assets import export_trained_dae
        from .data import Bundle
        return {'manifest': export_trained_dae(Bundle(a.data, a.fm, a.split), a.views_manifest, a.pretrain_run, a.output)}
    if a.command == 'prepare-focus':
        from .data import Bundle
        from .focus_data import prepare
        return {'manifest': prepare(Bundle(a.data, a.fm, a.split), a.text_metadata, a.output)}
    if a.command == 'prepare-codmil':
        from .codmil_data import prepare
        from .data import Bundle
        return {'manifest': prepare(Bundle(a.data, a.fm, a.split), a.index, a.schema, a.text_metadata, a.output)}
    from .io import write_json
    from .registry import inspect_method, methods, profiles
    if a.command == 'prepare-deepgraphsurv':
        from .data import Bundle
        from .deepgraphsurv_data import prepare
        from .io import read_json
        return str(prepare(Bundle(a.data,a.fm,a.split),read_json(a.schema),a.output))
    if a.command == 'prepare-triagemil':
        from .data import Bundle
        from .io import read_json
        from .triagemil_data import prepare
        return str(prepare(Bundle(a.data,a.fm,a.split),read_json(a.schema),a.output))
    if a.command == 'prepare-c2mil':
        from .c2mil_data import prepare
        from .data import Bundle
        from .io import read_json
        return str(prepare(Bundle(a.data,a.fm,a.split),a.index,read_json(a.schema),a.output,a.sampling_rng_state))
    if a.command == 'prepare-gatmamba':
        from .data import Bundle
        from .gatmamba_data import prepare
        from .io import read_json
        return str(prepare(Bundle(a.data,a.fm,a.split), a.index, read_json(a.schema), a.output))
    if a.command == 'prepare-protosurv':
        from .data import Bundle
        from .io import read_json
        from .protosurv_data import prepare
        return str(prepare(Bundle(a.data,a.fm,a.split), a.index, read_json(a.schema), a.output,
            a.prototypes, read_json(a.prototype_provenance) if a.prototype_provenance else None))
    if a.command == "methods":
        rows = profiles() if a.runnable else methods()
        if a.json:
            return rows
        for r in rows:
            print(f"{r['id']:<24} {r['status']:<26} {r.get('name', r.get('recipe', ''))}")
        print(f"{len(rows)} {'runnable profiles' if a.runnable else 'admission records'}; paper reproduction status is separate.")
    elif a.command == "coverage":
        from .registry import coverage
        return coverage()
    elif a.command == "setup":
        from .setup_sources import prepare_sources
        return prepare_sources(a.method, plan_only=a.plan)
    elif a.command=='prepare-h2mil':
        from .data import Bundle
        from .h2mil_data import prepare
        from .io import read_json
        path=prepare(Bundle(a.data,a.fm,a.split),a.index,read_json(a.schema),a.output)
        return {'manifest':str(path),'registration':'Set encoders.<FM>.h2mil.manifest in task.yaml to this path'}
    elif a.command=='prepare-hipt-weights':
        from .hipt_source import export_backbone
        return {'backbone':str(export_backbone(a.checkpoint,a.output))}
    elif a.command=='prepare-hipt':
        from .data import Bundle
        from .hipt_data import prepare
        from .io import read_json
        path=prepare(Bundle(a.data,a.fm,a.split),a.index,read_json(a.schema),a.weights,a.output)
        return {'manifest':str(path),'registration':'Set encoders.<FM>.hipt.manifest in task.yaml to this path'}
    elif a.command == 'prepare-simil':
        from .data import Bundle
        from .io import read_json
        from .simil_data import prepare
        path=prepare(Bundle(a.data,a.fm,a.split),a.index,read_json(a.schema),a.output)
        return {'manifest':str(path),'registration':'Set encoders.<FM>.simil.manifest in task.yaml to this path'}
    elif a.command == 'prepare-ttis':
        import pandas as pd

        from .data import Bundle, resolve
        from .ttis_data import prepare_ttis
        bundle = Bundle(a.data, a.fm, a.split)
        guidance = None
        if a.guidance_index:
            index = Path(a.guidance_index).resolve()
            frame = pd.read_csv(index, dtype=str, keep_default_na=False)
            if set(frame) != {'slide_id', 'path'} or frame.slide_id.duplicated().any() or frame.eq('').any().any():
                raise ValueError('TTIS guidance index requires unique slide_id/path columns')
            if {r['slide_id'] for r in bundle.records('test')} != set(frame.slide_id):
                raise ValueError('TTIS guidance index must cover TEST identities without unrelated slides')
            guidance = {row['slide_id']: str(resolve(index.parent, row['path'])) for row in frame.to_dict('records')}
        manifest = prepare_ttis(bundle, a.output, a.semantic_metadata, a.cluster_backend, a.clusters,
                                a.cluster_seed, a.n_init, guidance, a.coordinate_frame)
        return {'manifest': str(manifest), 'encoder_entry': {'ttis': {'manifest': str(manifest)}}}
    elif a.command == 'prepare-mammil-graphs':
        from .data import Bundle
        from .io import digest_object
        from .mammil_graph import prepare_graphs
        output = Path(a.output)
        if output.exists():
            raise FileExistsError('Use a new graph manifest output')
        bundle = Bundle(a.data, a.fm, a.split)
        inventory = bundle.check(deep=True, require_coords=True)
        directory, graphs = prepare_graphs(bundle, a.graph_cache, a.neighbors, a.graph_seed)
        write_json(output, {'schema_version': 1, 'graphs': graphs, 'graph_cache': str(directory),
                            'fm': bundle.fm, 'bundle_sha256': bundle.fingerprint,
                            'features_sha256': inventory['feature_inventory_sha256'],
                            'graph_seed': a.graph_seed, 'neighbors': a.neighbors})
        return {'graphs': len(graphs), 'manifest': str(output), 'graph_cache': str(directory),
                'graph_manifest_sha256': digest_object(graphs)}
    elif a.command == 'prepare-smmile':
        from .data import Bundle
        from .smmile_data import prepare_superpixels
        manifest = prepare_superpixels(Bundle(a.data, a.fm, a.split), a.output, a.patch_stride,
            a.coordinate_frame, a.patches_per_superpixel, a.compactness, a.provenance,
            a.max_grid_cells, a.max_dense_bytes)
        return {'manifest': manifest, 'state': 'superpixel_assets_prepared'}
    elif a.command == 'prepare-pamil':
        from .data import Bundle
        from .pamil_data import prepare_annotations
        return {'manifest': prepare_annotations(Bundle(a.data, a.fm, a.split), a.annotations, a.output, a.provenance)}
    elif a.command == 'prepare-tissues':
        from .data import Bundle
        from .pamoe_data import build_tissue_prototypes
        return {'manifest': build_tissue_prototypes(Bundle(a.data, a.fm, a.split), a.assignments, a.output, a.names, a.provenance)}
    elif a.command == 'prepare-text':
        from .text_preparation import prepare_text
        return {'metadata': prepare_text(a.kind, a.features, a.metadata, a.output)}
    elif a.command=='prepare-slip-assets':
        from .slip_assets import prepare_clip
        return prepare_clip(a.output,a.architecture)
    elif a.command=='prepare-dispro-assets':
        from .dispro_assets import prepare_biobert
        return prepare_biobert(a.output)
    elif a.command == 'prepare-dspamil':
        from .data import Bundle
        from .dspamil_data import prepare
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    elif a.command == 'prepare-feather-assets':
        from .feather_assets import prepare
        return {'manifest':str(prepare(a.encoder,a.checkpoint,a.output))}
    elif a.command == 'prepare-feather':
        from .data import Bundle
        from .feather_data import prepare
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    elif a.command == 'prepare-dyko-assets':
        from .dyko_assets import prepare
        return {'manifest':str(prepare(a.titan_directory,a.output))}
    elif a.command == 'prepare-dyko':
        from .data import Bundle
        from .dyko_data import prepare
        return {'manifest':str(prepare(Bundle(a.data,a.fm,a.split),yaml.safe_load(Path(a.schema).read_text()),a.output))}
    elif a.command == 'demo-rmdl':
        from .rmdl_demo import create_demo
        return {'created':str(create_demo(a.output,a.weights))}
    elif a.command == 'demo-attention2majority':
        from .attention2majority_demo import create_attention2majority_demo
        return {'created':str(create_attention2majority_demo(Path(a.output),a.weights,a.classes,a.patches))}
    elif a.command == 'prepare-attention2majority':
        from .attention2majority_prepare import prepare
        return {'dataset':str(prepare(a.data,a.output,a.template_slide,a.normal_reference,a.component_space,
            fm=a.fm,split=a.split,seed=a.seed,threads=a.threads,device=a.device,tensorflow_python=a.tensorflow_python))}
    elif a.command == 'prepare-nciemil':
        from .nciemil_prepare import prepare
        return prepare(a.data,a.weights,a.output,yaml.safe_load(Path(a.pretrain_config).read_text()),a.k,a.device,a.resume)
    elif a.command == "inspect":
        return inspect_method(a.method)
    elif a.command == "demo":
        if a.taxomil:
            from .taxomil_demo import create_taxomil_demo
            if not a.taxomil_assets:raise ValueError('TaxoMIL demo needs actual --taxomil-assets')
            return {'data':str(create_taxomil_demo(a.output,a.taxomil_assets,a.taxomil_dataset,a.patients))}
        if a.gmmamba:
            if any(value is True for key,value in vars(a).items() if key!='gmmamba'):
                raise ValueError('GMMamba fixture cannot mix task modes')
            from .gmmamba_demo import create_gmmamba_demo
            return {'created':create_gmmamba_demo(a.output,a.patients,a.gmmamba_dimension,a.classes)}
        if a.gmmamba_dimension!=512:
            raise ValueError('--gmmamba-dimension belongs to the explicit --gmmamba fixture')
        if a.muse:
            if not a.muse_weights or any(value is True for key,value in vars(a).items() if key!='muse'):
                raise ValueError('MUSE fixture requires actual --muse-weights and cannot mix task modes')
            from .muse_demo import create_muse_demo
            return {'created':create_muse_demo(a.output,a.muse_weights,a.patients,a.muse_cohort,a.muse_dimension)}
        if a.muse_weights is not None:
            raise ValueError('--muse-weights belongs to the explicit --muse fixture')
        if a.libramil:
            if not a.libramil_weights or any(value is True for key, value in vars(a).items() if key != 'libramil'):
                raise ValueError('Libra-MIL fixture needs --libramil-weights and cannot mix input/task modes')
            from .libramil_demo import create_libramil_demo
            return {'created': create_libramil_demo(a.output, a.libramil_weights, a.patients)}
        if a.libramil_weights is not None:
            raise ValueError('--libramil-weights belongs to --libramil')
        if a.m4:
            if (any(value is True for key,value in vars(a).items() if key != 'm4')
                    or any(getattr(a,key,None) is not None for key in ['backbone_weights','vlsa_weights','vision_language','smmile','multiscale','conmil_text','convl_weights','center_assets'])):
                raise ValueError('M4 multi-endpoint fixture cannot be mixed with other input/task modes')
            from .m4_demo import create_m4_demo
            return {'created':create_m4_demo(a.output,a.patients,a.m4_dimension,a.m4_endpoints,a.m4_patches)}
        if a.convl:
            if (not a.convl_weights or any(value is True for key, value in vars(a).items() if key != 'convl')
                    or any(getattr(a, key, None) is not None for key in ['backbone_weights', 'vlsa_weights', 'vision_language', 'smmile', 'multiscale', 'conmil_text'])):
                raise ValueError('ConVL fixture requires actual --convl-weights and cannot be mixed with other input/task modes')
            from .convl_demo import create_convl_demo
            return {'created': create_convl_demo(a.output, a.convl_weights, a.patients, a.convl_cohort)}
        if a.convl_weights is not None:
            raise ValueError('--convl-weights belongs to the explicit --convl fixture')
        if a.center_embeddings:
            if (a.classes != 2 or any(value is True for key, value in vars(a).items() if key != 'center_embeddings')
                    or any(getattr(a, key, None) is not None for key in ['backbone_weights', 'vlsa_weights', 'vision_language', 'smmile', 'multiscale', 'conmil_text'])):
                raise ValueError('Center Embeddings binary raw RGB fixture cannot be mixed with other input modes')
            from .center_embeddings_demo import create_center_embeddings_demo
            return {'created': create_center_embeddings_demo(a.output, patients=a.patients, asset_root=a.center_assets)}
        if a.miladapter:
            if (any(value is True for key, value in vars(a).items() if key != 'miladapter')
                    or any(getattr(a, key, None) is not None for key in ['backbone_weights', 'vlsa_weights', 'vision_language', 'smmile', 'multiscale', 'conmil_text'])):
                raise ValueError('MIL-Adapter synthetic fixture cannot be mixed with other input modes')
            from .miladapter_demo import create_miladapter_demo
            return {'created': create_miladapter_demo(a.output, patients=a.patients, project=a.miladapter_project, encoder=a.miladapter_encoder)}
        if a.conmil:
            if (any(value is True for key, value in vars(a).items() if key != 'conmil')
                    or any(getattr(a, key, None) is not None for key in ['backbone_weights', 'vlsa_weights', 'vision_language', 'smmile', 'multiscale'])):
                raise ValueError('ConMIL text/visual fixture cannot be mixed with other input modes')
            from .conmil_demo import create_conmil_demo
            return {'created': create_conmil_demo(a.output, patients=a.patients, classes=a.classes, text_assets=a.conmil_text)}
        if a.sparseconvmil:
            if (any(value is True for key, value in vars(a).items() if key != 'sparseconvmil')
                    or any(getattr(a, key, None) is not None for key in ['backbone_weights', 'vlsa_weights', 'vision_language', 'smmile', 'multiscale'])):
                raise ValueError('SparseConvMIL raw spatial RGB demo cannot be mixed with other input modes')
            from .sparseconvmil_demo import create_sparseconvmil_demo
            return {'created': create_sparseconvmil_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.c2c:
            if any(value is True for key, value in vars(a).items() if key != 'c2c'):
                raise ValueError('C2C raw RGB demo cannot be mixed with frozen-feature or other raw input modes')
            from .c2c_demo import create_c2c_demo
            return {'created': create_c2c_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.sammil:
            if any(value is True for key, value in vars(a).items() if key != 'sammil'):
                raise ValueError('SAM-MIL demo must use its own explicit synthetic input mode')
            from .sammil_demo import create_sammil_demo
            return {'created': create_sammil_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.lrmil:
            if any(value is True for key, value in vars(a).items() if key != 'lrmil'):
                raise ValueError('LRMIL fixture requires its independent measured RGB and native intermediate layers')
            from .lrmil_demo import create_lrmil_demo
            return {'created': create_lrmil_demo(a.output, patients=a.patients, classes=a.classes)[0]}
        if a.hdmil:
            if any(value is True for key, value in vars(a).items() if key != 'hdmil'):
                raise ValueError('HDMIL fixture requires its independent paired RGB/features task')
            from .hdmil_demo import create_hdmil_demo
            return {'created': create_hdmil_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.kmgmamba:
            if (any(value is True for key, value in vars(a).items() if key != 'kmgmamba')
                    or any(getattr(a, key, None) is not None for key in
                           ['backbone_weights', 'vlsa_weights', 'vision_language', 'smmile', 'multiscale'])):
                raise ValueError('KMG-Mamba fixture requires its independent measured dual-scale inputs')
            from .kmgmamba_demo import create_kmgmamba_demo
            return {'created': create_kmgmamba_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.mgtrans:
            if (any(value is True for key, value in vars(a).items() if key != 'mgtrans')
                    or any(getattr(a, key, None) is not None for key in
                           ['backbone_weights', 'vlsa_weights', 'vision_language', 'smmile', 'multiscale'])):
                raise ValueError('MG-Trans fixture requires its independent measured dual-scale inputs')
            from .mgtrans_demo import create_mgtrans_demo
            return {'created': create_mgtrans_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.zoommil:
            if any([getattr(a, flag, False) for flag in ['augdiff','vlsa','slip','dispro','amiml','dtmil','pamil_policy','focus','codmil','ins','sclwc','deepgraphsurv','triagemil','c2mil','gatmamba','protosurv','h2mil','hipt','survival','regression','omics','multiscale','raw_images','raw_regions','backbone_weights','vision_language','smmile']]):
                raise ValueError('ZoomMIL fixture requires independent three-scale slide classification, no mixed inputs')
            from .zoommil_demo import create_zoommil_demo
            return {'created': create_zoommil_demo(a.output,patients=a.patients,classes=a.classes,geometry=a.zoommil_geometry)}
        if a.augdiff:
            if (any(value is True for key, value in vars(a).items() if key != 'augdiff')
                    or any(getattr(a, key, None) is not None for key in
                           ['backbone_weights', 'vlsa_weights', 'vision_language', 'smmile', 'multiscale'])):
                raise ValueError('AugDiff fixture requires only its explicit synthetic classification inputs')
            from .augdiff_demo import create_augdiff_demo
            return {'created': create_augdiff_demo(a.output, a.patients, a.augdiff_dimension,
                                                   a.augdiff_instances, a.classes)}
        if a.vlsa:
            if (any(value is True for key,value in vars(a).items() if key!='vlsa')
                    or a.backbone_weights is not None or not a.vlsa_weights):
                raise ValueError('VLSA demo requires --vlsa-weights and rejects mixed input/task flags')
            from .vlsa_demo import create_vlsa_demo
            return {'created':create_vlsa_demo(a.output,a.vlsa_weights,patients=a.patients,query_key=a.vlsa_query_key)}
        if a.vlsa_weights is not None:
            raise ValueError('--vlsa-weights belongs to the explicit --vlsa fixture')
        if a.slip:
            if (any(value is True for key,value in vars(a).items() if key!='slip')
                    or a.backbone_weights is not None):
                raise ValueError('SLIP native frozen-feature demo rejects mixed task/input conditions')
            from .slip_demo import create_slip_demo
            return {'created':create_slip_demo(a.output,patients=a.patients,classes=a.classes,architecture=a.slip_architecture)}
        if a.dispro:
            flags=['amiml','dtmil','pamil_policy','focus','codmil','ins','sclwc','deepgraphsurv',
                   'triagemil','c2mil','gatmamba','protosurv','h2mil','hipt','raw_images','raw_regions',
                   'backbone_weights','survival','regression','vision_language','smmile','omics','multiscale']
            if any(getattr(a,k,False) for k in flags) or a.classes!=2:
                raise ValueError('DisPro has its own incomplete patient WSI/RNA survival fixture; no mixed demos')
            from .dispro_demo import create_dispro_demo
            return {'created':create_dispro_demo(a.output,patients=a.patients)}
        if a.amiml:
            if any(value is True for key,value in vars(a).items() if key!='amiml'):
                raise ValueError('AMIML patient-bag demo rejects mixed input formats')
            from .amiml_demo import create_amiml_demo
            return {'created': create_amiml_demo(a.output,patients=a.patients,classes=a.classes)}
        if a.dtmil:
            if any(value is True for key,value in vars(a).items() if key!='dtmil'):
                raise ValueError('DT-MIL physical-grid demo rejects mixed input formats')
            from .dtmil_demo import create_dtmil_demo
            return {'created': create_dtmil_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.pamil_policy:
            if any(getattr(a, flag, False) for flag in ['focus', 'codmil', 'ins', 'sclwc', 'deepgraphsurv', 'triagemil', 'c2mil', 'gatmamba', 'protosurv', 'h2mil', 'hipt', 'survival', 'regression', 'omics', 'multiscale', 'raw_images', 'raw_regions', 'backbone_weights', 'vision_language', 'smmile']):
                raise ValueError('PAMIL policy fixture requires individual slide classification, no mixed input conditions')
            from .pamil_policy_demo import create_pamil_policy_demo
            return {'created': create_pamil_policy_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.focus:
            if any(getattr(a, flag, False) for flag in ['codmil', 'ins', 'sclwc', 'deepgraphsurv', 'triagemil', 'c2mil', 'gatmamba', 'protosurv', 'h2mil', 'hipt', 'survival', 'regression', 'omics', 'multiscale', 'raw_images', 'raw_regions', 'backbone_weights', 'vision_language', 'smmile']):
                raise ValueError('FOCUS fixture requires individual slide classification, no mixed demo conditions')
            from .focus_demo import create_focus_demo
            return {'created': create_focus_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.codmil:
            if any([a.ins, a.sclwc, a.deepgraphsurv, a.triagemil, a.c2mil, a.gatmamba, a.protosurv, a.h2mil, a.hipt,
                    a.raw_images, a.raw_regions, a.survival, a.regression, a.multiscale, a.vision_language,
                    a.smmile, a.omics, a.backbone_weights]):
                raise ValueError('CoD uses its separate physical two-scale and ordered text fixture')
            from .codmil_demo import create_codmil_demo
            return {'created': create_codmil_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.ins:
            if any([a.sclwc, a.deepgraphsurv, a.triagemil, a.c2mil, a.gatmamba, a.protosurv, a.h2mil, a.hipt,
                    a.raw_images, a.raw_regions, a.survival, a.regression, a.multiscale, a.vision_language,
                    a.smmile, a.omics, a.backbone_weights]) or a.classes != 2:
                raise ValueError('INS uses its separate binary frozen-feature slide fixture')
            from .ins_demo import create_ins_demo
            return {'created': create_ins_demo(a.output, patients=a.patients)}
        if a.sclwc:
            if any([a.deepgraphsurv, a.triagemil, a.c2mil, a.gatmamba, a.protosurv, a.h2mil, a.hipt,
                    a.raw_images, a.raw_regions, a.survival, a.regression, a.multiscale, a.vision_language,
                    a.smmile, a.omics, a.backbone_weights]):
                raise ValueError('SCL-WC uses its separate normal-tissue slide classification fixture')
            from .sclwc_demo import create_sclwc_demo
            return {'created': create_sclwc_demo(a.output, patients=a.patients, classes=a.classes)}
        if a.deepgraphsurv:
            from .deepgraphsurv_demo import create_deepgraphsurv_demo
            return {'created':create_deepgraphsurv_demo(a.output,patients=a.patients)}
        if a.triagemil:
            if any([a.c2mil,a.gatmamba,a.protosurv,a.h2mil,a.hipt,a.raw_images,a.raw_regions,a.regression,a.multiscale,a.vision_language,a.smmile,a.omics,a.backbone_weights]) or a.classes != 2:
                raise ValueError('TRIAGE-MIL uses its own explicit selected-WSI survival fixture')
            from .triagemil_demo import create_triagemil_demo
            return {'created':create_triagemil_demo(a.output,patients=a.patients)}
        if a.c2mil:
            if any([a.gatmamba,a.protosurv,a.h2mil,a.hipt,a.raw_images,a.raw_regions,a.regression,a.multiscale,a.vision_language,a.smmile,a.omics]):
                raise ValueError('C²MIL uses its own single-diagnostic-WSI survival fixture')
            from .c2mil_demo import create_c2mil_demo
            return {'created':create_c2mil_demo(a.output,patients=a.patients)}
        if a.gatmamba:
            if any([a.protosurv,a.h2mil,a.hipt,a.raw_images,a.raw_regions,a.regression,a.multiscale,a.vision_language,a.smmile]):
                raise ValueError('GAT-Mamba uses a separate synthetic multi-slide survival graph fixture')
            from .gatmamba_demo import create_gatmamba_demo
            return {'created':create_gatmamba_demo(a.output,patients=a.patients)}
        if a.protosurv:
            if a.h2mil or a.hipt or a.raw_regions or a.raw_images or a.omics or a.multiscale or a.regression or a.smmile or a.vision_language or a.backbone_weights or a.classes != 2:
                raise ValueError('ProtoSurv demo is a separate patient survival graph fixture')
            from .protosurv_demo import create_protosurv_demo
            return {'created':create_protosurv_demo(a.output,patients=a.patients)}
        if a.h2mil:
            if a.hipt or a.raw_regions or a.raw_images or a.survival or a.omics or a.multiscale or a.regression or a.smmile or a.vision_language or a.backbone_weights:
                raise ValueError('H2-MIL demo requires its separate three-level classification fixture')
            from .h2mil_demo import create_h2mil_demo
            return {'created':create_h2mil_demo(a.output,patients=a.patients,classes=a.classes)}
        if a.hipt:
            if a.raw_regions or a.raw_images or a.survival or a.omics or a.multiscale or a.regression or a.smmile or a.vision_language or not a.backbone_weights:
                raise ValueError('HIPT demo requires separate classification pyramids and actual --backbone-weights')
            from .hipt_demo import create_hipt_demo
            return {'created':create_hipt_demo(a.output,a.backbone_weights,classes=a.classes,patients=a.patients)}
        if a.raw_regions:
            if a.raw_images or a.survival or a.omics or a.multiscale or a.regression or a.smmile or a.vision_language or a.classes != 2 or a.patients % 2:
                raise ValueError('Raw ROI demo requires a separate binary fixture and an even patient count')
            from .raw_region_demo import create_raw_region_demo
            return {'created': create_raw_region_demo(a.output, patients_per_class=a.patients//2, initial_weights=a.backbone_weights)}
        if a.raw_images:
            if a.survival or a.omics or a.multiscale or a.regression or a.smmile or a.vision_language:
                raise ValueError('Raw RGB demo is a separate full-image ordinal fixture')
            from .raw_demo import create_raw_demo
            return {'created': create_raw_demo(a.output, patients=a.patients, classes=a.classes, initial_weights=a.backbone_weights)}
        if a.backbone_weights:
            raise ValueError('--backbone-weights requires --hipt, --raw-images or --raw-regions')
        if a.smmile:
            if a.survival or a.regression or a.omics or a.multiscale or a.vision_language:
                raise ValueError('SMMILe demo is a separate spatial classification/multilabel fixture')
            from .smmile_demo import create_smmile_demo
            classes = 3 if a.smmile == 'multilabel' else a.classes
            return {'created': create_smmile_demo(a.output, a.smmile, a.patients, classes=classes)}
        if a.vision_language:
            if a.survival or a.regression or a.omics or a.multiscale:
                raise ValueError('Vision-language demo is a separate classification fixture')
            from .vision_language_demo import create_vision_demo
            return {'data': create_vision_demo(a.output, a.vision_language, patients=a.patients, classes=a.classes)}
        from .demo import create_demo
        return {"created": create_demo(a.output, a.patients, classes=a.classes, survival=a.survival, omics=a.omics, multiscale=a.multiscale, regression=a.regression)}
    elif a.command == "check":
        from .data import Bundle
        b = Bundle(a.data, a.fm, a.split)
        if a.method:
            from .registry import check_compatibility, resolve_for_bundle
            spec = resolve_for_bundle(a.method, b)
            check_compatibility(spec, b)
        result = b.check(deep=a.deep)
        result.pop("features")
        return result
    elif a.command == "run":
        import uuid
        from datetime import datetime, timezone

        from .engine import train
        from .config_io import load_training_config
        cfg = load_training_config(a.config)
        if a.epochs:
            cfg["epochs"] = a.epochs
        if a.resume and not a.output:
            raise ValueError("--resume requires --output pointing to the existing run")
        output = a.output or str(Path("runs") / (a.method + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]))
        return {"run": train(a.data, a.method, output, a.fm, a.split, a.seed, a.device, cfg, a.resume, stop_after=a.stop_after)}
    elif a.command == "evaluate":
        from .engine import evaluate
        return evaluate(a.run, a.subset, a.device)
    elif a.command == "compare":
        import pandas as pd

        from .metrics import paired_bootstrap
        result = paired_bootstrap(pd.read_csv(a.left), pd.read_csv(a.right), a.classes, a.metric, a.unit, a.repetitions)
        write_json(a.output, result)
        return result
    elif a.command == "matrix":
        from .execution import matrix
        rows = matrix(a.data, a.methods, a.fms, a.seeds, a.output)
        return {"total": len(rows), "ready": sum(r["state"] == "ready" for r in rows), "plan": a.output}
    elif a.command == "run-matrix":
        from .execution import run_matrix
        result = run_matrix(a.plan, a.output, a.devices, a.config, a.epochs)
        if result["failed"]:
            print(json.dumps(result, indent=2))
            raise SystemExit(1)
        return result
    elif a.command == "report":
        from .execution import summarize_runs
        return summarize_runs(a.runs, a.output, a.subset)
    elif a.command == "prepare-tcga":
        from .preparation import prepare_tcga
        return prepare_tcga(a.inventory, {s: getattr(a, s + "_csv") for s in ("train", "val", "test")}, a.project, a.labels, a.output)
    elif a.command == "index-features":
        from .preparation import index_features
        return index_features(a.data, a.feature_dir, a.fm, a.dimension, a.weights, a.preprocessing, a.coords_dir)
    elif a.command == "trident":
        from .execution import trident_command
        command = trident_command(a.repo, a.python, a.wsi_dir, a.output, a.fm, a.mag, a.patch_size, a.checkpoint, a.custom_list)
        print(shlex.join(command))
        if a.execute:
            subprocess.run(command, check=True, cwd=a.repo)
    elif a.command == "fetch-source":
        from .execution import fetch_source
        return {"source": fetch_source(a.source, a.destination)}
    elif a.command == "hcluster":
        from .execution import hcluster_script
        text, command = hcluster_script(a.repository, a.data, a.method, a.output, a.image, a.project, a.group,
                                       a.gpu, a.cpu, a.memory_mib)
        Path(a.script).write_text(text)
        print(text)
        if a.submit:
            subprocess.run(command, check=True)
