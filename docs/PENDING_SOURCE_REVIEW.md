# 本轮未准入的来源，2026-10-22

这些审查不增加可运行方法数。后续需要完整实现与训练证据才能改变状态。

## LongMIL

[官方来源](https://github.com/invoker-LL/Long-MIL)，固定
`457b1cf6e33a05b3c8359e1fe98114cd8241f772`。已检查完整 14 次提交历史。
README 将已发布代码标为 v1，v2 的局部/分块实现仍为 coming soon，训练建议借用外部框架。
不能把 v1 的三层注意力直接改名为论文 v2；后者还需要真实空间局部注意力及 2×2 池化。

v1 全尺寸位置偏置生成器会分配 90,000² 项矩阵，因此没有执行该规模分配。
已对原始生成循环做三个缩小网格的独立检查，确认整型距离截断与对称化行为。
这仅验证偏置内核，尚未验证完整模型、训练及 v2 方法。
原代码还存在固定 CUDA 分配、非 buffer 的偏置、评估仍有 attention dropout，
以及全头共同最小值掩码等需要明确保留或命名修正的行为。

## LongMIL development update,0.104.0

The preceding source review is retained as history. The complete original
90000x90000int8 table has now been generated, every element checked against
the source formula, and all8.1GBserialized bytes losslessly preserved.
The full public-v1 native CUDA source model, logits, every gradient, Adam
moments and updated weights passed exact comparison on1025patches.
The full384D/1head/FF1536 paper-v2 reference separately implements genuine2D
radius10 sparse/chunked attention,2x2 spatial pooling and global attention.
Its equation/gradient checks passed, but it is explicitly a reconstruction;
unpublished author-v2 code has not been obtained. See [LONGMIL.md](LONGMIL.md).
Full public lifecycle, native budgets and package/global acceptance are separate
gates. This does not turn v1 into the original NeurIPS-v2 implementation.

## H²-MIL：此早期审查已由0.49完整接入替代

H²-MIL目前已准入，两条来源兼容/显式修正配置、真实层级图接口、完整训练与验证见 [H2MIL.md](H2MIL.md)。以下保留的是接入前发现的问题，不再代表当前待接入状态。

[官方来源](https://github.com/lin-lcx/H2-MIL)，固定
`ea472fefd5eab553c58041a5acc203ac7d467d06`。
已逐段读取 `main.py`、`RAConv.py` 和 `IHPool.py`。

- 主网络在多种输出宽度后复用 `LayerNorm(in_feats)`；最终输出写死为 4 类。
- 对 softmax 概率再用交叉熵；训练设置和最终二分类汇总也有范围不一致。
- `IHPool.weight_1/weight_2` 计算的 fitness 仅参与 sort、距离和硬 cluster 索引。
  进入后续损失的特征是按这些硬索引直接求均值；没有可微路径把损失传回 fitness 参数。
- `lin`、`att`、`gnn_score` 等已定义参数没有参与该 forward 的聚合。
- 层级图、父节点和真实多尺度输入仍需完整核对；不能以单尺度伪层级替代。

这些观察针对当前代码版本，不推断论文历史实验有同样问题。
这些发现已在0.49接入中逐项保留或明确修正；当前实际状态以 H2MIL.md 与方法注册表为准。
# ProtoSurv: 2026-10-22 full-network source audit, adapter pending

This section records executable source evidence; it does not add a runnable
method or count a completed integration. The reviewed source is
[wjx-error/ProtoSurv](https://github.com/wjx-error/ProtoSurv), commit
`551eaa3c0a2844e2c3557ef412d8a1fd2fec6e20`, corresponding to the
[NeurIPS2024 heterogeneous-graph paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/760341adc5632de3f1cf2e8d22215a93-Abstract-Conference.html).
Do not confuse it with similarly named multimodal or CT methods.

The actual full source network ran on synthetic two-graph PyG batches with
input1024, four GCN layers/hidden128, prototype768, five tissue categories,
eight prototypes per category, two prototype-attention iterations and four
fusion layers. Original Cox, global compatibility and orthogonal objectives
backpropagated with finite gradients, and an actual Adam update completed;
292 parameter tensors had nonzero gradients. This is a mechanism audit, not
a full trainer, GPU run, clinical result or reproduced paper number.

Full integration needs the following contracts and source behaviors:

- Physical spatial graphs and aligned node categories need versioned inputs.
  The original fine-tuned UNI classifier is not released; the author provides
  a CONCH alternative and fixed prototype assets. Record those extra model
  conditions and the feature space. Never reuse UNI-space prototypes for an
  unrelated FM merely because dimensions agree.
- The released graph builder fits a feature-space HNSW index but queries it with
  coordinates for latent edges. The source model reads spatial `edge_index`;
  its unused latent-edge construction must still have an explicit preprocessing
  decision, not a hidden substitution.
- Compatibility is **not cross-entropy**: it sums true-class distances and
  logsumexp of negated non-target distances, divided by node_count*class_count.
  An independent two-class example matched this exact formula. Its
  `F.one_hot(labels)` omits `num_classes`, and actually fails when the highest
  tissue category is absent. Any shape correction must preserve the objective.
- A direct raw graph without node categories fails in the model, but that is
  not the complete source dataset path. Executing the actual dataset confirmed
  that fixed-prototype mode supplies an explicit-1unknown-category vector.
  The default nonzero compatibility term then fails on those negative IDs.
  Support this mode only with a clearly declared compatible objective; do not
  invent tissue categories or misreport the dataset as omitting the field.
- `pad_batch(seed=...)` resets global Torch RNG even without subsampling;
  source versus local-generator behavior requires explicit profiles. TRAIN
  has a25000-node padding/sample cap; VAL retains all nodes.
- The full trainer has batch6 custom event/censor sampling, gc2 accumulation,
  censored-batch skips and an unconditional tail optimizer step. Exact update
  semantics, Cox tie/epsilon behavior, Adam5e-5/1e-5, cosine T_max5 across100epochs,
  source validation C-index and selection/early stopping all remain to integrate.

Audit scripts, source hashes and numerical/gradient evidence are persisted only
under the user's own directory:
`/mnt/medeval-share/xiangyuli/mil_baseline_puls_work/development/protosurv_development`.

## ProtoSurv development update

The preceding section is the historical first source audit. The complete graph/category/prototype bridge, all three objectives, actual source sampler/augmentation/gc/Adam/cosine/validation recipe and exact checkpoint state are now implemented in the0.53candidate; source and named repaired routes are documented in PROTOSURV.md.28focused checks, all207profile runs and four full native-width source-budget wheel runs passed. The fresh global regression and final package acceptance remain separate gates; this update alone does not count admission.

## C²MIL and GAT-Mamba: executable source review, adapters pending

C²MIL official source: https://github.com/mimic0127/C2MIL, commit fd4d54fc115466e43019c77015cdef01d90c85c3 (MIT). Actual full1024feature/128graph-hidden source network, six-patient TRAIN reference population with1500patches per patient, five original joint objectives and one Adam update executed. Only CPU Cox placement was changed; inactive plotting/rotary/scatter/timm imports were isolated. It still needs actual thumbnail and TRAIN-reference input contracts, the batch16 final-batch-merge rule,100epoch/40epoch LR-halving recipe, source VAL tie selection and dynamic cluster/bias checkpoint buffers. Native source state_dict omits those dynamic centers/biases; the original final weight-only overwrite loses them. See validation/c2mil_source_audit.json; this is not a runnable adapter.

GAT-Mamba official source: https://github.com/rina-ding/gat-mamba, commit18979497990a66ffe6a75880f0336f85914552f0 (GPL-3.0). The declared Mamba1.2.0.post1 source was separately pinned to34076d664838588a3c97727b263478ab9f621a07. Actual native1024UNI/64projection/16position/80state-channel source graph model and source Cox+L1 Adam update ran using that same-version author CPU slow branch and selective_scan_ref; no CUDA claim. Under declared PyG2.6.1, the original GATConv omits edge_dim: changing all categorical/continuous edge values leaves risks bitwise unchanged, and edge encoders have no data gradients, only source L1 gradients. Preserve that source behavior and separately name any edge-conditioned paper repair. Full adaptation still needs the original physical8-neighbor/disjoint-patient graph and six-class LUAD classifier provenance, shuffled VAL batch objective/plateau scheduler, inclusive201epoch cap and source last-tie selection plus exact recovery. See validation/gatmamba_source_audit.json.

Both executable probes and pinned external source checkouts are persisted under the user-owned directory /mnt/medeval-share/xiangyuli/mil_baseline_puls_work/development/next_sources. Neither source probe increments the admitted method count.

ProtoSurv acceptance update: the0.53current full2320-case suite and final independent package checks completed successfully. All207CPUprofile runs and four native-width original-budget runs passed. ProtoSurv is now admitted; the earlier development/pending language above describes the historical audit stage. C²MIL and GAT-Mamba source probes remain uncounted.

GAT-Mamba acceptance update:0.54adds the complete original graph/Mamba/source-training path and separately named edge-conditioning correction, with23focused checks,2343complete CPU checks,209profile runs and four original-budget native-width wheel runs. The earlier adapter-pending statement describes the historical source audit. C²MIL remains pending its complete adapter.
