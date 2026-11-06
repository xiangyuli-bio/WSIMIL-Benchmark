# C2Aug: complete paper-reference profile

`c2aug_transmil_paper_reference` (alias `C2Aug`) is a new, explicitly identified
reference implementation of the complete paper mechanisms. It is **not** an
unchanged execution of the released Lightning trainer and is not a numerical
reproduction of the paper. GPU and natural WSI validation remain separate.

Sources: [official repository](https://github.com/weiaicunzai/mixup), pinned
`14cb53bc71f5161105b0dbef45351243d7330561`;
[CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Contrastive_Cross-Bag_Augmentation_for_Multiple_Instance_Learning-based_Whole_Slide_Image_CVPR_2026_paper.pdf);
[six-page supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhang_Contrastive_Cross-Bag_Augmentation_CVPR_2026_supplemental.pdf).
Paper PDF SHA256: `61e20b0dc0df0fe2e7fe946a37a729173601716c2a1a76df2ab3071ffec1c3be`;
supplement SHA256: `378d47a7f0b49034fe8cf6ad0ce3336e67886e63a8108c8641b82c9a59a73c2a`.
Original code stays an external checkout because no license was found in the
pinned source tree. Only verified Git definitions are executed by the adapter.

## Full training path

The native author TransMIL remains 512-wide, with its two Nyström blocks and
PPEG; only the CUDA-only CLS allocation is changed to the input device. The
registered FM width and class count replace source launcher hardcoding.
Its original square-padding behavior is retained. Prototype grouping consumes
the returned `contra_feature` tokens, including the repeated padding tokens;
the adapter does not silently trim them or claim a different token definition.

Both branches receive independent TRAIN augmentations: class-consistent
multi-view fusion, instance expansion and learned cross-attention compression.
Fusion uses the original instance as the query (paper Eq2), other same-class
bag views as keys/values, and a row-wise random mask retaining at least one
view. Sampling is disjoint within an augmented set. Compression uniformly
draws a ratio2..16 and samples padding from its own input bag, including very
short bags. The supplement specifies linear256 reduction/restoration around
augmentation,8 random views,256 FIFO entries,8 prototypes, Adam1e-4,batch1 and
200epochs; these are the defaults. Here `max_views=8` means up to8 additional
views, excluding the original query, rather than the source CLI's total-view
interpretation. This choice is explicit and is part of the checkpoint contract.

The objective is CE + alpha*bag InfoNCE + beta*group KL. Bag embeddings are
linearly projected and L2 normalized. The FIFO stores detached teacher keys
and slide IDs; every occurrence of the current slide is excluded from its
negative set. At the first step the queue is empty, so positive-only InfoNCE
is exactly0. Subsequent distinct bags provide actual negatives. No random
negative initialization is invented.

Eight trainable student prototypes perform cosine assignment and per-group
query/key/value attention. Empty groups use their prototype as their own
key/value. Group features undergo linear projection and L2 normalization.
Teacher centering/sharpening precedes the forward KL(student||teacher), summed
over **all** corresponding groups and embedding coordinates (Eq8). Gradients
reach student prototypes, projections, augmentation and the full MIL. All
teacher parameters, including prototypes, are frozen and updated with EMA.

The public source supplies no active bag contrast in `training_step`, does not
include root prototypes in its optimizer, and broadcasts only two student
groups against two five-row teacher chunks in its DINO loss. Its100-entry
temperature array also ends before the200epoch budget. These are documented
source differences, not retained as hidden omissions in the paper profile.
The extra source FFT augmentation is absent from the paper's three-operation
definition and is not enabled here. No statement about historical paper
results follows from inspecting this particular source version.

## Explicit implementation choices

The paper/supplement do not specify every needed detail. This profile chooses
one-head scaled cross-attention; linear128-dimensional contrastive projections;
alpha=beta=1; bag temperature.07, student.1,teacher.04; teacherEMA.9999 and
centerEMA.9; source-like .25MV/.25expansion and independent .5compression.
The teacher remains in train mode under no_grad during training, allowing
independent MIL dropout. After Adam commits, EMA updates the complete teacher,
the center incorporates pre-update teacher groups and the FIFO adds its key.
These choices are recorded and must not be reported as author-disclosed
defaults. Temperatures are constant through the complete configured budget.

## Dataset and selection boundary

Only TRAIN bags contribute reservoir rows, maximum bag size and cross-bag
views. The current slide is excluded from the augmentation reservoir, matching
the paper's other-bag wording. Therefore enough other same-class TRAIN rows
must exist for disjoint views. Insufficient data raises an actionable error;
it does not silently repeat features or use validation/test data. Explicitly
set fewer views for a low-data experiment and report that condition.

The reservoir stores offsets/counts, not a duplicate tensor of all FM features.
H5/NPY reads retrieve only selected rows; PT reads load one relevant bag at a
time. Sampling uses a bounded-memory index selection. Data, FM and reservoir
contracts, the private DataLoader generator, global RNG, both networks,
prototypes, optimizer, FIFO values/identities/cursor, teacher center and
augmentation counters are checkpointed for exact epoch-boundary continuation.

The first strictly best **distinct validation** slide macroaccuracy selects the
checkpoint; the default200epoch budget has no early stopping. Standardized
patient/slide metrics remain separate. Inference runs only the student MIL on
the original bag, without augmentation, labels, reservoirs, groups or FIFO
changes. Test predictions are generated only by explicit evaluation after
selection. The source merges TRAIN+VAL for its reservoir and reuses TEST for
selection; this benchmark deliberately seals the outer TEST split.

```bash
python -m pip install '.[crossbag]'
milplus setup --method C2Aug
milplus run --data /path/to/prepared_task --method C2Aug --output /path/to/new_run
milplus evaluate --run /path/to/new_run
```

Use `--help` for the actual CLI options. Real TCGA labels/patient splits,
TRIDENT extraction, GPU behavior and published scores still need their own
validation; synthetic checks establish software mechanics only.

## Validation scope

The focused CPU suite has39 passing cases, including six bitwise native
TransMIL output/loss/gradient comparisons, independent augmentation and loss
calculations, H5/NPY/PT reservoir equivalence, actual updates to every intended
student module, queue identity exclusion and exact checkpoint continuation.
Two independent full trainers execute90 optimizer updates across1024-dimensional
binary and1536-dimensional three-class inputs and agree after every update.

The actual unified CLI and an externally installed wheel each complete a
three-epoch135-update synthetic run with the original512-wide model and the
supplement's8views/256queue/8prototypes/256reduction settings. Both reload the
selected checkpoint and generate reproducible predictions. These are two
short native runs, **not** completed200epoch experiments. The separate
203-profile CPU run exercises every registered configuration. Full-regression
acceptance and artifact identities are recorded in [VALIDATION.md](VALIDATION.md)
and [validation/summary.json](validation/summary.json).
