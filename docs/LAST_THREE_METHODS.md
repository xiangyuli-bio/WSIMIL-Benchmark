# DyKo、DSPAMIL、FEATHER：代码接入与使用

0.131.0 将已有完整模型驱动接入公共 `setup / inspect / check / run / evaluate`。
原有396套配置不变，增加19套，共415套，覆盖164条方法记录。
本轮交付代码可用性，不启动全预算训练；`adapted_under_verification` 和
`formal_admission: false` 保留，原始预训练执行、完整训练、真实数据与论文复现不因此通过。
DSPAMIL目录合并了两种名称，但这里只接入DSPA-MIL2025参考实现，DSPA-U未实现。

## 各自安装，不混装

在独立Python环境执行相应的一条命令：

```bash
python -m pip install -e '.[dyko]'
python -m pip install -e '.[feather]'
python -m pip install -e '.[dspamil]'
```

这三条表示三个环境，不能在同一环境顺序合装。DyKo和FEATHER的Transformers、
tokenizers、hub版本不同，并要求NumPy1；当前仓库的基础`.venv`没有替它们安装
这些互斥依赖。CUDA使用者应先安装对应Torch2.5.1的CUDA构建。
DyKo extra提供CPU FAISS；显式CUDA-FAISS配置另需真实可用的GPU FAISS1.8运行时，
不会退回CPU后仍报告GPU聚类。额外的原始源码、权重和输入要求如下。

## DyKo

```bash
milplus setup --method DyKo --plan
milplus setup --method DyKo
milplus prepare-dyko-assets --titan-directory /path/to/authorized/TITAN \
  --output /path/to/new-titan-assets.json
milplus prepare-dyko --data /path/to/task --schema /path/to/dyko-schema.json \
  --output /path/to/new-dyko-inputs.json
```

按[DyKo输入说明](dyko_native_candidate.md)填写schema，并在对应encoder下绑定
`dyko: {manifest: /path/to/new-dyko-inputs.json}`。必须是原生对齐768维特征、
对应cohort的概念和原始TITAN模型，不能用同维随机特征或其他文本塔替代。

```bash
export MILPLUS_DYKO_PYTHON=/path/to/dyko-env/bin/python
milplus check --data /path/to/task --method DyKo --deep
milplus run --data /path/to/task --method DyKo --device cuda:0 --output /path/to/new-run
milplus evaluate --run /path/to/new-run --device cuda:0
```

家族名根据已准备输入选择CAMELYON16、NSCLC、RCC或UBC，默认明确使用CPU FAISS
参考聚类，模型本身仍可在CUDA运行。无数据的`inspect DyKo`显示NSCLC参考默认，
不代表所有任务都是NSCLC。需要GPU聚类时明确指定，例如
`dyko_nsclc_cuda_faiss_requested_reference`。总计4个cohort×2种FAISS配置。
保留16-shot默认、Adam、梯度累积4、最大1000轮及40轮耐心；本轮不自动执行。

## FEATHER

```bash
milplus setup --method FEATHER
milplus prepare-feather-assets --encoder uni --checkpoint /path/to/original/model.safetensors \
  --output /path/to/new-feather-assets.json
milplus prepare-feather --data /path/to/task --schema /path/to/feather-schema.yaml \
  --output /path/to/new-feather-inputs.json
```

按[FEATHER输入说明](FEATHER_NATIVE_CANDIDATE.md)绑定
`feather: {manifest: /path/to/new-feather-inputs.json}`。原生encoder只支持
`conch_v15 / uni / uni_v2`，维数分别768/1024/1536；原始24K检查点和视觉权重
身份分别检查，不自动获取受限资产，也不接受随机初始化作为预训练替代。

```bash
export MILPLUS_FEATHER_PYTHON=/path/to/feather-env/bin/python
milplus check --data /path/to/task --method FEATHER --deep
milplus run --data /path/to/task --method FEATHER --device cuda:0 --output /path/to/new-run
milplus evaluate --run /path/to/new-run --device cuda:0
```

家族名选择准备好的encoder及`finetune_val20`协议；无数据inspect默认展示UNI。
9套配置为3个encoder×`finetune_val20 / finetune_no_val10 / frozen_knn20`。
选择其他协议须使用显式名称，例如`feather_uni_frozen_knn20_reference`。
冻结KNN严格要求至少20个TRAIN切片，不会自动降低K。
微调保持完整20/10轮参考配方，不能使用通用`--epochs`静默缩小它。

## DSPAMIL：仅DSPA-MIL2025参考

```bash
milplus setup --method DSPA-MIL
milplus prepare-dspamil --data /path/to/six-grade-UNI-task --schema /path/to/dspa-schema.yaml \
  --output /path/to/new-dspa-inputs.json
```

按[DSPA说明](DSPAMIL.md)绑定`dspamil`输入。需要原生UNI1024维袋、六个有序等级、
仅TRAIN的实测模式标签及有依据的模式/等级概念和文本对齐资产。
不能从等级猜测模式标签；维数相同不能证明文本对齐。未发布的方程、头结构、
早停与等级判定选择必须在配置中明确说明，示例不是原作者未公开配置的恢复。

```bash
milplus check --data /path/to/task --method DSPA-MIL --deep
milplus run --data /path/to/task --method dspamil_2025_released_mask_reference \
  --config configs/dspamil_2025_full_reference.yaml \
  --device cuda:0 --output /path/to/new-run
milplus evaluate --run /path/to/new-run --device cuda:0
```

如使用独立DSPA环境，以上命令均可由该环境的`milplus`执行；也可在配置的
`model.runtime_python`指定独立解释器。另一配置是
`dspamil_2025_paper_mask_reference`，与发布源码的掩码语义分开登记。
保持完整162,690,072参数、12层1024维PAT及双路头，完整50轮参考配方。
`DSPA-U-MIL`会明确报未实现，不会映射成2025实现冒充支持。

所有输出均须使用新目录。没有所需权重/输入会在训练前报错，不占用GPU空跑，
也不会因注册入口就把历史验证标记为通过。
