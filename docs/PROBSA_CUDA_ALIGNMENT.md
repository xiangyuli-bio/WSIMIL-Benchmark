# ProbSA：实际CUDA稀疏乘法对齐修复

真实BRCA训练尝试暴露`CUDA error: misaligned address`。在既有H200、
Torch2.5.1+cu124上，不调用MIL训练，仅用`torch.bmm`计算稀疏`[4,N,N]`
与稠密`[4,N,1]`，N2691即可独立复现；因此不是把失败归咎于标签或模型收敛。

`probsa.graph_product`只将稠密右操作数的输出通道工作区补零至16字节对齐，
乘积取回原输出通道。没有增加/删除实例，不补图结点，不改变坐标、边权、
KL归一化、MC抽样数、随机数状态、网络结构、损失或训练预算。
CPU和稠密图路径继续原样调用原`torch.bmm`。

## 实际验证

- 既有ProbSA完整CPU机制/训练/恢复测试与新增乘法测试：73通过，8项CUDA用例跳过。
- 独立H200运行：19通过，无跳过。包括float32/float64、N3/128/2691/16848，
  与独立scatter公式核对输出和梯度，并核对ragged袋的完整KL及mu/logvar梯度、
  padding梯度为零、RNG不变。图边权导数仅在小图检查，真实大图保持常量，
  避免测试本身构造无关的N平方稠密梯度。

```bash
python -m pytest -q tests/test_probsa.py tests/test_probsa_cuda_alignment.py
```

证据位于本任务共享目录`brca_uni_idc_ilc_20261105/probsa_cuda_candidate/`：
`alignment_and_probsa_cpu.xml`、`alignment_gpu_v2.xml`、`repair_receipt.json`。
原始训练源码SHA与修复后源码SHA分别保留；已核对独立GPU候选只改动`probsa.py`。

## 训练边界

不原地修改当前GPU训练的软件身份。新候选位于GPU本地
`/root/milplus_brca_uni_20261105/code_probsa_cuda_alignment_20261105`，
失败项重试进入独立下一轮。只提高MC分配保护阈值以容纳完整真实袋，
不降低MC抽样数、实例数或epoch数。

上述是算子及代码验证，不是整轮BRCA训练成功，也不提升164个方法的整库验收状态。
