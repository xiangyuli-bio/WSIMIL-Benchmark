# CPU 数学库首次调用：0.119.0 候选

此候选尚未通过整体验收。计划仍为 164 条，当前写入 161 条、352 套配置；原来的三个缺项与全部准入要求保持不变。

## 实际故障证据

0.118.0 的原始 SparseConvMIL CPU 比较出现逐值失败。相同首批输入、随机状态、初始权重和首层梯度，在第一次 Adam 更新后产生不同权重。差异集中在首层 9408 个元素的前 4704 个；其中 4692 个值不同，最大绝对差约 5.9232e-7。

使用原故障张量和正确参考的 Adam 状态，把平方根替换为硬件近似 `x * rsqrt(x)`，其余原始 Adam 算式保持不变，可以逐值重建这 4704 个异常结果。这是低精度平方根路径的实验线索，不能单独证明触发该路径的原因。

[Intel 的原始问题记录](https://community.intel.com/t/vmssqrt-vml-ha-sometimes-returns-low-accuracy-results-when-threads-make-their-first-call/143910)描述了静态链接 MKL 首次并发 VML 调用的相似问题，并给出串行首次调用的规避办法。该报告使用的 MKL 版本与当前 PyTorch 链接的 2024.2 不同。当前环境中，独立 pthread 对照尚未复现故障，因此不能把这个报告直接当作当前故障的根因证明。

[PyTorch 2.5.1 的 VML 实现](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/cpu/vml.h)显式请求高精度；[CPU 一元算子](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp)会将连续平方根数组分块执行。这些源码用于约束对照设计，不能代替实际复现。

## 候选改动

`milplus.numerics.initialize_cpu_vector_math()` 使用正常 PyTorch CPU 标量调用，分别初始化 float32、float64 平方根。每个进程只执行一次，线程锁保证首次调用串行；fork 子进程重置锁和进程标记。公共训练、评价和设定随机种子的入口都会先调用它。

没有替换模型运算、优化器、损失、参数量或原训练阶段，没有改变 CPU 测试设备、线程数或数值相等断言，也没有调用 `vmlSetMode` 修改全局精度设置。初始化只使用常量，不消耗随机数。

## 验证边界

新增检查覆盖随机状态、框架线程和标志的保留，并发首次请求、父进程持锁时的 fork，以及公共随机种子入口的调用次序。原有 SparseConvMIL 原始模型／训练／恢复比较继续执行。通过的诊断或局部测试不自动解除旧故障；独立安装包、完整测试域及底层对照仍需分别核验。

0.118.0 的冻结源码、安装包、未验收候选 ZIP、原始失败和实际 Tensor 记录均保留。旧版本的 4017 项正例覆盖不计入 0.119.0 的通过数量。真实 BRCA／COAD／WSI 实验和论文指标复现尚未运行。

## ProtoMIL 安装依赖

原始 wandb 0.16.1 需要 `pkg_resources`。`protomil` 可选依赖现固定 `setuptools==80.9.0`，避免环境解析到不再提供该模块的 setuptools 84。该依赖在独立前缀中安装，完整 ProtoMIL 检查会使用这个声明的版本；已有训练环境保持原状。此项仍待当前完整回归确认。
