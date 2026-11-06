# PAMIL：动态策略，而非原型注意力

0.62 独立开发候选，尚未正式计入方法数。该条目是 Zheng 等人的 *Dynamic Policy-Driven Adaptive Multi-Instance Learning for Whole Slide Image Classification*，与库内已有的原型注意力 PAMIL 是不同论文。

原始来源：`titizheng/PAMIL@d62960af2c115f87d570a0fbf8852049745b4e94`。[CVPR 主文](https://openaccess.thecvf.com/content/CVPR2024/papers/Zheng_Dynamic_Policy-Driven_Adaptive_Multi-Instance_Learning_for_Whole_Slide_Image_Classification_CVPR_2024_paper.pdf)、[官方补充材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zheng_Dynamic_Policy-Driven_Adaptive_CVPR_2024_supplemental.pdf)。检查的 Git 树没有根许可证，因此私有获取并加载作者源码，不在发行包中复制。补充材料也不随代码分发。

## 配置与输入

```bash
milplus setup --method pamil_policy
milplus run --data /absolute/task_bundle --method pamil_policy \
  --output /absolute/new_run --device cuda
milplus evaluate --run /absolute/new_run
```

数据包是切片级分类任务，提供完整 patch 特征；在对应 encoder 中登记非空 `row_order_provenance`，描述真实提取与保存顺序。插值依据的是稳定的行顺序和行身份，不能把它解释为真实物理坐标建模。程序不构造假物理网格。库的患者与特征内容哈希、划分检查继续适用。

两条明确命名的配置：

| ID | 条件 |
|---|---|
| `pamil_policy_paper_reference` | 论文参考：融合后的历史用于后续融合和 TCM；默认 LIIS 使用实际前一组行 ID 对策略分数插值，选 top-M 并移除已选行 |
| `pamil_policy_released_sampling_reference_training` | 公开源码对照：保留原始索引归一化插值、有放回抽样、首组未移除、填充与 floor-T、原始 raw-token 历史路由；训练部分仍为明确重建 |

二者均包含完整训练，不能把第二条标成“作者原始训练复现”：公开仓库没有完整训练循环。默认别名 `pamil_policy` / `PAMIL dynamic policy` 指向论文参考路线；不覆盖已有 `pamil_prototype_*`。

原生 512 维分支按作者 Lung 代码绕过 `fc1`，1024 维分支按 CAMELYON 代码经过 `fc1`。其他 FM 完整 D 维经可学习 D→512 投影，明确登记为输入维度扩展，保留全部通道。可以显式强制或限制投影条件，不能把扩展称为作者原生输入。

## 完整机制

保持原始 512 隐藏维度、两层八头注意力、历史融合、切片 TCM、512 隐藏的 GRU 与 Gaussian actor/critic。原始策略分数由 Sigmoid 均值、高斯采样及 [0,1] 截断得到；保留该实际实现，不替换成其他分布或普通注意力池化。

参考路线支持 `scheme: liis/gmss/ghss`。LIIS 选 top-M；GMSS 按补充材料的字面规则 `P>xi` 选择与当前状态最相似的剩余实例，否则随机采样；GHSS 混合最近半组和远端半组。补充材料未规定距离范数、远端边界及插值重复索引细节，本参考显式采用余弦、排序位置 `2M` 起的远端，池过小时取尚未选中的最远实例。初始组随机且不放回，最后不足 M 的组保持真实大小，全部原始行只访问一次。源码对照保留其实际有放回行为。

三项训练目标全部执行：切片 CE、每步伪包 CE、相邻步骤预测器的对称 stop-gradient 负余弦 SIA。第一步没有前驱，因此该项贡献 0，整体按 T 归一化，此边界选择明确登记。反馈来自实际训练切片：正确分类奖励减去 CLS 与融合 token 的余弦；奖励在进入 PPO 前 detach。验证与测试标签不参与采样或选择预测路线。

论文默认每组 512、AdaMax lr1e-4 / wd1e-5、300 轮、batch1。保持独立 PPO Adam lr1e-5、gamma0.1、K3、clip0.2，以及原代码的 value-MSE / entropy 项。先对 detached rollout 执行 PPO，再对独立主目标反传；训练顺序未由作者公开，这是显式重建选择。每步包括最后一步的策略抽样与 PPO 记忆结构按发布代码保留。两个优化器分别保存，主优化器不更新策略或旧策略。

使用作者余弦函数；参考端点为 SIA 0.5→0、STL 0→0.1。补充材料给出初始示例和 STL 上限，但未完整公开历史端点；不得把参考值写成已核实的历史超参数。CAMELYON 奖励为 1，Lung 可在配置中明确设 2；新任务的奖励是预先固定的实验条件。

## 源码与论文差异

源码 SFFR 的历史来自 `msg_states`，融合结果写入 `merge_msg_states`，但 TCM 又读取未融合的 `msg_states`。论文参考将后续融合和 TCM 路由到融合历史；源码对照保留 raw 历史。其差异通过实际原生网络检查，不把同名类的可运行性当作机制一致。

源码 LIIS 用归一化索引插值后按概率有放回抽样，而补充材料写 top-M；源码首组还保留在剩余池中。两条路线分别记录。公开 GMSS 的 `P>xi` 字面规则与“xi 从 0 增至 0.9 表示更多目标选择”的叙述存在方向差异；实现采用字面规则并披露，不猜测作者的历史行为。

主文公式 5 依据真实标签分支选择最终融合，不能用于未知病例。本实现保留公开测试代码的 label-independent TCM 输出；不根据 TEST 标签决定采用哪条预测。

CPU/GPU 兼容处理保留原始数值路径：注册高斯尺度 buffer、在实际设备创建状态和索引、修正单维动作的 squeeze 形状，以及单步 reward.std 的有限边界。空填充和全零策略分数有明确处理，普通原生 512 维、多步骤路径与未改算法的作者对照比较。

## 选模、检查点与证据范围

完成全部配置轮数，不根据未公开的规则提前终止。默认以验证 macro-AUC 选择首个最优检查点，也可明确选验证 log-loss；这是 benchmark 选模协议，并非声称作者提供了相同规则。

检查点包含全部主网络、真实策略和同步的旧策略、Gaussian scale、两个优化器、完整余弦时域、训练顺序、实际采样摘要、计数和全部 RNG。每个 bag 重置临时记忆；支持 epoch 边界精确恢复。检查主/PPO 更新次数、真实 TRAIN 身份、每张 bag 的行数和选样摘要，不把验证病例写入训练记忆。

当前的短轮数训练与源码机制检查用于软件验证。完整原生 512 分组 / 300 轮独立安装包、全库、全部配置和最终重载仍须完成验收；没有自然 BRCA/COAD、GPU 或论文数值复现结论。

R1 已通过18项专项检查，相关结果仅属于原修订版，未用于R2验收。R2 增加实际原生512分组、1536维、1537实例与单例末组的GMSS/GHSS标量策略完整梯度和独立PPO更新对照；当前R2全部20项专项已重新运行并通过（见pamil_policy_v62_focused_r3.xml）。完整2490项/219配置与四组300轮预算仍待完成。

R2 修正阈值推进粒度：默认 `threshold_schedule: train_bag_iteration` 按每次 TRAIN bag 更新推进，首个为0、最后计划更新为0.9；保存的主更新计数同时用于精确恢复。`epoch_reference` 单独保留早期按epoch推进的参考选项。原论文未公开实际训练器，默认采用“迭代”的训练bag粒度，不把粗粒度选项静默声称为相同调度。R1记录保留且不计入修正版验收。
