# 当前开发候选快照

本工作树为0.109：161个完整实现候选、352个执行配置。RMDL完整源流程已实际运行，当前正在执行公开入口、独立安装包、跨阶段恢复及完整预算验证；它尚未计入软件阶段已验证的160条。
剩余未写完整适配器为DyKo、DSPA-MIL／DSPA-U-MIL和FEATHER。RMDL完成全部所需验证前，也仍属于待最终验收条目。

已发布开发版151条／321配置，稳定版117条／219配置。全库最终验收和真实BRCA／COAD实验未完成。

下方为继承的历史记录，不能用其中旧状态替代上述本候选快照。

# Current candidate

Unpublished 0.108.0 contains 160 implementation candidates and 350 profiles.
159 entries passed their earlier software gates. The new complete MExD
reference is written but is not counted as the 160th software-verified entry
until its independent source/wheel lifecycle checks pass. The published
development export remains 151/321 and stable remains 117/219.

MExD preserves both complete 512D TransMIL adapters, all expert objectives,
100+200 training epochs and the original 100-path/500-step diffusion posterior.
54 related focused/interface checks and a complete independent CPU CLI recovery
preflight passed. Original released graph conflicts and all reference choices
are disclosed in [MExD](docs/MEXD.md). Full CUDA source/wheel and native-budget
checks remain separate. All current new evidence is synthetic engineering
validation, not real WSI or paper-score reproduction.

The new LPGMIL adapter preserves the complete512D/two-Mamba1.1.1/six-prototype
masked8head architecture, top1000 selection and dualCE+L1 training. Its missing
prototype preparation is an explicitly named TRAIN-only lymphocyte-count/Ward
reference; released width, constructor and output defects are documented.
See [LPGMIL](docs/LPGMIL.md).

LPGMIL passed all four source/wheel software lifecycle conditions and two
full-dimensional source-equation oracles; its full native budget is tracked
independently under the frozen 0.107 source identity.

Prompt-MIL passed all12 source/wheel software lifecycle conditions and all3
full-dimensional source-equation oracles. Its12 native40/50epoch conditions
are running separately. All new checks are synthetic engineering evidence.

Whole-library final validation, all164-entry acceptance, real WSI experiments
and paper metric reproduction remain incomplete.
