"""Generate human-readable scope and state from the same runtime registry."""
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    res=root/'src/milplus/resources'
    methods=json.loads((res/'methods.json').read_text())
    profiles=json.loads((res/'profiles.json').read_text())
    lines=['# 接入状态与剩余范围','',
           f'当前候选 {len(methods)} 条，可运行配置 {len(profiles)} 个，覆盖 {len({p["family"] for p in profiles})} 个候选家族。',
           '', '**完整 164 条候选的实现尚未完成。下表是实际状态，不是完成承诺。**', '',
           'CPU 合成验证、GPU 验证、真实 WSI 实验和论文数值复现分别登记。CPU 合成结果不能用于 benchmark 结论。', '',
           '## 可运行配置','', '| ID | 家族 | 软件状态 | 配方 |','|---|---|---|---|']
    for p in profiles:
        lines.append(f'| {p["id"]} | {p["family"]} | {p["status"]} | {p["recipe"]} |')
    lines += ['', '## 全部候选', '', '| ID | 名称 | 分组 | 实现状态 | 来源 |', '|---|---|---|---|---|']
    for m in methods:
        source=f'[{m["source_status"]}]({m["source_url"]})' if m['source_url'] else '待核原始来源'
        lines.append(f'| {m["id"]} | {m["name"]} | {m["group"]} | {m["status"]} | {source} |')
    lines += ['', '## 剩余工作', '',
              '1. 补齐未确认条目的论文身份、原始源码、训练入口与必要资源。原先 Excel/ZIP 未存在于当前工作区；本台账据会话名单重建，未冒充此前文件内容。',
              '2. 按方法实现完整适配器与配方，再增加机制测试和参考运行。待接入不会自动回退到基础模型。',
              '3. 多尺度接口已支持 CS-MIL 与 HAG-MIL CE；继续实现其他多尺度、图文、持续学习等专轨；组学/通路和离散生存接口已支持 SurvPath、MCAT、PORPOISE、CMTA、MOTCat，物理图接口已支持 Patch-GCN，其余专轨方法仍逐项接入。',
              '   空间分类新增 SmMIL、PSA-MIL、SlideGraph+；其坐标单位、固定作者源码、专用目标和适配修正见 SPATIAL.md。Bayes-MIL、CAPRMIL 和 VarMIL 的七配置与公式/数值修正见 BAYES_CONTEXT_VARIANCE.md。',
              '   MiCo 两任务与 EfficientMIL 两循环网络的初始化、目标和范围见 CONTEXT_SEQUENCE.md；AGDR 的完整双分支、原配方、无梯度分级器边界和随机验证语义见 AGDRMIL.md；PAMIL 的完整分阶段与额外实例监督见 PAMIL.md。',
              '   SCMIL 连续混合分布与 DeepAttnMISL 表型簇生存的输入、目标和评价范围见 CONTINUOUS_SURVIVAL.md。',
              '   MIL-Dropout 四骨干与 PackMIL 分类/生存的机制、优化配方及源码差异见 DROPOUT_PACKING.md。',
              '   S-DFP 两阶段、MFC-MIL 记忆修正版与 DyHG 动态超图的七配置见 PROMPT_FREQUENCY_HYPERGRAPH.md。',
              '   两个同名 CAMIL 的独立来源、网络、完整训练配方和 TensorFlow 对照见 CAMIL.md。',
              '   HEALNet 多模态生存和 HE2RNA 表达回归、空间 supertiles 与作者优化配方见 MOLECULAR_REGRESSION.md。',
              '   CATE 三项目标、ViLa 双尺度提示与源码/修正版比较见 VISION_LANGUAGE.md。',
              '   RankMix 的两骨干、监督初始化、源码/修正版与混合目标见 RANKMIX.md；IIB-MIL 缺失训练定义见 IIBMIL_AUDIT.md。',
              '   TPMIL 的原型 KL、cDP-MIL 公开训练路径及 INS 缺项见 PROTOTYPE_DP.md。',
              '   AdvMIL 四配置的区域协议、源码/修正版和标签隐藏见 ADVERSARIAL_SURVIVAL.md。',
              '   MFE-MIL 的五种分类/六种生存路线、共享重建目标及作者配方见 MFE_MIL.md。',
              '   MSMMIL 的三路 AFWMamba、共享注意力和完整 Lookahead/RAdam 配方见 MSMMIL.md。',
              '   MoMIL 三路序列、构造器修复、最后轮/验证选模和显式生存扩展见 MOMIL.md。',
              '   ReCo-MIL 四配置的训练专用原型、稀有实例、双尺度和原始调度见 RECOMIL.md。',
              '   MamMIL 的 Mamba2/SSD、多序列拓扑、源码/修正层循环与患者图扩展见 MAMMIL.md。',
              '   MambaMIL 的三个序列分支、显式扫描后端、完整分类/生存训练及修复见 MAMBAMIL.md。',
              '   DGMIL 的动态实例筛选、分布推理修复和原始排名指标见 DGMIL.md。',
              '   PAMoE 的组织原型、患者批训练、源码/修正索引和两骨干见 PAMOE.md。',
              '   PathOmics 两阶段、三分支、源码冻结及修复边界见 PATHOMICS.md。',
              '   MMP 的四种原型/融合配置、完整患者批训练和显式源码修复见 MMP.md。',
              '   PIBD 的原型瓶颈、联合 CLUB 估计器、患者批训练和 RNA 归一化见 PIBD.md。',
              '   MicroMIL 的冻结FM迁移、完整聚类/硬选择/图/BCE训练与DGL形状修复见 MICROMIL.md。',
              '   ProbSA-MIL 的作者库版本、概率注意力/图KL/周期训练和两种预测定义见 PROBSA.md。',
              '   GTP 的完整GCN/mincut/ViT、Adam调度、源码/修正路线和原生GraphCAM见 GTP.md。',
              '   SETMIL 的物理PPE、TPMF/iRPE、完整增强/AdamW配方与裁剪修正见 SETMIL.md。',
              '   StableMIL 的物理融合、区域语义/局部全局注意力、随机二维RoPE和完整分类/生存配方见 STABLEMIL.md。',
              '   MONAI 的完整RGB/ResNet训练、序数BCE、五条原生聚合路线及金字塔、AdamW/cosine配方见 MONAI_MIL.md。',
              '   Campanella 的原始CNN硬筛选、最佳编码器、专用循环聚合和完整两阶段原图配方见 CAMPANELLA.md。',
              '   MAMMOTH 的原始软路由/低秩专家、八分类骨干和完整抽样/AdamW/cosine配方、ILRA命名修正与源码边界见 MAMMOTH.md。',
              '   PMIL Shapley 的渐进伪包、完整多轮重启、六条源码/修正路线和主程序逐位对照见 PMIL_SHAPLEY.md。',
              '   Wibawa MultiAttentionMIL 的三层残差注意力、完整补充训练流程、显式冻结WSI适配及推理Dropout修正见 MSATTENTION.md。',
              '   TTIS 的六骨干、完整Lookahead训练、TEST专用FPS与特征融合、语义空间和聚类边界见 TTIS.md。',
              '   官方2DMambaMIL的真实物理网格、三条源码/修正路线、完整分类/生存配方与CUDA边界见 MAMBA2D.md。',
              '   SI-MIL的完整双分支、真实PathExpert对齐、TRAIN专用精确归一化、原始/完整VJP梯度及解释输出见 SIMIL.md。',
              '   HIPT的实际预训练教师、真实完整16×16物理网格、原生层级分类器、平衡抽样和两条完整训练配置见 HIPT.md。',
              '   H²-MIL的真实缩略图/两倍率特征与父子图、完整RAConv/IHPool和固定批训练、显式源码/分类修正及结构导出见 H2MIL.md。',
              '   C2Aug的原生TransMIL、三种跨包增强、完整双对比目标、EMA教师与TRAIN专用索引库及显式论文参考重建见 C2AUG.md。',
              '   GAT-Mamba的真实坐标/六类亚型图、原始GAT与Mamba1.2、完整Cox/L1/Adam/随机验证/plateau和最后并列选模、显式边条件修正与精确恢复见 GATMAMBA.md。',
              '   ProtoSurv的实际空间图/组织类别或原型登记、完整图/多原型融合与三项目标、原始患者采样/增强/累积/Adam/cosine和源码/显式修正配置见 PROTOSURV.md。',
              '   PathRWKV的完整TimeMix/ChannelMix、多任务分类/回归/生存配方、真实物理坐标、原生CUDA与显式张量后端、源码/损失及流式状态修正见 PATHRWKV.md。',
              '4. 构建并验证 CUDA 训练镜像；在有实际权重、数据、标签、患者划分和特征的条件下执行真实 BRCA/COAD 参考实验。',
              '5. 完成原始参考复现、预算匹配、多 FM 主矩阵和预注册统计分析。', '']
    (root/'docs/ADAPTATION_STATUS.md').write_text('\n'.join(lines))
    with (root/'docs/methods.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,lineterminator='\n',fieldnames=['id','name','group','status','source_url','source_commit','source_status','profiles','paper_reproduction','real_wsi_validation'])
        writer.writeheader()
        for m in methods:
            writer.writerow({k:';'.join(m[k]) if isinstance(m.get(k),list) else m.get(k) for k in writer.fieldnames})
    counts=Counter(m['group'] for m in methods)
    print(dict(counts))


if __name__=='__main__':
    main()
