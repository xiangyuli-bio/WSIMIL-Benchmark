# H 集群与 BRCA / COAD

## 当前环境证据

Workspace 中安装了本仓库的独立 CPU 测试环境 `.venv`，不是 GPU 训练环境。

2026-10-15 短作业 `milplus-envcheck-67561971` 实测用户此前使用的镜像
`registry.h.pjlab.org.cn/ailab/cuda:13.2.1-devel-ubuntu22.04-tools`：其默认 `python3`
中没有 torch、numpy、h5py、pandas、scikit-learn、PyYAML、einops。
因此不能仅选择 CUDA tools 镜像就直接运行 MIL。该检查正常退出，未保持占卡。

GPU 实验必须使用已安装本项目及 CUDA 版 PyTorch 的训练镜像或环境。仓库的 Dockerfile 是构建入口，当前未在 H 仓库发布镜像；CPU venv 也不能直接当作 GPU 环境。

## 本项目存储位置与挂载

用户指定本项目自己的写入根目录为 `/mnt/medeval-share/xiangyuli`，直接位于
`medeval-share` 下。其他人的 `zhonglanfeng` 目录仅作为已获授权的数据读取来源，
代码、特征缓存、日志、checkpoint 和新实验结果应写入自己的目录。

2026-10-22，当前目录已通过 SSHFS 连接到真实 GPFS。另有本任务自己的
临时挂载别名 `/mnt/xiangyuli-milplus-gpfs`，两处映射至同一
`medeval-share/xiangyuli`，已通过随机字节的跨挂载读写检查。新的共享验证归档写入
`xiangyuli/mil_baseline_puls_work`，不覆盖同目录下其他项目。
CPU全量回归的短期合成文件可使用开发机独立内存临时目录，启用空间/内存保护；
通过后的数值收据、日志和XML校验后归档到上述自己的共享目录。

本任务的专用桥接为闲时CPU RJob `xiangyuli-shared-async-bridge-v2-6022837`，
申请1CPU、1024MiB内存、0GPU，最多运行24小时，可能被抢占。服务只提供指向本用户
共享目录的SFTP，使用专用客户端密钥和经平台日志核对的服务器公钥。镜像自带的SSH
运行时不适合这个临时桥接，因此采用独立AsyncSSH服务。私钥不进入代码库或共享盘。

临时挂载依赖该CPU任务与当前开发机；GPFS中的文件不会因桥接结束而自动删除。
HDF5覆盖写和断点恢复需保持文件系统一致性；本次专用SSHFS挂载使用同步写入、关闭
属性/目录缓存和预读，并通过80次覆盖/修改检查及25项受影响的专项测试。
普通SSHFS默认缓存曾导致HDF5重写失败，失败日志保留，不能按模型通过结果登记。
长期使用应在保存当前任务后，从开发机平台配置原生GPFS挂载：

```text
源：gpfs://gpfs1/medeval-share/xiangyuli
目标：/mnt/medeval-share/xiangyuli
```

此目录已经存在，不需要新建存储卷。平台若要求关机或重启，先保存活动验证与开发
结果。目录存在本身不能证明挂载成功；普通mkdir也不能代替平台GPFS挂载。

## 已知原始数据范围

来自本工作区之前保存的 inventory（不是本仓库重新完成的病理标签审查）：

| 队列 | 原始根目录 | 匹配现有标签的 DX 切片 | 患者 | 候选终点 |
|---|---|---:|---:|---|
| TCGA-BRCA | `/mnt/medeval-share/zhonglanfeng/TCGA/TCGA-BRCA` | 856 | 856 | IDC 679 / ILC 177 |
| TCGA-COAD | `/mnt/medeval-share-2/zhonglanfeng/data/TCGA-COAD` | 438 | 432 | COAD 375 / MACR 63 |

这些数量来自文件和标签匹配，不代表通过了 WSI 解码或组织 QC。COAD 的已有形态标签不能用于宣称 MSI、KRAS 或生存分析。源共享目录作为输入，不往别人的数据根写实验结果。

## 先准备任务

在同时能访问 inventory 和源 CSV 的环境中执行。源 CSV 路径：
`/mnt/medeval-share/zhonglanfeng/data/TCGA-OT-csv/datasets/tcga-ot_{train,val,test}.csv`。
Workspace 若未挂载这些卷，在有对应 GPFS 挂载的任务中执行。

```bash
milplus prepare-tcga \
  --inventory /root/tcga_raw_wsi_inventory_20261014.csv \
  --project TCGA-BRCA --labels IDC ILC \
  --train-csv /mnt/medeval-share/zhonglanfeng/data/TCGA-OT-csv/datasets/tcga-ot_train.csv \
  --val-csv /mnt/medeval-share/zhonglanfeng/data/TCGA-OT-csv/datasets/tcga-ot_val.csv \
  --test-csv /mnt/medeval-share/zhonglanfeng/data/TCGA-OT-csv/datasets/tcga-ot_test.csv \
  --output /mnt/medeval-share/xiangyuli/milplus/tasks/brca_idc_ilc
```

COAD 对应 `--project TCGA-COAD --labels COAD MACR` 和独立输出目录。程序保留源患者划分，不重新随机拆分。若过滤后某子集缺类别，会拒绝并要求修订协议。

`prepare-tcga` 生成任务元数据、患者划分和 `wsis.csv`。此时还没有特征，所以不能训练。按 README 的 TRIDENT 流程提取后运行 `index-features`。

## 镜像、路径与任务提交

在联网 Workspace 构建 CUDA 训练镜像并推送到自己有上传权限的仓库；把代码与任务包放在自己实际挂载的持久存储中。默认 CPU 测试环境不能打包冒充 CUDA 环境。

```bash
docker build -t YOUR_AUTHORIZED_IMAGE:milplus-0.1.0 .
# 使用已授权的镜像仓库，完成登录后再推送。
docker push YOUR_AUTHORIZED_IMAGE:milplus-0.1.0

milplus hcluster \
  --repository /mnt/medeval-share/xiangyuli/mil_baseline_puls \
  --data /mnt/medeval-share/xiangyuli/milplus/tasks/brca_idc_ilc \
  --method abmil \
  --output /mnt/medeval-share/xiangyuli/milplus/runs/brca_abmil_uni_v1_seed2026 \
  --image YOUR_AUTHORIZED_IMAGE:milplus-0.1.0 \
  --gpu 1 --cpu 8 --memory-mib 32768 \
  --script submit_abmil.sh
bash submit_abmil.sh
```

生成器为当前用户的 namespace `ailab-medpeak`、公共组 `medai4s_gpu_pool` 和四个已知卷生成命令；可覆盖 namespace/group。这里不加入私有机器参数、不使用 S3 FUSE。CLI 内存单位按已检查的 brainpp 2.7.12.16 使用 MiB。

普通 MIL 实验通常一份配置一张 GPU；两张卡用于两个独立实验。`run-matrix --devices cuda:0 cuda:1` 需要任务实际分配两张卡。

启动命令运行真实训练，训练结束后 RJob 正常退出释放资源。代码、输出、日志与 checkpoint 都应位于明确的持久存储；容器本地 `/root` 不应成为唯一副本。

完整共享卷容量已较紧，先用少量 WSI 测量 patch 数与特征体积，再选择可用输出空间。
本阶段只有合成软件验证，尚未执行真实 WSI 特征提取；源数据目录未写入新结果。
