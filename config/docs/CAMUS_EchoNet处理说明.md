# CAMUS / EchoNet 原始数据与当前数据接入

依据：MemSAM官方仓库固定版本 [36f1b988f14b518a6346d2edb1267340387e646c](https://github.com/dengxl0520/MemSAM/tree/36f1b988f14b518a6346d2edb1267340387e646c)。核对文件为README、utils/preprocess_camus.py、utils/preprocess_echonet.py、utils/data_us.py和utils/utils_contour.py。MemSAM基于SAM1；TUM-US继续使用SAM2.1 Small，迁移数据协议与监督规则。

## 两套数据分别怎样处理

|事项|CAMUS|EchoNet-Dynamic|
|---|---|---|
|原始输入|patient目录、2CH/4CH NIfTI、Info_2CH.cfg/Info_4CH.cfg|Videos/*.avi、FileList.csv、VolumeTracings.csv|
|划分单位|patient；同一患者两种切面同分区|保留FileList.csv的TRAIN/VAL/TEST；不重新随机分配|
|MemSAM原始导出|读取half_sequence与half_sequence_gt，resize256，ED>ES时翻转视频和GT，保存C,T,H,W与fnum_mask|保留完整RGB视频，按描记CSV的实际Frame生成两个LV内腔mask，保存fnum_mask|
|当前本地标注|train/val 800序列均有每存储帧索引mask|train/val 8667序列，每序列恰好两个索引mask|
|默认TUM-US训练采样|全序列范围均匀取最多16个唯一帧；一个视图一个有界窗口|两个已标注帧的时间区间内取最多16个唯一帧，包含两端|
|有效监督|仅采到且有真实存储标注的帧|仅两个描记帧，区间中间帧mask_valid=false|
|开发/评价|多窗口覆盖全部已有标注帧，重复出现的帧只计一次监督，窗口重置记忆|默认描记区间；另提供完整视频协议作敏感性实验|
|默认可比目标|LV endo / LV内腔|LV endo / LV内腔|

MemSAM活动代码使用有序split()而不是已定义但未调用的random_split()：500患者时为前350 train、后50 val、最后100 test。原始脚本可用-f提供固定JSON。不能把SEED=42直接描述为实际随机划分。

MemSAM的load_video_and_mask_file按最早/最晚mask帧取区间；这只是时间顺序，字典顺序不能证明哪一个是ED或ES。原函数按固定frame_length取整可能重复帧。TUM-US保留唯一、递增的存储帧号；短区间保留实际帧数，不重复来凑16帧。太短的片段可能不能产生容量溢出效用状态，这是正常结果。

EchoNet区间协议利用已知标注帧的位置界定输入，属于离线标注区间基准；它不等同于无需标注自动定位心动周期。完整视频协议用于检验未裁剪上下文，两个协议的结果分别报告。所有记忆政策在各自协议内使用相同帧、相同读者和相同监督。

MemSAM视频读取器把sub_path固定为EchoNet并在classes==2时清除mask>1，所以其活动视频路径的CAMUS同样是LV内腔任务。TUM-US的config/cardiac_memsam.json与默认study.json采用共同LV内腔目标。补充config/cardiac_multiclass.json为CAMUS内腔、心肌、左房；CAMUS原始class2是心肌区域，使用原预留global ID24命名lv_myocardium，旧ID6保留为lv_epi。心肌环与外膜轮廓包围的整个区域不是同一个标签，不能直接互换。

多结构评价保留原整体Dice，并新增dice_by_target逐结构结果。旧配置中把class2交给ID6训练的检查点应保留原配置解释；新心肌ID24使用相应配置训练。

TUM-US使用自动目标query；训练真值只计算损失，不从真值生成输入点提示。与使用交互点的MemSAM结果比较时，需要统一提示协议后重跑，不能直接并列不同输入条件的论文分数。

## 接入已有的病例清单

当前数据已经是MemSAM式的.npy/.npz导出，不需要为了运行模型重新做一次NIfTI/AVI转换。npz里的fnum_mask键已提供mask对应存储帧号，可直接索引。这解决了此前“缺原始导出日志，所以分割帧不可用”的判断；缺的是原始采集方向/裁剪过程的额外证据，而不是当前数组内的对应。

在项目根目录运行：

```powershell
python .\utils\prepare.py --inventory 'data/dataset_task_manifest/dataset_task_manifest.jsonl' --config .\config\cardiac_memsam.json --index-cardiac --output 'run/cardiac_inputs'
```

该命令读取train/val的本地标注并核对视频头部：验证键不越界、mask与视频空间尺寸一致、EchoNet恰好两帧、类别值合法；保持原患者划分，生成cardiac_index/frame_map.json、inventory_trainval.jsonl、cardiac_audit.json及prepared_config.json。默认不打开test数组、不训练、不计算预测。失败时排除整个实体并登记，CAMUS两个视图一起处理。原数据不会修改。

也可分两步，供重复实验复用已索引文件：

```powershell
python .\utils\index_cardiac.py --data-root 'data' --inventory 'data/dataset_task_manifest/dataset_task_manifest.jsonl' --output 'run/cardiac_index'
python .\utils\prepare.py --inventory 'run/cardiac_index/inventory_trainval.jsonl' --frame-map 'run/cardiac_index/frame_map.json' --config .\config\cardiac_memsam.json --output 'run/cardiac_inputs'
```

本地完整train/val核查：CAMUS400患者/800序列/15468个索引mask，EchoNet8667序列/17334个索引mask；均未发现无效实体。CAMUS350/50患者、EchoNet7394/1273实体的训练/开发清单均已就绪。每帧mask存在是导出文件事实，不代表已独立核实每帧均由人工绘制。没有原始记录时，存储顺序也不能宣称等同于原始采集方向。

完成输入准备后，train.py --stage run/study/baselines可直接使用生成的prepared_config.json。EchoNet仍默认作为heldout_source，不自动加入readers训练；要单独训练其读者，可显式--stage readers --endpoints echonet并使用独立输出目录。

## 从原始CAMUS重新导出

MemSAM README原操作记录：

```powershell
python utils/preprocess_camus.py -i /path/CAMUS_public/database_nifti -o /path/CAMUS_public_processed
python utils/preprocess_echonet.py -i /path/EchoNet-Dynamic -o /path/EchoNet_processed
```

TUM-US也提供这两个入口，参数-i/-o一致。原始CAMUS目录应有：

```text
database_nifti/patient0001/
  patient0001_2CH_half_sequence.nii.gz
  patient0001_2CH_half_sequence_gt.nii.gz
  Info_2CH.cfg
  patient0001_4CH_half_sequence.nii.gz
  patient0001_4CH_half_sequence_gt.nii.gz
  Info_4CH.cfg
```

也支持_sequence.nii.gz；若没有序列GT，auto模式读取_ED_gt/_ES_gt，通过Info中的一基ED/ES索引映射到真实存储位置。

```powershell
python .\utils\preprocess_camus.py -i 'data/raw/CAMUS/database_nifti' -o 'data/derived/CAMUS' -f 'data/raw/CAMUS/split.json' --size 256 --mask-mode auto --temporal-order source
```

split.json示例结构：{"train":["patient0001"],"val":["patient0351"],"test":["patient0401"]}，填写全部实际患者，不允许跨分区。省略-f使用有序70/10/20划分；为了重现已有的划分，建议从现有清单导出固定患者列表再传-f。

默认保留source时间方向。若需要复现MemSAM的ED→ES规范化，传--temporal-order ed-to-es；ED>ES时视频和GT同步翻转，并记录stored→raw帧号。--mask-mode ed-es仅导出两相真实标注；--mask-mode dense要求存在同长度的序列GT。不会用ED/ES mask插值制造中间帧真值。

## 从原始EchoNet重新导出

```text
EchoNet-Dynamic/
  FileList.csv
  VolumeTracings.csv
  Videos/0X....avi
```

```powershell
python .\utils\preprocess_echonet.py -i 'data/raw/EchoNet-Dynamic' -o 'data/derived/EchoNet' --size 112
```

保留官方分区；同时支持FileList有/无.avi后缀。按Frame的零基整数索引分组，默认每帧21条描记，沿两侧边界反向闭合轮廓，排除第一条长轴记录，与MemSAM的polygon生成一致。保留全部视频帧，但只导出两个描记mask。EF/EDV/ESV仅保存CSV提供值，未在本模型中作为额外监督；不从字典顺序判断ED/ES，也不虚构物理spacing。

## 新导出文件与依赖

```text
output/
  videos/train/name.npy          # C,T,H,W，RGB
  annotations/train/name.npz     # mask[K,H,W]、frame_indices[K]，纯数字数组
  frame_map.json                 # storage帧号、mask帧号、有效位、raw索引
  inventory.jsonl               # 以output作为data_root
  preprocess_report.json        # 源路径、划分、几何、翻转与错误记录
```

新导出用mask+frame_indices代替需要pickle的fnum_mask，TUM-US同时支持两种格式。图像线性插值、mask最近邻；区别于上游CAMUS对两者均最近邻。CAMUS原始几何与resize后的spacing记录在报告中；EchoNet物理spacing标为unknown。默认转换train/val；需要导出test时显式加--splits train val test。导出脚本不覆盖已有视频/标注文件。

新增可选依赖为SimpleITK、opencv-python-headless、scikit-image，见pyproject.toml的preprocess extra。新环境可执行python -m pip install -e ".[preprocess]"。

使用新导出时，通过--data-root指定上述output目录，例如：

```powershell
python utils/prepare.py --data-root 'data/derived/CAMUS' --inventory 'data/derived/CAMUS/inventory.jsonl' --frame-map 'data/derived/CAMUS/frame_map.json' --config config/cardiac_memsam.json --output 'run/camus_raw_inputs'
```

该清单仅包含此单来源，可用train.py --stage readers --endpoints camus单独验证。现有多来源训练应使用各自的清单和共同的data_root；合并新导出时先将相对路径统一到共同data_root。原始NIfTI/AVI输入目录不直接用作已处理视频清单。

## 本轮调优范围

对相同采样和标签比较0/8MiB标注LRU缓存，按文件stat变化失效，缓存有上限。CAMUS有界训练窗口与EchoNet区间协议是监督协议调整，单独记录采样帧和标签数，不将少用训练帧包装成纯速度收益。
