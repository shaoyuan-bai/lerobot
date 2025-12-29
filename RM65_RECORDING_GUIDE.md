# RM65 数据录制指南

本指南提供两种方法为RM65双臂机器人录制训练数据。

## 方法对比

### **方法1: 直接录制 (推荐)** ✅
使用 `record_rm65_demo.py` 直接生成LeRobot v3.0标准格式
- **优点**: 一步到位,数据格式标准,视频自动编码为MP4
- **缺点**: 需要配置LeRobot环境
- **适用**: 新录制数据

### **方法2: 先录后转**
使用自定义脚本录制,再用 `convert_custom_to_lerobot.py` 转换
- **优点**: 录制脚本简单,不依赖LeRobot
- **缺点**: 需要两步操作,占用更多存储空间
- **适用**: 已有自定义格式数据需要转换

---

## 方法1: 直接录制 (推荐)

### 1. 在Jetson上同步代码

```bash
cd ~/lerobot
git pull origin main
git checkout origin/main -- record_rm65_demo.py record_rm65.sh
```

### 2. 配置参数

编辑 `record_rm65.sh`:

```bash
HF_USER="woosh"  # 改为你的Hugging Face用户名
NUM_EPISODES=10  # 录制集数
EPISODE_TIME_S=30  # 每集时长(秒)
FPS=30  # 帧率
```

### 3. 开始录制

```bash
bash record_rm65.sh
```

或者直接使用Python:

```bash
python record_rm65_demo.py \
  --repo_id woosh/rm65_demo \
  --num_episodes 10 \
  --episode_duration 20 \
  --fps 30
```

### 4. 录制流程

1. **准备**: 确保机械臂已开机并连接
2. **起始位置**: 按提示移动机械臂到起始位置
3. **录制**: 按回车后,按住使能按钮手动拖动机械臂演示任务
4. **重复**: 完成后重复步骤2-3,直到录制完所有episode

### 5. 输出格式

数据保存在 `~/.cache/huggingface/lerobot/{user}/{dataset_name}/`:

```
rm65_demo/
├── data/
│   └── chunk-000/
│       └── file-000.parquet       # 主数据(关节+时间戳)
├── meta/
│   ├── info.json                  # 数据集信息
│   ├── stats.json                 # 统计信息
│   ├── tasks.parquet              # 任务列表
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet   # Episode索引
└── videos/
    └── observation.images.top/
        └── chunk-000/
            └── file-000.mp4       # 相机视频(H.264编码)
```

---

## 方法2: 转换现有数据

如果你已经用自定义脚本录制了数据(格式如 `outputs/rm65_recordings/`),可以使用转换脚本。

### 1. 在Jetson上同步代码

```bash
cd ~/lerobot
git pull origin main
git checkout origin/main -- convert_custom_to_lerobot.py
```

### 2. 运行转换

```bash
python convert_custom_to_lerobot.py \
  --input_dir outputs/rm65_recordings \
  --repo_id woosh/rm65_converted \
  --fps 30
```

### 3. 参数说明

- `--input_dir`: 自定义格式数据集路径
- `--repo_id`: 输出的LeRobot数据集ID
- `--output_dir`: (可选)输出路径,默认 `~/.cache/huggingface/lerobot/{repo_id}`
- `--fps`: 帧率,默认30

### 4. 转换过程

脚本会:
1. 读取 `dataset_summary.json` 获取episode列表
2. 遍历每个episode:
   - 读取 `states.json` (关节角度)
   - 读取 `images/top_*.jpg` (图像)
3. 使用LeRobot API写入标准格式:
   - 生成 `data/chunk-000/file-000.parquet`
   - 编码视频为 `videos/observation.images.top/chunk-000/file-000.mp4`
   - 生成元数据文件

---

## 数据验证

### 检查数据集结构

```bash
# 查看目录结构
tree ~/.cache/huggingface/lerobot/woosh/rm65_demo -L 3

# 查看parquet文件
python -c "
import pyarrow.parquet as pq
table = pq.read_table('~/.cache/huggingface/lerobot/woosh/rm65_demo/data/chunk-000/file-000.parquet')
print(table.schema)
print(f'Total rows: {len(table)}')
"

# 查看视频
ffplay ~/.cache/huggingface/lerobot/woosh/rm65_demo/videos/observation.images.top/chunk-000/file-000.mp4
```

### 加载数据集测试

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="woosh/rm65_demo", root="~/.cache/huggingface/lerobot/woosh/rm65_demo")
print(f"Total frames: {len(dataset)}")
print(f"Episodes: {dataset.num_episodes}")

# 读取第一帧
frame = dataset[0]
print(f"Observation keys: {frame['observation'].keys()}")
print(f"Action keys: {frame['action'].keys()}")
```

---

## 上传到Hugging Face Hub

```bash
# 登录(如果还没登录)
huggingface-cli login

# 上传数据集
huggingface-cli upload woosh/rm65_demo ~/.cache/huggingface/lerobot/woosh/rm65_demo
```

---

## 常见问题

### Q: 录制时相机无法打开?
A: 检查相机是否被占用:
```bash
ls -l /dev/video*
# 确保只有一个相机,索引为0
```

### Q: 录制时帧率不稳定?
A: 降低FPS或图像分辨率,编辑 `record_rm65_demo.py`:
```python
OpenCVCameraConfig(
    index_or_path=0,
    fps=15,  # 降低到15fps
    width=320,  # 降低分辨率
    height=240,
)
```

### Q: 转换脚本报错"找不到states.json"?
A: 确保输入目录结构正确:
```
outputs/rm65_recordings/
├── dataset_summary.json  # 必须存在
├── episode_0000/
│   ├── metadata.json
│   ├── states.json       # 必须存在
│   └── images/           # 必须存在
│       └── top_*.jpg
```

### Q: 如何修改机械臂IP地址?
A: 编辑 `record_rm65_demo.py` 第47-48行:
```python
left_arm_ip="169.254.128.20",   # 左臂IP
right_arm_ip="169.254.128.21",  # 右臂IP
```

---

## 下一步

录制完数据后,你可以:

1. **可视化数据集**: `python -m lerobot.visualize --repo_id woosh/rm65_demo`
2. **训练策略**: `python -m lerobot.train --dataset.repo_id woosh/rm65_demo`
3. **评估策略**: `python -m lerobot.eval --policy.path outputs/train/...`

查看官方文档了解更多: https://github.com/huggingface/lerobot
