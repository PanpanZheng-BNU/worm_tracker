# Worm-Tracker

> 一个用于追踪多只*秀丽隐杆线虫*（*C. elegans*）在行为实验中位置的工具箱。

## 目录

- [0. 前置要求](#0-前置要求)
  - [0.1 命名规范](#01-命名规范)
  - [0.2 Python环境](#02-python环境)
  - [0.3 感兴趣区域（ROI）](#03-感兴趣区域roi)
  - [0.4 校正](#04-校正)
- [1. 快速开始](#1-快速开始)
- [2. 主要功能](#2-主要功能)
- [3. 输出结果](#3-输出结果)
- [4. 高级用法](#4-高级用法)
- [5. 工作原理](#5-工作原理)

## 0. 前置要求

### 0.1 命名规范

拥有一致且定义明确的命名规范可以帮助防止奇怪的错误。

- `N2_group_X.avi`: 行为实验的视频录像。`N2` 表示*秀丽隐杆线虫*的品系，`group` 指实验组，`X` 是培养皿的编号。
- `date_correcting.avi`: 用于校正的视频，日期为 `date`。

**示例**:
```
N2_control_1.avi    # N2品系，对照组，1号培养皿
N2_control_2.avi
N2_test_1.avi       # N2品系，测试组，1号培养皿
2024.11.04_correcting.avi  # 校正视频
```

### 0.2 Python环境

通过 `conda` 创建Python虚拟环境：

#### 步骤 0: 安装 Anaconda 或 Miniconda

确保您已安装 [anaconda](https://anaconda.com/) 或 [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)。

#### 步骤 1: 创建环境

请确保您在正确的目录中，并在终端（或Windows上的CMD/PowerShell）中执行以下命令来创建新环境：

```bash
conda env create -f environment.yml
```

#### 步骤 2: 激活环境

```bash
conda activate worm-tracker
```

**环境包含的依赖**:
- Python 3.9
- NumPy
- OpenCV
- Plotly
- Pandas
- progressbar2
- Matplotlib

### 0.3 感兴趣区域（ROI）

> [!NOTE]
> 我们仅在感兴趣区域（ROI）内追踪线虫，并丢弃任何落在其外的轨迹。

#### 0.3.1 ROI 中心

请从视频录像中提取一帧，并手动用<span style="color:red; font-weight:bold">红色圆圈</span>标记培养皿的中心，如下所示：

<img src="./.imgs/ROI_Center.png" style="zoom:25%;" alt="ROI 中心" />

> [!CAUTION]
> 在提取和标注过程中请保持帧分辨率不变。

**标注要求**:
- 文件名必须与视频文件名匹配（仅扩展名不同）
  - 视频: `N2_control_1.avi`
  - ROI标注图: `N2_control_1.jpg`
- 使用红色（HSV颜色空间中的红色范围）标记圆圈
- 圆圈应标记在培养皿的中心位置

#### 0.3.2 ROI 半径

ROI半径参数（定义如下图所示）通过执行 `main.py` 时的输入参数确定。（默认半径为900像素）

<figure style="text-align:center; margin: 0 auto;">
<img src="./.imgs/P2.svg" style="width:80%;" alt="ROI半径定义" />
<em>ROI半径的定义</em>
</figure>

**设置ROI半径示例**:
```bash
python main.py --p2vs "包含视频的文件夹路径" --radius 800
```

> [!TIP]
> ROI实际上是**矩形区域**，而不是圆形：`[roi_x ± radius, roi_y ± radius]`

### 0.4 校正

运行校正程序：

```bash
python utils/correcting.py --p2v "视频路径"
```

## 1. 快速开始

### 准备输入文件

在开始之前，确保您有：

1. **视频文件**: 放在同一个文件夹中
   ```
   /实验数据/2024.11.04/
   ├── N2_control_1.avi
   ├── N2_control_2.avi
   └── N2_test_1.avi
   ```

2. **ROI标注图**: 每个视频对应一个jpg文件
   ```
   /实验数据/2024.11.04/
   ├── N2_control_1.avi
   ├── N2_control_1.jpg  ← 红色圆圈标记培养皿中心
   ├── N2_control_2.avi
   ├── N2_control_2.jpg
   ├── N2_test_1.avi
   └── N2_test_1.jpg
   ```

### 运行完整追踪流程

#### 步骤 1: 运行检测和初步追踪

```bash
python main.py --p2vs "/实验数据/2024.11.04/" --radius 900 --date "11.04"
```

此步骤将生成初步的轨迹片段，但这些片段可能是断断续续的。

#### 步骤 2: 链接轨迹片段为完整线虫轨迹

运行 `find_worm.ipynb` notebook 来将轨迹片段链接成完整的线虫轨迹：

1. 在 Jupyter Notebook 或 JupyterLab 中打开 `find_worm.ipynb`
2. 修改 notebook 中的分析文件夹路径：
   ```python
   # 指定要分析的培养皿文件夹
   p2trackers = os.listdir("./simple_trackers_result/")
   analysis_folder = p2trackers[0]  # 或手动指定，如 "n2_control_1"
   ```
3. 依次运行所有单元格：
   - **Part I**: 加载轨迹并转换为精细轨迹
   - **Part II**: 识别初始稳定的线虫（需根据可视化图确定线虫数量）
   - **Part III**: 链接完整线虫轨迹并保存验证结果

> [!IMPORTANT]
> 在 Part II 中，需要根据可视化图（每帧线虫数量图）的最高点来确定培养皿中的线虫数量，然后修改：
> ```python
> all_ini = find_initial(long_dfs, 10, new_summarize)  # 10 为线虫数量
> ```

### 查看结果

处理完成后，您将在以下位置找到结果：

```
simple_trackers_result/
├── n2_control_1/
│   ├── trackers.json      # 所有线虫的轨迹数据
│   ├── long_dfs.csv        # 每帧的检测结果
│   └── centroids.txt       # ROI中心坐标
├── n2_control_2/
│   └── ...
└── n2_test_1/
    └── ...
```

## 2. 主要功能

### 2.1 核心流程

Worm-Tracker 包含四个主要阶段：

```
视频文件 → [1.检测] → 每帧CSV → [2.初步追踪] → 轨迹片段 → [3.轨迹链接] → 完整轨迹 → [4.分析]
```

- **阶段 1 (检测)**: 使用背景减除检测每帧中的线虫位置
- **阶段 2 (初步追踪)**: 使用 IoU 算法生成初步的轨迹片段（`main.py`）
- **阶段 3 (轨迹链接)**: 将断断续续的轨迹片段链接成完整的线虫轨迹（`find_worm.ipynb`）
- **阶段 4 (分析)**: 对完整的线虫轨迹进行下游分析

### 2.2 命令行参数

#### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--p2vs` | 包含视频的目录路径 | `"/data/2024.11.04/"` |

#### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--radius` | 900 | ROI半径（像素） |
| `--date` | 当前日期 | 日期字符串（格式：MM.DD） |
| `--pool` | 4 | 并行处理的进程数 |
| `--p2det` | `./detect_results` | 检测结果输出路径 |
| `--p2trackers` | `./simple_trackers_result` | 追踪结果输出路径 |
| `--vis` | false | 处理过程中显示可视化 |
| `--img` | false | 保存每帧的标注图像 |
| `--video` | false | 保存标注后的视频 |

### 2.3 使用示例

**基本使用**:
```bash
python main.py --p2vs "/data/videos/"
```

**自定义ROI和日期**:
```bash
python main.py --p2vs "/data/videos/" --radius 850 --date "11.10"
```

**保存可视化结果**:
```bash
python main.py --p2vs "/data/videos/" --video true --img true
```

**调整并行处理**:
```bash
python main.py --p2vs "/data/videos/" --pool 8
```

## 3. 输出结果

### 3.1 目录结构

```
项目根目录/
├── detect_results/              # 检测阶段的中间结果
│   └── n2_control_1_11.04/
│       ├── csv/                 # 每帧的检测数据（处理后会删除）
│       ├── centroids.txt        # ROI中心坐标
│       └── sample_img_*.jpg     # 随机采样的帧图像
│
└── simple_trackers_result/      # 最终追踪结果
    └── n2_control_1/
        ├── trackers.json        # 轨迹数据
        ├── long_dfs.csv         # 完整检测数据
        └── centroids.txt        # ROI中心坐标
```

### 3.2 输出文件说明

#### main.py 输出文件

##### trackers.json

包含所有初步检测到的轨迹片段信息：

```json
{
  "0": {
    "start_frame": 100,
    "end_frame": 500,
    "bboxes": [[x, y, w, h], ...],      // 每帧的边界框
    "centroids": [[cx, cy], ...]         // 每帧的质心坐标
  },
  "1": {
    "start_frame": 105,
    "end_frame": 480,
    ...
  }
}
```

**字段说明**:
- `start_frame`: 轨迹开始的帧号
- `end_frame`: 轨迹结束的帧号
- `bboxes`: 边界框列表，格式为 `[x, y, width, height]`
- `centroids`: 质心坐标列表，格式为 `[x, y]`

> [!NOTE]
> 这些是初步的轨迹片段，可能是断断续续的，需要通过 `find_worm.ipynb` 进一步处理。

##### long_dfs.csv

每帧的原始检测数据：

| 列名 | 说明 |
|------|------|
| `frame` | 帧号 |
| `x`, `y` | 边界框左上角坐标 |
| `w`, `h` | 边界框宽度和高度 |
| `cX`, `cY` | 质心坐标 |

##### centroids.txt

ROI中心坐标，格式：`[x, y]`

例如：`[960, 540]`

#### find_worm.ipynb 输出文件

运行 `find_worm.ipynb` 后，会在分析的培养皿文件夹中生成以下最终结果文件：

##### worms.json

包含所有有效线虫的完整轨迹数据，每只线虫一个条目。

**结构说明**:
- 每只线虫的轨迹由多个 tracker 片段链接而成
- 包含完整的起止帧信息和位置数据
- 只保留通过验证的有效线虫（满足时长、位置等条件）

##### valid_worms_info.csv

记录每只有效线虫的统计信息：

| 列名 | 说明 |
|------|------|
| `worm_id` | 线虫编号 |
| `start_frame` | 起始帧号 |
| `end_frame` | 结束帧号 |
| `total_frames` | 总帧数 |
| `trackers_used` | 使用的 tracker 片段数量 |

> [!TIP]
> 这些是最终的分析结果，可以直接用于下游的行为分析和统计。

## 4. 高级用法

### 4.1 单独运行检测

如果只想运行检测阶段：

```python
from detector.concat_videos import find_all_videos
from detector.detector import detect

# 查找所有视频
videos = find_all_videos("/path/to/videos/")

# 检测单个视频
detect(videos[0], "./detect_results", radius=900, vis=False,
       imgs=False, video=False, date="11.04")
```

### 4.2 单独运行追踪

如果已有检测结果，只想运行追踪：

```python
from tracker.iou import simple_iou_tracker
import pandas as pd

# 加载检测结果
all_dfs = [pd.read_csv(f"frame_{i}.csv") for i in range(1, 1001)]

# 运行追踪
trackers = simple_iou_tracker(all_dfs, t_min=10, sigma_iou=0.3)
```

**参数说明**:
- `t_min`: 最小轨迹长度（帧数），短于此长度的轨迹将被丢弃
- `sigma_iou`: IoU阈值，用于判断两帧之间的目标是否为同一个线虫

### 4.3 轨迹链接详细使用

在 `find_worm.ipynb` 中手动调整参数以获得最佳结果：

#### 步骤 1: 加载和可视化

```python
# 加载轨迹数据
all_trackers, long_dfs, centroid = load_subj('simple_trackers_result', analysis_folder)
new_trackers, new_summarize = trackers2fine(all_trackers, long_dfs)

# 可视化每帧线虫数量
long_dfs.groupby('frame').size().plot()
```

根据可视化图确定培养皿中的线虫数量（取最高点的值）。

#### 步骤 2: 识别初始线虫

```python
# 第二个参数为线虫数量，根据上一步的可视化结果调整
all_ini = find_initial(long_dfs, 10, new_summarize)  # 10 为线虫数量
ini_indx = all_ini.sort_values(by='start_frame').tracker_id.values
```

#### 步骤 3: 链接和验证

```python
# 链接完整轨迹
worms = find_worms(ini_indx, new_summarize)

# 验证线虫轨迹
# 参数说明：
# - 870: ROI半径（应与main.py中的--radius参数一致）
# - start_frame=600: 线虫必须在此帧之前开始（过滤晚出现的线虫）
valid_worms = diagnosis_worms_square(worms, new_summarize, centroid,
                                     long_dfs, 870, start_frame=600)

# 保存结果
write_results(analysis_folder, valid_worms, new_trackers, centroid, 900, shape='square')
```

**关键参数调整**:
- `find_initial()` 的第二个参数：培养皿中的线虫数量
- `diagnosis_worms_square()` 的 `radius` 参数：ROI半径，应与 `main.py` 的 `--radius` 一致
- `diagnosis_worms_square()` 的 `start_frame` 参数：线虫必须在此帧之前开始才被认为有效

### 4.4 批量处理多个日期的数据

```bash
#!/bin/bash
dates=("11.04" "11.05" "11.06")
for date in "${dates[@]}"; do
    python main.py --p2vs "/data/2024.${date}/" --date "${date}"
done
```

### 4.5 提取视频帧

```bash
python utils/extract_img.py
```

### 4.6 检测ROI圆圈

单独测试ROI检测功能：

```bash
python utils/detect_circle.py
```

修改 `detect_circle.py` 中的 `image_path` 变量来指定图像路径。

## 5. 工作原理

### 5.1 检测算法

**步骤**:

1. **背景减除**: 使用KNN背景减除器检测运动物体
   ```python
   cv2.createBackgroundSubtractorKNN(history=3500, dist2Threshold=80)
   ```

2. **形态学操作**: 使用3×3椭圆核进行开运算和闭运算，去除噪声

3. **轮廓检测**: 查找所有轮廓并过滤
   - 面积范围: 5-250像素
   - 对应线虫的典型大小

4. **特征提取**: 计算每个检测到的线虫的：
   - 边界框 `(x, y, w, h)`
   - 质心坐标 `(cX, cY)`

### 5.2 追踪算法

使用**简单的IoU（交并比）追踪**算法：

1. **初始化**: 第一帧的所有检测创建新轨迹

2. **帧间匹配**: 对于每个新帧：
   - 计算当前检测与上一帧轨迹的IoU
   - 选择IoU最大的匹配（贪心策略）
   - 如果 IoU ≥ 阈值（默认0.3），则延续轨迹
   - 否则创建新轨迹

3. **轨迹终止**: 无法匹配的轨迹被终止

4. **后处理**: 过滤掉长度 < `t_min`（默认10帧）的短轨迹

**优点**:
- 简单高效
- 无需训练
- 适合线虫运动速度相对较慢的场景

**局限**:
- 线虫重叠时可能产生ID切换
- 不适合高速运动或频繁交叉的场景
- 生成的轨迹可能是断断续续的片段

### 5.3 轨迹链接算法

轨迹链接算法（在 `find_worm.ipynb` 中实现）用于将 IoU 追踪生成的断断续续的轨迹片段链接成完整的线虫轨迹。

**主要步骤**:

1. **轨迹细化** (`trackers2fine()`):
   - 将原始 tracker 转换为更精细的数据结构
   - 生成轨迹摘要（开始帧、结束帧、帧数等）

2. **识别初始线虫** (`find_initial()`):
   - 基于帧覆盖率和稳定性识别初始的稳定轨迹
   - 返回最可能代表真实线虫的初始 tracker 列表
   - 需要指定培养皿中的线虫数量

3. **轨迹链接** (`find_worms()`):
   - 从初始 tracker 开始，向前和向后搜索可能的连接
   - 基于时间和空间距离判断两个 tracker 是否属于同一只线虫
   - 构建完整的线虫轨迹路径

4. **轨迹验证** (`diagnosis_worms_square()`):
   - 检查线虫轨迹的有效性：
     - 起始时间要求（`start_frame` 参数）
     - 结束时间要求（必须追踪到视频末尾）
     - ROI 边界约束（轨迹不能过于接近边界）
   - 过滤掉不符合要求的轨迹

5. **结果保存** (`write_results()`):
   - 保存有效线虫的完整轨迹
   - 生成统计信息文件

**优点**:
- 解决了简单 IoU 追踪的 ID 切换问题
- 生成更长、更稳定的轨迹
- 通过验证步骤确保轨迹质量

**参数调整建议**:
- 如果线虫数量识别不准确，调整可视化图解读或 `find_initial()` 的线虫数量参数
- 如果有效线虫太少，降低 `start_frame` 要求或调整 ROI 半径
- 如果轨迹链接效果不佳，可能需要调整 `find_worms()` 中的距离阈值

### 5.4 ROI检测原理

**红色圆圈检测流程**:

1. **颜色空间转换**: BGR → HSV

2. **颜色掩码**: 提取红色区域
   - 低红色范围: `[0, 100, 100]` 到 `[10, 255, 255]`
   - 高红色范围: `[160, 100, 100]` 到 `[179, 255, 255]`

3. **去噪**: 高斯模糊（9×9核）

4. **霍夫圆检测**: 检测圆形
   ```python
   cv2.HoughCircles(blurred_mask, cv2.HOUGH_GRADIENT,
                    dp=1, minDist=50, param1=50, param2=30,
                    minRadius=5, maxRadius=200)
   ```

5. **返回中心**: 检测到的圆圈中心坐标

## 常见问题

### Q1: 为什么检测不到线虫？

**可能原因**:
1. ROI标注文件缺失或命名不匹配
2. ROI半径设置不当
3. 视频对比度太低
4. 线虫太小或太大（不在5-250像素范围内）

**解决方案**:
- 检查 `.jpg` 文件是否存在且命名正确
- 调整 `--radius` 参数
- 尝试调整 `detector.py` 中的背景减除器参数

### Q2: 轨迹频繁中断怎么办？

**可能原因**:
- IoU阈值太高
- 线虫运动速度太快

**解决方案**:
- 降低 `sigma_iou` 参数（在 `tracker/simple_tracking.py` 中修改）
- 降低 `t_min` 参数以保留更多短轨迹

### Q3: 如何提高处理速度？

**方法**:
1. 增加并行进程数: `--pool 8`
2. 关闭可视化: `--vis false`
3. 不保存中间图像: `--img false --video false`

### Q4: 输出文件在哪里？

**位置**:
- 中间结果: `./detect_results/`
- 最终结果: `./simple_trackers_result/`
- 可通过 `--p2det` 和 `--p2trackers` 参数自定义

### Q5: find_worm.ipynb 中找到的有效线虫数量太少怎么办？

**可能原因**:
1. `start_frame` 参数设置过严格
2. ROI 半径参数与实际不符
3. 线虫数量参数设置不准确

**解决方案**:
- 增大 `diagnosis_worms_square()` 的 `start_frame` 参数值（例如从 600 改为 1000）
- 确保 `diagnosis_worms_square()` 的 `radius` 参数与 `main.py` 中的 `--radius` 一致
- 检查可视化图，重新确认 `find_initial()` 的线虫数量参数
- 降低验证条件的严格程度

### Q6: 如何确定培养皿中有多少只线虫？

**方法**:

在 `find_worm.ipynb` 的 Part I 中运行：

```python
long_dfs.groupby('frame').size().plot()
```

这会生成每帧检测到的线虫数量图。取图中的**最高点**的值作为培养皿中的线虫数量。

**注意**:
- 如果图中最高点不稳定（波动很大），取稳定阶段的平均最高值
- 这个数量会用于 Part II 的 `find_initial()` 函数

### Q7: 为什么有些线虫的轨迹没有被链接起来？

**可能原因**:
1. 轨迹片段之间的时间或空间间隔太大
2. 初始线虫识别不准确
3. 线虫在视频中出现或消失的时间不合适

**解决方案**:
- 检查并调整 `find_worms()` 函数中的距离阈值参数
- 重新审查可视化图，确保线虫数量参数正确
- 降低 `diagnosis_worms_square()` 的验证条件
- 检查视频质量，确保线虫在整个录制期间都可见

## 技术支持

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 查看代码文档
- 阅读 `CLAUDE.md` 了解更多技术细节

## 许可证

请根据实际情况添加许可证信息。
