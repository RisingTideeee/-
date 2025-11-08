# YOLOv11使用指南

## 概述

本项目已集成YOLOv11模型，可以替代传统的图像处理方法进行缺陷检测和分类。YOLOv11提供了更高的检测精度和端到端的检测-分类能力。

## 快速开始

### 1. 使用预训练模型（无需训练）

```bash
# 基本使用
python main.py --input image.jpg --use_yolo

# 指定模型大小（n/s/m/l/x，从小到大）
python main.py --input image.jpg --use_yolo --yolo_size s

# 使用分割模式（提供掩码）
python main.py --input image.jpg --use_yolo --yolo_task segment

# 调整检测阈值
python main.py --input image.jpg --use_yolo --conf_threshold 0.3 --iou_threshold 0.5
```

### 2. 使用自定义训练模型

首先训练模型（见下方），然后使用：

```bash
python main.py --input image.jpg --use_yolo --yolo_model runs/defect_detection/weights/best.pt
```

## 训练自定义模型

### 步骤1：准备数据集

YOLO需要的数据集格式：

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── val/
│       ├── image1.txt
│       └── ...
└── dataset.yaml
```

**标注文件格式（.txt）**：
每行一个目标，格式为：`class_id center_x center_y width height`
- 所有坐标都是归一化的（0-1之间）
- center_x, center_y是边界框中心点
- width, height是边界框的宽度和高度

**dataset.yaml格式**：
```yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: 划痕
  1: 漆点
  2: 凹痕
  3: 水渍
  4: 污点
```

### 步骤2：训练模型

```bash
# 基本训练
python train_yolo.py --data dataset.yaml --epochs 100

# 完整参数
python train_yolo.py \
    --data dataset.yaml \
    --model_size s \
    --task segment \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device cuda \
    --project runs \
    --name defect_detection
```

### 步骤3：验证模型

训练完成后，模型会自动在验证集上评估。结果保存在 `runs/defect_detection/` 目录。

## 模型大小选择

- **n (nano)**: 最小最快，适合实时应用，精度较低
- **s (small)**: 平衡速度和精度，推荐用于大多数场景
- **m (medium)**: 较高精度，速度适中
- **l (large)**: 高精度，速度较慢
- **x (xlarge)**: 最高精度，速度最慢

## 任务类型

- **detect**: 目标检测，只输出边界框和类别
- **segment**: 实例分割，输出边界框、类别和掩码

对于缺陷检测任务，推荐使用 `segment` 模式以获得更精确的分割结果。

## 参数说明

### 检测参数

- `--conf_threshold`: 置信度阈值（默认0.25），过滤低置信度检测
- `--iou_threshold`: IoU阈值（默认0.45），用于非极大值抑制（NMS）

### 训练参数

- `--epochs`: 训练轮数（默认100）
- `--batch`: 批次大小（默认16），根据GPU内存调整
- `--imgsz`: 输入图像尺寸（默认640）
- `--device`: 训练设备（'cpu', 'cuda', '0', '1'等）

## 输出结果

使用YOLO模式时，会生成以下文件：

- `yolo_detection_result.jpg`: 检测结果可视化（包含边界框、标签、置信度）
- `segmented_mask.jpg`: 分割掩码（如果使用segment模式）
- `segmentation_result.jpg`: 分割结果可视化

## 性能优化建议

1. **GPU加速**: 使用CUDA可以大幅提升训练和推理速度
   ```bash
   python main.py --input image.jpg --use_yolo  # 自动使用GPU（如果可用）
   ```

2. **模型量化**: 训练后可以导出为ONNX或TensorRT格式以提升推理速度
   ```python
   from yolo_segmenter import YOLODefectDetector
   detector = YOLODefectDetector(model_path="model.pt")
   detector.export_model(format='onnx')
   ```

3. **批量处理**: 对于多张图像，可以使用批量检测
   ```python
   from yolo_segmenter import YOLODefectDetector
   detector = YOLODefectDetector()
   results = detector.predict_batch(images)
   ```

## 与传统方法对比

| 特性 | 传统方法 | YOLOv11 |
|------|---------|---------|
| 检测精度 | 中等 | 高 |
| 分类能力 | 需要额外训练 | 端到端 |
| 速度 | 快 | 快（GPU） |
| 数据需求 | 少 | 需要标注数据 |
| 适用场景 | 简单缺陷 | 复杂缺陷 |

## 常见问题

**Q: 如何将现有标注转换为YOLO格式？**
A: 可以使用 `prepare_yolo_dataset.py` 脚本，或使用标注工具（如LabelImg）直接导出YOLO格式。

**Q: 预训练模型可以直接使用吗？**
A: 预训练模型是在通用数据集上训练的，对于特定缺陷类型，建议使用自己的数据微调。

**Q: 如何提高检测精度？**
A: 
1. 增加训练数据
2. 使用更大的模型（s→m→l→x）
3. 调整数据增强策略
4. 增加训练轮数

**Q: 模型文件太大怎么办？**
A: 可以使用模型量化或剪枝技术，或使用较小的模型尺寸（n或s）。

## 示例

完整的使用示例：

```bash
# 1. 准备数据集（手动或使用脚本）
python prepare_yolo_dataset.py --images ./images --annotations ./annotations --output ./dataset --classes "划痕,漆点,凹痕"

# 2. 训练模型
python train_yolo.py --data ./dataset/dataset.yaml --epochs 100 --batch 16 --device cuda

# 3. 使用训练好的模型
python main.py --input test.jpg --use_yolo --yolo_model runs/defect_detection/weights/best.pt --yolo_task segment
```

