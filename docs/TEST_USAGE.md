# YOLO模型测试使用指南

## 概述

`test_yolo_model.py` 脚本提供了在测试集上验证模型性能的完整功能，包括：
- 在测试集上运行验证，获取mAP等评估指标
- 批量处理测试图像，生成可视化结果
- 生成详细的测试报告

## 快速开始

### 基本使用

```bash
# 使用训练好的模型在测试集上验证
python test_yolo_model.py --model runs/defect_detection6/weights/best.pt
```

### 完整参数

```bash
python test_yolo_model.py \
    --model runs/defect_detection6/weights/best.pt \
    --data dataset/dataset.yaml \
    --test_images dataset/test/images \
    --output test_results \
    --device cuda \
    --conf 0.25 \
    --iou 0.45
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | **必需** | 模型路径 (.pt文件) |
| `--data` | `dataset/dataset.yaml` | 数据集配置文件路径 |
| `--test_images` | `dataset/test/images` | 测试图像目录 |
| `--test_labels` | `None` | 测试标签目录（可选） |
| `--output` | `test_results` | 输出目录 |
| `--device` | `cpu` | 设备 (cpu, cuda, 0, 1, etc.) |
| `--imgsz` | `640` | 图像尺寸 |
| `--conf` | `0.25` | 置信度阈值 |
| `--iou` | `0.45` | IoU阈值 |
| `--skip_validation` | `False` | 跳过验证步骤，只进行批量测试 |
| `--skip_batch` | `False` | 跳过批量测试，只进行验证 |

## 功能说明

### 1. 模型验证

脚本会在测试集上运行YOLO的验证功能，计算以下指标：
- **mAP50 (BBox)**: 边界框检测的mAP@0.5
- **mAP50-95 (BBox)**: 边界框检测的mAP@0.5:0.95
- **mAP50 (Mask)**: 分割掩码的mAP@0.5（如果是分割模型）
- **mAP50-95 (Mask)**: 分割掩码的mAP@0.5:0.95
- **Precision**: 精确率
- **Recall**: 召回率

### 2. 批量测试图像

脚本会：
- 处理测试目录中的所有图像
- 对每张图像进行检测
- 保存可视化结果（带检测框和标签的图像）
- 生成JSON格式的详细结果

### 3. 测试报告

脚本会生成：
- **test_report.md**: Markdown格式的测试报告
- **test_results.json**: JSON格式的详细结果
- **images/**: 所有可视化结果图像

## 输出文件说明

### test_results.json

包含所有测试图像的详细检测结果：

```json
{
  "model_path": "runs/defect_detection6/weights/best.pt",
  "test_images_dir": "dataset/test/images",
  "total_images": 34,
  "total_defects": 45,
  "avg_defects_per_image": 1.32,
  "conf_threshold": 0.25,
  "iou_threshold": 0.45,
  "timestamp": "2024-01-01T12:00:00",
  "results": [
    {
      "image_name": "DSC_0012_JPG.rf.74c8f7f3e1874ed84ad39ce3c570f46c.jpg",
      "num_defects": 2,
      "defects": [
        {
          "class_id": 0,
          "class_name": "dirt",
          "confidence": 0.85,
          "bbox": [100, 200, 50, 30],
          "bbox_xyxy": [100, 200, 150, 230],
          "area": 1500
        }
      ]
    }
  ]
}
```

### test_report.md

包含：
- 模型信息
- 验证指标
- 批量测试统计
- 按类别统计

### images/

每张测试图像的可视化结果，文件名格式：`{原文件名}_result.jpg`

## 使用示例

### 示例1：完整测试（验证 + 批量测试）

```bash
python test_yolo_model.py \
    --model runs/defect_detection6/weights/best.pt \
    --device cuda
```

### 示例2：只进行验证（不处理图像）

```bash
python test_yolo_model.py \
    --model runs/defect_detection6/weights/best.pt \
    --skip_batch
```

### 示例3：只进行批量测试（不验证）

```bash
python test_yolo_model.py \
    --model runs/defect_detection6/weights/best.pt \
    --skip_validation
```

### 示例4：调整检测阈值

```bash
python test_yolo_model.py \
    --model runs/defect_detection6/weights/best.pt \
    --conf 0.3 \
    --iou 0.5
```

### 示例5：指定不同的测试目录

```bash
python test_yolo_model.py \
    --model runs/defect_detection6/weights/best.pt \
    --test_images custom_test/images \
    --output custom_test_results
```

## 注意事项

1. **数据集配置**: 确保 `dataset.yaml` 中包含 `test:` 路径，或者使用 `--test_images` 参数指定测试图像目录

2. **GPU加速**: 如果有GPU，使用 `--device cuda` 可以大幅提升处理速度

3. **内存占用**: 批量处理大量图像时，注意内存占用。如果内存不足，可以分批处理

4. **置信度阈值**: 较低的置信度阈值会检测到更多目标，但可能包含更多误检。建议根据实际需求调整

## 常见问题

**Q: 验证时提示找不到test集？**
A: 检查 `dataset.yaml` 中是否包含 `test:` 配置，或者使用 `--skip_validation` 跳过验证步骤

**Q: 如何只测试部分图像？**
A: 可以创建一个临时目录，将需要测试的图像复制到该目录，然后使用 `--test_images` 指定该目录

**Q: 如何查看详细的检测结果？**
A: 查看 `test_results.json` 文件，其中包含每张图像的详细检测信息

**Q: 可视化结果在哪里？**
A: 在输出目录的 `images/` 子目录中，文件名格式为 `{原文件名}_result.jpg`

