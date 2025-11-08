# 汽车漆面缺陷检测系统

本项目实现了基于图像处理和深度学习的汽车漆面缺陷检测系统，支持传统图像处理方法和YOLOv11深度学习模型两种模式。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 使用预训练模型进行检测

```bash
python main.py --input <图像路径> --mode all --use_yolo
```

### 3. 训练自定义模型

```bash
python train_yolo.py --data dataset/dataset.yaml --epochs 100
```

### 4. 测试模型性能

```bash
python test_yolo_model.py --model runs/defect_detection6/weights/best.pt
```

### 5. 测试模型性能

```bash
# 测试模型并生成详细报告
python test_yolo_model.py --model runs/defect_detection6/weights/best.pt

# 查看测试结果
python show_results.py
```

### 6. K折交叉验证训练

```bash
# 5折交叉验证训练
python train_kfold_cv.py --data dataset/dataset.yaml --k 5 --epochs 100
```

### 7. 分析训练结果

```bash
# 分析训练结果，找出最佳epoch
python scripts/analyze_training.py
```

## 项目结构

- **根目录**: 主要功能脚本（main.py, train_yolo.py, test_yolo_model.py等）
- **docs/**: 详细文档和使用指南
- **models/**: 预训练模型文件
- **scripts/**: 工具脚本（分析、批量处理等）
- **dataset/**: 数据集文件
- **runs/**: 训练结果和模型权重
- **output/**: 输出结果

## 主要功能

1. **图像增强**: 直方图均衡化、CLAHE、对比度拉伸等
2. **缺陷检测**: 
   - 传统方法：Bradley-Roth自适应阈值、形态学操作
   - 深度学习方法：YOLOv11目标检测和实例分割
3. **特征提取与分类**: Gabor滤波器、LBP特征、SVM分类器

## 详细文档

- [完整使用指南](docs/USAGE.md)
- [YOLO使用指南](docs/YOLO_USAGE.md)
- [测试使用指南](docs/TEST_USAGE.md)
- [图像增强使用指南](docs/ENHANCEMENT_USAGE.md)
- [批量增强使用指南](docs/BATCH_ENHANCE_USAGE.md)

## 工具脚本

- **`scripts/analyze_training.py`**: 分析训练结果，找出最佳epoch
- **`scripts/batch_enhance.py`**: 批量图像增强工具
- **`scripts/compare_image_quality.py`**: 对比增强前后的图像质量
- **`scripts/prepare_yolo_dataset.py`**: 准备YOLO数据集
- **`scripts/test_system.py`**: 系统测试脚本
- **`scripts/example.py`**: 示例演示脚本

## 验证结果查看

验证结果保存在以下位置：

1. **控制台输出**: 运行 `test_yolo_model.py` 时直接显示
2. **训练结果**: `runs/defect_detection*/results.csv` - 包含所有训练轮次的指标
3. **测试结果**: `test_results/` - 包含测试报告和结果图像
4. **验证结果**: `runs/defect_detection*/val/` - 包含验证图像和图表（如果生成）

使用 `show_results.py` 脚本可以方便地查看测试结果。

