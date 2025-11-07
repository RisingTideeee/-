# 汽车漆面缺陷检测系统

本项目实现了基于图像处理和深度学习的汽车漆面缺陷检测系统，支持传统图像处理方法和YOLOv11深度学习模型两种模式。

## 项目结构

```
AImodel/
├── image_enhancement.py      # 图像增强模块
├── defect_segmentation.py    # 传统缺陷分割模块
├── yolo_segmenter.py         # YOLOv11缺陷检测模块（新增）
├── feature_extraction.py     # 特征提取模块
├── classification.py          # 分类模块
├── evaluation.py              # 评价指标模块
├── main.py                    # 主程序
├── train_yolo.py              # YOLO模型训练脚本（新增）
├── prepare_yolo_dataset.py    # YOLO数据集准备脚本（新增）
└── utils.py                   # 工具函数
```

## 功能模块

### 1. 图像增强 (30%)
- 直方图均衡化
- CLAHE (对比度受限自适应直方图均衡化)
- 对比度拉伸
- 图像质量评价 (BRISQUE/NIQE)

### 2. 缺陷分割 (40%)
**传统方法：**
- Bradley-Roth 自适应阈值
- 形态学操作（开运算、闭运算、重建）
- 连通域分析

**YOLOv11方法（新增）：**
- 基于深度学习的缺陷检测
- 支持目标检测和实例分割
- 自动分类功能
- IoU 计算

### 3. 特征提取与分类 (30%)
**传统方法：**
- Gabor 滤波器特征
- LBP (局部二值模式) 特征
- SVM 分类器

**YOLOv11方法：**
- 端到端检测和分类
- 无需额外特征提取

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 使用传统方法

```bash
python main.py --input <图像路径> --mode all
```

### 使用YOLOv11模型

```bash
# 使用预训练模型
python main.py --input <图像路径> --mode all --use_yolo

# 使用自定义训练模型
python main.py --input <图像路径> --mode all --use_yolo --yolo_model <模型路径>

# 指定模型大小和任务类型
python main.py --input <图像路径> --mode all --use_yolo --yolo_size s --yolo_task segment
```

### 训练YOLOv11模型

```bash
python train_yolo.py --data <数据集yaml文件> --epochs 100 --batch 16 --device cuda
```

## 评价指标

- **图像增强**: BRISQUE/NIQE 分数
- **缺陷分割**: IoU (要求 > 50%)
- **分类**: 准确率、精确率、召回率、F1-Score

## YOLOv11优势

1. **更高的检测精度**：深度学习模型能够学习更复杂的特征
2. **端到端训练**：检测和分类一体化
3. **实时性能**：YOLO系列模型具有优秀的推理速度
4. **易于部署**：支持多种格式导出（ONNX、TensorRT等）
