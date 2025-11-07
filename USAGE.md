# 使用说明

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用

#### 运行完整流程（图像增强 + 缺陷分割 + 分类）

```bash
python main.py --input <图像路径> --mode all
```

#### 仅运行图像增强

```bash
python main.py --input <图像路径> --mode enhance
```

#### 仅运行缺陷分割

```bash
python main.py --input <图像路径> --mode segment
```

#### 运行分割并计算IoU（需要提供真实掩码）

```bash
python main.py --input <图像路径> --mode segment --gt_mask <真实掩码路径>
```

#### 运行完整流程并分类（需要提供训练数据）

```bash
python main.py --input <图像路径> --mode all --train_data <训练数据目录>
```

### 3. 参数说明

- `--input`: 输入图像路径（必需）
- `--output`: 输出目录（默认：output）
- `--mode`: 运行模式
  - `all`: 运行所有模块
  - `enhance`: 仅图像增强
  - `segment`: 仅缺陷分割
  - `classify`: 仅分类
- `--gt_mask`: 真实分割掩码路径（用于计算IoU）
- `--train_data`: 训练数据目录（用于分类）

## 模块说明

### 任务一：图像增强

**功能**：
- 直方图均衡化
- CLAHE（对比度受限自适应直方图均衡化）
- 对比度拉伸
- 伽马校正
- 自适应增强（组合多种方法）

**评价指标**：
- BRISQUE分数（越低越好）
- NIQE分数（越低越好）

**使用示例**：

```python
from image_enhancement import ImageEnhancer
from evaluation import ImageQualityAssessment
import cv2

# 加载图像
image = cv2.imread("input.jpg")

# 创建增强器
enhancer = ImageEnhancer()
iqa = ImageQualityAssessment()

# 应用CLAHE增强
enhanced = enhancer.enhance(image, method='clahe', clip_limit=2.0)

# 计算质量分数
brisque = iqa.calculate_quality_score(enhanced, 'brisque')
print(f"BRISQUE分数: {brisque:.4f}")
```

### 任务二：缺陷分割

**功能**：
- Bradley-Roth自适应阈值
- 形态学操作（开运算、闭运算、重建）
- 孔洞填充
- 连通域分析

**评价指标**：
- IoU（交并比，要求 > 0.5）
- Dice系数
- 像素准确率

**使用示例**：

```python
from defect_segmentation import DefectSegmenter
from evaluation import SegmentationEvaluation
import cv2

# 加载增强后的图像
enhanced_image = cv2.imread("enhanced.jpg")

# 创建分割器
segmenter = DefectSegmenter()

# 执行分割
mask, regions = segmenter.segment(
    enhanced_image,
    window_size=15,
    threshold=0.15,
    min_area=100
)

# 可视化结果
visualization = segmenter.visualize_segmentation(
    original_image, mask, regions
)

# 计算IoU（如果有真实掩码）
gt_mask = cv2.imread("ground_truth.jpg", 0)
eval_seg = SegmentationEvaluation()
iou = eval_seg.calculate_iou(mask, gt_mask)
print(f"IoU: {iou:.4f}")
```

### 任务三：特征提取与分类

**功能**：
- Gabor滤波器特征
- LBP（局部二值模式）特征
- 几何特征
- 纹理特征
- SVM分类器
- 特征可视化（PCA降维）

**评价指标**：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1-Score

**使用示例**：

```python
from feature_extraction import FeatureExtractor
from classification import DefectClassifier
from evaluation import ClassificationEvaluation
import numpy as np

# 提取特征
extractor = FeatureExtractor()
features = extractor.extract_all_features(
    defect_roi, defect_mask,
    use_gabor=True,
    use_lbp=True,
    use_geometric=True,
    use_texture=True
)

# 训练分类器
X_train = np.array([...])  # 训练特征
y_train = np.array([...])  # 训练标签

classifier = DefectClassifier(kernel='rbf', C=1.0)
train_info = classifier.train(X_train, y_train)

# 预测
predictions = classifier.predict(X_test)

# 评价
eval_class = ClassificationEvaluation()
metrics = eval_class.calculate_metrics(y_true, predictions)
eval_class.print_metrics(metrics)

# 特征可视化
classifier.visualize_features(X_train, y_train, n_components=2)
```

## 输出文件说明

运行程序后，在输出目录中会生成以下文件：

- `enhanced_*.jpg`: 不同方法增强的图像
- `enhanced_best.jpg`: 最佳增强结果
- `enhancement_comparison.png`: 增强前后对比图
- `segmented_mask.jpg`: 分割掩码
- `segmentation_result.jpg`: 分割结果可视化
- `feature_visualization.png`: 特征分布可视化
- `classification_result.jpg`: 分类结果可视化

## 注意事项

1. **图像格式**：支持常见图像格式（jpg, png, bmp等）
2. **图像大小**：建议图像尺寸在合理范围内（如800x600到2000x1500）
3. **参数调整**：根据实际图像特点，可能需要调整以下参数：
   - Bradley-Roth的`window_size`和`threshold`
   - 形态学操作的`kernel_size`和`iterations`
   - 连通域分析的`min_area`
4. **训练数据**：分类功能需要提供训练数据，数据格式需要根据实际情况调整

## 示例脚本

运行示例脚本查看各模块的使用方法：

```bash
python example.py
```

## 常见问题

**Q: IoU计算需要什么格式的真实掩码？**
A: 真实掩码应该是二值图像（黑白图像），缺陷区域为白色（255），背景为黑色（0）。

**Q: 如何准备训练数据？**
A: 训练数据需要包含：
- 特征矩阵（n_samples × n_features）
- 标签向量（n_samples）
- 每个样本对应一个缺陷区域的特征和类别标签

**Q: 如何调整分割参数？**
A: 可以在`defect_segmentation.py`的`segment()`方法中调整：
- `window_size`: 增大可检测更大缺陷，减小可检测更小缺陷
- `threshold`: 增大可检测更多缺陷，减小可减少误检
- `min_area`: 过滤小面积噪声

