# 图像增强训练使用指南

## 概述

现在训练脚本已集成自定义图像增强算法！你可以在训练YOLO模型时自动应用图像增强，提升模型性能。

## 功能特点

- ✅ **自动增强**：训练前自动对训练集和验证集图像进行增强
- ✅ **多种方法**：支持5种增强方法（直方图均衡化、CLAHE、对比度拉伸、伽马校正、自适应增强）
- ✅ **临时处理**：使用临时目录处理增强图像，训练完成后自动清理
- ✅ **无缝集成**：只需添加 `--enhance` 参数即可启用

## 使用方法

### 1. 基本训练（使用图像增强）

```bash
# 使用默认的自适应增强方法
python train_yolo.py --data dataset/dataset.yaml --enhance

# 指定增强方法
python train_yolo.py --data dataset/dataset.yaml --enhance --enhance-method clahe

# 使用CLAHE并自定义参数
python train_yolo.py --data dataset/dataset.yaml --enhance --enhance-method clahe --clip-limit 3.0
```

### 2. K折交叉验证（使用图像增强）

```bash
# 5折交叉验证 + 自适应增强
python train_kfold_cv.py --data dataset/dataset.yaml --enhance --k 5

# 使用CLAHE增强
python train_kfold_cv.py --data dataset/dataset.yaml --enhance --enhance-method clahe --k 5
```

### 3. 支持的增强方法

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `adaptive` | 自适应增强（CLAHE + 对比度拉伸） | **推荐**，适用于大多数场景 |
| `clahe` | 对比度受限自适应直方图均衡化 | 图像对比度低、光照不均 |
| `hist_eq` | 全局直方图均衡化 | 图像整体偏暗或偏亮 |
| `contrast_stretch` | 对比度拉伸 | 图像对比度不足 |
| `gamma` | 伽马校正 | 需要调整图像亮度 |

### 4. 完整训练示例

```bash
# 使用自适应增强 + 余弦学习率调度
python train_yolo.py \
    --data dataset/dataset.yaml \
    --enhance \
    --enhance-method adaptive \
    --epochs 100 \
    --batch 16 \
    --device cuda \
    --cos_lr \
    --patience 15
```

## 参数说明

### 增强相关参数

- `--enhance`: 启用图像增强（必需）
- `--enhance-method`: 选择增强方法
  - 可选值: `hist_eq`, `clahe`, `contrast_stretch`, `gamma`, `adaptive`
  - 默认值: `adaptive`
- `--clip-limit`: CLAHE的对比度限制（仅当 `--enhance-method=clahe` 时使用）
  - 默认值: `2.0`
  - 建议范围: `1.0-4.0`
- `--gamma`: 伽马值（仅当 `--enhance-method=gamma` 时使用）
  - 默认值: `1.5`
  - 建议范围: `0.5-2.5`

## 工作原理

1. **预处理阶段**：
   - 读取原始数据集配置
   - 创建临时增强数据集目录
   - 对训练集和验证集图像应用增强算法
   - 复制标签文件到临时目录
   - 创建新的数据集配置文件

2. **训练阶段**：
   - 使用增强后的数据集进行训练
   - YOLO会在此基础上应用自己的数据增强（Mosaic、HSV等）

3. **清理阶段**：
   - 训练完成后自动删除临时增强数据集
   - 释放磁盘空间

## 注意事项

1. **磁盘空间**：增强过程会创建临时数据集，确保有足够空间（约等于原始数据集大小）

2. **处理时间**：增强过程需要一些时间，特别是对于大型数据集
   - 720张训练图像 + 70张验证图像 ≈ 2-5分钟（取决于CPU性能）

3. **内存使用**：增强过程逐张处理图像，内存占用较小

4. **增强效果**：
   - 增强后的图像会保存在临时目录
   - 训练完成后会自动清理
   - 如需保留增强图像，可以修改代码或使用 `scripts/batch_enhance.py` 预先处理

## 对比实验建议

建议进行对比实验，评估增强效果：

```bash
# 实验1：不使用增强
python train_yolo.py --data dataset/dataset.yaml --name baseline

# 实验2：使用自适应增强
python train_yolo.py --data dataset/dataset.yaml --enhance --name enhanced_adaptive

# 实验3：使用CLAHE增强
python train_yolo.py --data dataset/dataset.yaml --enhance --enhance-method clahe --name enhanced_clahe
```

然后使用 `show_results.py` 比较不同实验的结果。

## 故障排除

### 问题1：找不到 image_enhancement 模块

**解决方案**：确保 `image_enhancement.py` 文件在项目根目录

### 问题2：临时目录清理失败

**解决方案**：手动删除临时目录（路径会在训练日志中显示）

### 问题3：增强效果不理想

**解决方案**：
- 尝试不同的增强方法
- 调整增强参数（如 `--clip-limit`, `--gamma`）
- 检查原始图像质量

## 技术细节

- 增强算法在训练前一次性处理所有图像
- 使用临时目录避免修改原始数据集
- 自动处理相对路径和绝对路径
- 保持数据集目录结构（images/train, images/val, labels/train, labels/val）

