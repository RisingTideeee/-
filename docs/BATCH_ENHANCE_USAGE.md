# 批量图像增强使用说明

## 功能概述

批量图像增强功能允许您一次性处理整个目录中的图像，非常适合：
- 为训练数据准备增强后的图像
- 批量处理测试数据
- 生成多种增强方法的对比结果

## 新增功能

### 1. `enhance_batch()` - 批量增强图像列表
对内存中的图像列表进行批量增强。

```python
from image_enhancement import ImageEnhancer
import cv2

enhancer = ImageEnhancer()

# 读取多张图像
images = []
for img_path in image_paths:
    img = cv2.imread(img_path)
    images.append(img)

# 批量增强
enhanced_images = enhancer.enhance_batch(images, method='adaptive')
```

### 2. `enhance_from_directory()` - 从目录批量处理
直接从目录读取图像，增强后保存到输出目录。

```python
from image_enhancement import ImageEnhancer

enhancer = ImageEnhancer()

# 批量处理目录中的图像
stats = enhancer.enhance_from_directory(
    input_dir='dataset/train/images',
    output_dir='dataset/train/images_enhanced',
    method='adaptive',
    preserve_structure=True  # 保持目录结构
)

print(f"成功处理: {stats['success']} 张")
print(f"失败: {stats['failed']} 张")
```

### 3. `enhance_multiple_methods()` - 单张图像多方法增强
对单张图像应用多种增强方法，便于对比。

```python
enhancer = ImageEnhancement()

image = cv2.imread('test.jpg')

# 应用所有增强方法
results = enhancer.enhance_multiple_methods(image)

# results 包含:
# - 'hist_eq': 直方图均衡化结果
# - 'clahe': CLAHE结果
# - 'contrast_stretch': 对比度拉伸结果
# - 'gamma': 伽马校正结果
# - 'adaptive': 自适应增强结果
```

### 4. `enhance_batch_multiple_methods()` - 批量多方法增强
对批量图像应用多种增强方法。

```python
enhancer = ImageEnhancement()

images = [img1, img2, img3, ...]

# 对每张图像应用所有方法
results = enhancer.enhance_batch_multiple_methods(images)

# results['clahe'] 包含所有图像的CLAHE增强结果
# results['adaptive'] 包含所有图像的自适应增强结果
# ...
```

## 命令行使用

### 基本用法

```bash
# 使用自适应增强方法批量处理
python batch_enhance.py --input dataset/train/images --output dataset/train/images_enhanced

# 使用CLAHE方法
python batch_enhance.py --input dataset/train/images --output dataset/train/images_enhanced --method clahe

# 使用伽马校正
python batch_enhance.py --input dataset/train/images --output dataset/train/images_enhanced --method gamma --gamma 1.5

# 不保持目录结构（所有图像放在输出目录根目录）
python batch_enhance.py --input dataset/train/images --output dataset/train/images_enhanced --no-preserve_structure

# 处理并评估图像质量
python batch_enhance.py --input dataset/train/images --output dataset/train/images_enhanced --evaluate
```

### 参数说明

- `--input`: 输入目录路径（必需）
- `--output`: 输出目录路径（必需）
- `--method`: 增强方法
  - `hist_eq`: 直方图均衡化
  - `clahe`: CLAHE（推荐）
  - `contrast_stretch`: 对比度拉伸
  - `gamma`: 伽马校正
  - `adaptive`: 自适应增强（默认，推荐）
- `--preserve_structure`: 保持目录结构（默认：True）
- `--evaluate`: 评估增强后的图像质量
- `--clip_limit`: CLAHE的对比度限制（默认：2.0）
- `--gamma`: 伽马值（默认：1.5）

## 使用场景

### 场景1：为训练数据准备增强图像

```bash
# 增强训练集
python batch_enhance.py \
    --input dataset/train/images \
    --output dataset/train/images_enhanced \
    --method adaptive \
    --preserve_structure

# 增强验证集
python batch_enhance.py \
    --input dataset/val/images \
    --output dataset/val/images_enhanced \
    --method adaptive \
    --preserve_structure
```

### 场景2：批量处理测试数据

```bash
python batch_enhance.py \
    --input dataset/test/images \
    --output output/enhanced_test \
    --method adaptive \
    --evaluate
```

### 场景3：生成多种增强方法对比

```python
from image_enhancement import ImageEnhancer
import cv2
from pathlib import Path

enhancer = ImageEnhancer()
input_dir = Path('dataset/train/images')
output_base = Path('output/comparison')

# 对每张图像应用所有方法
for img_path in input_dir.glob('*.jpg'):
    image = cv2.imread(str(img_path))
    
    # 应用所有方法
    results = enhancer.enhance_multiple_methods(image)
    
    # 保存结果
    for method, enhanced in results.items():
        output_path = output_base / method / img_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), enhanced)
```

## 性能优化建议

1. **批量处理时使用自适应增强**：`adaptive` 方法通常效果最好
2. **保持目录结构**：便于后续处理和组织
3. **处理大量图像时**：考虑分批处理，避免内存溢出
4. **评估质量**：使用 `--evaluate` 参数检查增强效果

## 注意事项

1. 确保输入目录存在且包含图像文件
2. 支持的图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
3. 输出目录会自动创建
4. 如果图像读取失败，会在统计信息中显示
5. 处理大量图像时可能需要较长时间，请耐心等待

## 示例

```bash
# 示例1：增强训练数据
python batch_enhance.py \
    --input "D:/Users/CY/Desktop/数字图像处理基础/dataset/train/images" \
    --output "D:/Users/CY/Desktop/数字图像处理基础/dataset/train/images_enhanced" \
    --method adaptive

# 示例2：增强测试数据并评估
python batch_enhance.py \
    --input "D:/Users/CY/Desktop/数字图像处理基础/dataset/test/images" \
    --output "output/enhanced_test" \
    --method clahe \
    --clip_limit 2.5 \
    --evaluate
```

