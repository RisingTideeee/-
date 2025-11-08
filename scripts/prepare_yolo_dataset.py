"""
YOLO数据集准备脚本
将标注数据转换为YOLO格式
"""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np


def convert_to_yolo_format(image_path: str, annotation_path: str,
                           output_dir: str, class_mapping: dict):
    """
    将标注转换为YOLO格式
    
    Args:
        image_path: 图像路径
        annotation_path: 标注文件路径（可以是JSON、XML等格式）
        output_dir: 输出目录
        class_mapping: 类别映射字典
    """
    # 这里需要根据实际标注格式实现转换
    # 示例：假设是COCO格式的JSON
    pass


def create_yolo_yaml(output_path: str, dataset_path: str,
                    class_names: list):
    """
    创建YOLO数据集配置文件
    
    Args:
        output_path: 输出yaml文件路径
        dataset_path: 数据集根目录
        class_names: 类别名称列表
    """
    content = f"""# YOLO数据集配置文件
path: {dataset_path}
train: images/train
val: images/val
test: images/test  # 可选

# 类别名称
names:
"""
    for i, name in enumerate(class_names):
        content += f"  {i}: {name}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"YOLO配置文件已创建: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='准备YOLO格式数据集')
    parser.add_argument('--images', type=str, required=True,
                       help='图像目录')
    parser.add_argument('--annotations', type=str, required=True,
                       help='标注文件目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--classes', type=str, required=True,
                       help='类别名称，用逗号分隔，如：划痕,漆点,凹痕')
    parser.add_argument('--format', type=str, default='coco',
                       choices=['coco', 'voc', 'yolo'],
                       help='输入标注格式')
    
    args = parser.parse_args()
    
    # 解析类别名称
    class_names = [name.strip() for name in args.classes.split(',')]
    
    # 创建输出目录结构
    output_dir = Path(args.output)
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 创建YOLO配置文件
    yaml_path = output_dir / 'dataset.yaml'
    create_yolo_yaml(str(yaml_path), str(output_dir.absolute()), class_names)
    
    print("\n数据集准备说明:")
    print("1. 将图像放在 images/train 和 images/val 目录")
    print("2. 将YOLO格式的标注文件（.txt）放在 labels/train 和 labels/val 目录")
    print("3. 标注文件格式：每行一个目标，格式为: class_id center_x center_y width height")
    print("   坐标和尺寸都是归一化的（0-1之间）")
    print(f"4. 使用配置文件: {yaml_path}")


if __name__ == '__main__':
    main()

