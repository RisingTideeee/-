"""
工具函数模块
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    加载图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        图像数组 (BGR格式)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return img


def save_image(image: np.ndarray, output_path: str):
    """
    保存图像
    
    Args:
        image: 图像数组
        output_path: 输出路径
    """
    cv2.imwrite(output_path, image)
    print(f"图像已保存至: {output_path}")


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    转换为灰度图
    
    Args:
        image: BGR图像
        
    Returns:
        灰度图像
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def create_output_dir(output_dir: str = "output"):
    """
    创建输出目录
    
    Args:
        output_dir: 输出目录路径
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def visualize_results(images: dict, titles: dict, save_path: Optional[str] = None):
    """
    可视化结果
    
    Args:
        images: 图像字典
        titles: 标题字典
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    
    for idx, (key, img) in enumerate(images.items()):
        if len(img.shape) == 2:
            axes[idx].imshow(img, cmap='gray')
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img_rgb)
        axes[idx].set_title(titles.get(key, key))
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    plt.show()

