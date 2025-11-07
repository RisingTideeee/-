"""
图像增强模块
实现直方图均衡化、CLAHE、对比度拉伸等方法
支持单张和批量图像处理
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List, Union
from pathlib import Path
import os
from tqdm import tqdm


class ImageEnhancer:
    """图像增强类"""
    
    def __init__(self):
        self.enhanced_image = None
        self.quality_score = None
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        全局直方图均衡化
        
        Args:
            image: 输入灰度图像
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            # 如果是彩色图像，转换为YUV空间，只对Y通道进行均衡化
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            enhanced = cv2.equalizeHist(image)
        return enhanced
    
    def clahe(self, image: np.ndarray, clip_limit: float = 2.0, 
              tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        对比度受限自适应直方图均衡化 (CLAHE)
        
        Args:
            image: 输入图像
            clip_limit: 对比度限制阈值
            tile_grid_size: 网格大小
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            # 彩色图像：在LAB空间对L通道应用CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(image)
        return enhanced
    
    def contrast_stretching(self, image: np.ndarray, 
                           lower_percent: float = 2.0,
                           upper_percent: float = 98.0) -> np.ndarray:
        """
        对比度拉伸（线性拉伸）
        
        Args:
            image: 输入图像
            lower_percent: 下百分位数
            upper_percent: 上百分位数
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            # 对每个通道分别处理
            enhanced = image.copy()
            for i in range(3):
                channel = image[:, :, i]
                p_low, p_high = np.percentile(channel, [lower_percent, upper_percent])
                enhanced[:, :, i] = np.clip(
                    (channel - p_low) * 255.0 / (p_high - p_low), 0, 255
                ).astype(np.uint8)
        else:
            p_low, p_high = np.percentile(image, [lower_percent, upper_percent])
            enhanced = np.clip(
                (image - p_low) * 255.0 / (p_high - p_low), 0, 255
            ).astype(np.uint8)
        return enhanced
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """
        伽马校正
        
        Args:
            image: 输入图像
            gamma: 伽马值
            
        Returns:
            增强后的图像
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(image, table)
        return enhanced
    
    def enhance(self, image: np.ndarray, method: str = 'clahe', 
                **kwargs) -> np.ndarray:
        """
        图像增强主函数
        
        Args:
            image: 输入图像
            method: 增强方法 ('hist_eq', 'clahe', 'contrast_stretch', 'gamma')
            **kwargs: 方法特定参数
            
        Returns:
            增强后的图像
        """
        if method == 'hist_eq':
            enhanced = self.histogram_equalization(image)
        elif method == 'clahe':
            clip_limit = kwargs.get('clip_limit', 2.0)
            tile_size = kwargs.get('tile_grid_size', (8, 8))
            enhanced = self.clahe(image, clip_limit, tile_size)
        elif method == 'contrast_stretch':
            lower = kwargs.get('lower_percent', 2.0)
            upper = kwargs.get('upper_percent', 98.0)
            enhanced = self.contrast_stretching(image, lower, upper)
        elif method == 'gamma':
            gamma = kwargs.get('gamma', 1.5)
            enhanced = self.gamma_correction(image, gamma)
        else:
            raise ValueError(f"未知的增强方法: {method}")
        
        self.enhanced_image = enhanced
        return enhanced
    
    def adaptive_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        自适应增强：结合多种方法
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        # 先应用CLAHE
        clahe_result = self.clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))
        # 再应用对比度拉伸
        enhanced = self.contrast_stretching(clahe_result, lower_percent=1.0, upper_percent=99.0)
        self.enhanced_image = enhanced
        return enhanced
    
    def enhance_batch(self, images: List[np.ndarray], 
                     method: str = 'adaptive',
                     **kwargs) -> List[np.ndarray]:
        """
        批量图像增强
        
        Args:
            images: 图像列表
            method: 增强方法 ('hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive')
            **kwargs: 方法特定参数
            
        Returns:
            增强后的图像列表
        """
        enhanced_images = []
        for image in tqdm(images, desc="批量增强中"):
            if method == 'adaptive':
                enhanced = self.adaptive_enhancement(image)
            else:
                enhanced = self.enhance(image, method=method, **kwargs)
            enhanced_images.append(enhanced)
        return enhanced_images
    
    def enhance_from_directory(self, input_dir: Union[str, Path],
                               output_dir: Union[str, Path],
                               method: str = 'adaptive',
                               preserve_structure: bool = True,
                               image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
                               **kwargs) -> dict:
        """
        从目录批量读取图像并增强
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            method: 增强方法
            preserve_structure: 是否保持目录结构
            image_extensions: 支持的图像扩展名
            **kwargs: 方法特定参数
            
        Returns:
            处理结果统计字典
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.rglob(f'*{ext}'))
            image_files.extend(input_dir.rglob(f'*{ext.upper()}'))
        
        if len(image_files) == 0:
            raise ValueError(f"在 {input_dir} 中未找到图像文件")
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 统计信息
        stats = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        # 批量处理
        for img_path in tqdm(image_files, desc="处理图像"):
            try:
                # 读取图像
                image = cv2.imread(str(img_path))
                if image is None:
                    stats['failed'] += 1
                    stats['failed_files'].append(str(img_path))
                    continue
                
                # 增强图像
                if method == 'adaptive':
                    enhanced = self.adaptive_enhancement(image)
                else:
                    enhanced = self.enhance(image, method=method, **kwargs)
                
                # 确定输出路径
                if preserve_structure:
                    # 保持相对路径结构
                    relative_path = img_path.relative_to(input_dir)
                    output_path = output_dir / relative_path
                else:
                    # 扁平化输出
                    output_path = output_dir / img_path.name
                
                # 创建输出目录
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存增强后的图像
                cv2.imwrite(str(output_path), enhanced)
                stats['success'] += 1
                
            except Exception as e:
                stats['failed'] += 1
                stats['failed_files'].append(str(img_path))
                print(f"处理 {img_path} 时出错: {e}")
        
        return stats
    
    def enhance_multiple_methods(self, image: np.ndarray,
                                 methods: Optional[List[str]] = None) -> dict:
        """
        对单张图像应用多种增强方法
        
        Args:
            image: 输入图像
            methods: 要应用的方法列表，如果为None则使用所有方法
            
        Returns:
            字典，键为方法名，值为增强后的图像
        """
        if methods is None:
            methods = ['hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive']
        
        results = {}
        for method in methods:
            try:
                if method == 'adaptive':
                    enhanced = self.adaptive_enhancement(image)
                else:
                    enhanced = self.enhance(image, method=method)
                results[method] = enhanced
            except Exception as e:
                print(f"应用方法 {method} 时出错: {e}")
        
        return results
    
    def enhance_batch_multiple_methods(self, images: List[np.ndarray],
                                       methods: Optional[List[str]] = None) -> dict:
        """
        对批量图像应用多种增强方法
        
        Args:
            images: 图像列表
            methods: 要应用的方法列表
            
        Returns:
            字典，键为方法名，值为增强后的图像列表
        """
        if methods is None:
            methods = ['hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive']
        
        results = {method: [] for method in methods}
        
        for image in tqdm(images, desc="批量多方法增强"):
            enhanced_dict = self.enhance_multiple_methods(image, methods)
            for method, enhanced in enhanced_dict.items():
                results[method].append(enhanced)
        
        return results

