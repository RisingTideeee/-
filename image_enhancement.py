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
    
    def gaussian_filter(self, image: np.ndarray, kernel_size: int = 5, 
                       sigma: float = 1.0) -> np.ndarray:
        """
        高斯滤波：用于降噪，在图像增强之前应用可以减少噪点
        
        Args:
            image: 输入图像
            kernel_size: 高斯核大小（必须是奇数，如3, 5, 7等）
            sigma: 高斯核标准差，控制平滑程度（越大越平滑）
            
        Returns:
            滤波后的图像
        """
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 应用高斯滤波
        filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return filtered
    
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
                # 防止除以0或接近0的数
                diff = p_high - p_low
                if diff < 1.0:  # 如果差值太小，直接返回原图
                    enhanced[:, :, i] = channel
                else:
                    enhanced[:, :, i] = np.clip(
                        (channel - p_low) * 255.0 / diff, 0, 255
                    ).astype(np.uint8)
        else:
            p_low, p_high = np.percentile(image, [lower_percent, upper_percent])
            # 防止除以0或接近0的数
            diff = p_high - p_low
            if diff < 1.0:  # 如果差值太小，直接返回原图
                enhanced = image.copy()
            else:
                enhanced = np.clip(
                    (image - p_low) * 255.0 / diff, 0, 255
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
    
    def adaptive_enhancement(self, image: np.ndarray, 
                            apply_gaussian_filter: bool = True,
                            gaussian_kernel_size: int = 6,
                            gaussian_sigma: float = 1.0) -> np.ndarray:
        """
        自适应增强：结合多种方法，使用更温和的参数避免过度处理
        确保不会让图像变暗
        
        Args:
            image: 输入图像
            apply_gaussian_filter: 是否在增强后应用高斯滤波（默认True，用于降噪）
            gaussian_kernel_size: 高斯核大小（默认5）
            gaussian_sigma: 高斯核标准差（默认1.0）
            
        Returns:
            增强后的图像
        """
        # 转换为灰度图用于分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 分析原图特征
        original_brightness = np.mean(gray)
        original_contrast = np.std(gray)
        
        # 使用更温和的CLAHE参数（降低clip_limit，避免过度增强）
        clahe_result = self.clahe(image, clip_limit=1.5, tile_grid_size=(8, 8))
        
        # 分析CLAHE结果
        if len(clahe_result.shape) == 3:
            clahe_gray = cv2.cvtColor(clahe_result, cv2.COLOR_BGR2GRAY)
        else:
            clahe_gray = clahe_result
        
        clahe_brightness = np.mean(clahe_gray)
        clahe_contrast = np.std(clahe_gray)
        
        # 如果CLAHE导致亮度显著降低（降低超过10%），需要补偿
        if clahe_brightness < original_brightness * 0.9:
            # 亮度补偿：使用线性变换提升亮度
            if len(clahe_result.shape) == 3:
                # 彩色图像：在LAB空间调整L通道
                lab = cv2.cvtColor(clahe_result, cv2.COLOR_BGR2LAB)
                # 提升亮度通道
                brightness_boost = (original_brightness - clahe_brightness) * 0.5  # 只补偿50%，避免过度
                lab[:, :, 0] = np.clip(lab[:, :, 0] + brightness_boost, 0, 255).astype(np.uint8)
                clahe_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                brightness_boost = (original_brightness - clahe_brightness) * 0.5
                clahe_result = np.clip(clahe_result.astype(np.float32) + brightness_boost, 0, 255).astype(np.uint8)
        
        # 检查是否需要对比度拉伸
        # 只有当对比度仍然较低时才进行拉伸
        if clahe_contrast < 40 and original_contrast < 30:
            # 使用非常温和的对比度拉伸（10%和90%）
            enhanced = self.contrast_stretching(clahe_result, lower_percent=10.0, upper_percent=90.0)
            
            # 再次检查亮度，确保没有变暗
            if len(enhanced.shape) == 3:
                enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            else:
                enhanced_gray = enhanced
            
            enhanced_brightness = np.mean(enhanced_gray)
            # 如果拉伸后亮度降低，使用CLAHE结果
            if enhanced_brightness < original_brightness * 0.95:
                enhanced = clahe_result
        else:
            # 对比度已经足够，直接使用CLAHE结果
            enhanced = clahe_result
        
        # 在增强之后应用高斯滤波降噪（平滑增强过程中产生的噪点）
        if apply_gaussian_filter:
            enhanced = self.gaussian_filter(enhanced, gaussian_kernel_size, gaussian_sigma)
        
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
                enhanced = self.adaptive_enhancement(image, **kwargs)
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
                    enhanced = self.adaptive_enhancement(image, **kwargs)
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

