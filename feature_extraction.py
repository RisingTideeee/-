"""
特征提取模块
实现Gabor滤波器和LBP特征提取
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage import feature as skfeature


class FeatureExtractor:
    """特征提取类"""
    
    def __init__(self):
        self.features = None
    
    def gabor_filter(self, image: np.ndarray,
                    frequencies: List[float] = [0.1, 0.2, 0.3],
                    orientations: List[float] = [0, 45, 90, 135],
                    kernel_size: int = 21,
                    sigma: float = 5.0,
                    gamma: float = 0.5) -> np.ndarray:
        """
        Gabor滤波器特征提取
        
        Args:
            image: 输入灰度图像
            frequencies: 频率列表
            orientations: 方向列表（度）
            kernel_size: 核大小
            sigma: 标准差
            gamma: 空间纵横比
            
        Returns:
            特征向量
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        for freq in frequencies:
            for theta in orientations:
                # 转换为弧度
                theta_rad = np.deg2rad(theta)
                
                # 创建Gabor核
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size),
                    sigma,
                    theta_rad,
                    2 * np.pi * freq,
                    gamma,
                    0,
                    ktype=cv2.CV_32F
                )
                
                # 应用滤波器
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                
                # 提取统计特征
                features.append(np.mean(filtered))
                features.append(np.std(filtered))
                features.append(np.var(filtered))
        
        return np.array(features)
    
    def lbp(self, image: np.ndarray,
           radius: int = 3,
           n_points: int = 24,
           method: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        """
        LBP (Local Binary Pattern) 特征提取
        
        Args:
            image: 输入灰度图像
            radius: 半径
            n_points: 采样点数
            method: LBP方法 ('default', 'uniform', 'ror', 'var')
            
        Returns:
            LBP图像和直方图特征
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用scikit-image的LBP
        lbp_image = skfeature.local_binary_pattern(
            image, n_points, radius, method=method
        )
        
        # 计算直方图作为特征
        hist, _ = np.histogram(
            lbp_image.ravel(),
            bins=n_points + 2,
            range=(0, n_points + 2),
            density=True
        )
        
        return lbp_image, hist
    
    def extract_geometric_features(self, mask: np.ndarray) -> np.ndarray:
        """
        提取几何特征
        
        Args:
            mask: 缺陷区域掩码
            
        Returns:
            几何特征向量
        """
        # 计算轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return np.zeros(10)
        
        # 使用最大轮廓
        contour = max(contours, key=cv2.contourArea)
        
        features = []
        
        # 1. 面积
        area = cv2.contourArea(contour)
        features.append(area)
        
        # 2. 周长
        perimeter = cv2.arcLength(contour, True)
        features.append(perimeter)
        
        # 3. 圆形度 (4π*面积/周长²)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        features.append(circularity)
        
        # 4. 边界框
        x, y, w, h = cv2.boundingRect(contour)
        features.extend([w, h, w/h if h > 0 else 0])
        
        # 5. 最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        extent = area / box_area if box_area > 0 else 0
        features.append(extent)
        
        # 6. 凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        features.append(solidity)
        
        # 7. 等效直径
        equi_diameter = np.sqrt(4 * area / np.pi)
        features.append(equi_diameter)
        
        return np.array(features)
    
    def extract_texture_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        提取纹理特征
        
        Args:
            image: 输入图像
            mask: 掩码（可选）
            
        Returns:
            纹理特征向量
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if mask is not None:
            # 只在掩码区域内提取特征
            image = cv2.bitwise_and(image, mask)
        
        features = []
        
        # 1. 灰度共生矩阵特征（简化版）
        # 计算局部方差
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        features.append(np.mean(local_var))
        features.append(np.std(local_var))
        
        # 2. 梯度特征
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        
        # 3. 统计特征
        features.append(np.mean(image))
        features.append(np.std(image))
        features.append(np.var(image))
        
        return np.array(features)
    
    def extract_all_features(self, image: np.ndarray, 
                            mask: Optional[np.ndarray] = None,
                            use_gabor: bool = True,
                            use_lbp: bool = True,
                            use_geometric: bool = True,
                            use_texture: bool = True) -> np.ndarray:
        """
        提取所有特征
        
        Args:
            image: 输入图像
            mask: 缺陷区域掩码（可选）
            use_gabor: 是否使用Gabor特征
            use_lbp: 是否使用LBP特征
            use_geometric: 是否使用几何特征
            use_texture: 是否使用纹理特征
            
        Returns:
            特征向量
        """
        features = []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Gabor特征
        if use_gabor:
            gabor_feat = self.gabor_filter(gray)
            features.extend(gabor_feat)
        
        # LBP特征
        if use_lbp:
            _, lbp_hist = self.lbp(gray)
            features.extend(lbp_hist)
        
        # 几何特征
        if use_geometric and mask is not None:
            geom_feat = self.extract_geometric_features(mask)
            features.extend(geom_feat)
        
        # 纹理特征
        if use_texture:
            texture_feat = self.extract_texture_features(gray, mask)
            features.extend(texture_feat)
        
        self.features = np.array(features)
        return self.features

