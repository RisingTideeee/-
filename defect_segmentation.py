"""
缺陷分割模块
实现Bradley-Roth自适应阈值、形态学操作、连通域分析
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional


class DefectSegmenter:
    """缺陷分割类"""
    
    def __init__(self):
        self.segmented_mask = None
        self.defect_regions = []
    
    def bradley_roth_threshold(self, image: np.ndarray, 
                               window_size: int = 15,
                               threshold: float = 0.15) -> np.ndarray:
        """
        Bradley-Roth 自适应阈值算法
        
        Args:
            image: 输入灰度图像
            window_size: 窗口大小
            threshold: 阈值比例 (0-1)
            
        Returns:
            二值化图像
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算积分图
        integral = cv2.integral(image)
        h, w = image.shape
        
        # 创建输出图像
        binary = np.zeros_like(image)
        
        # 计算窗口半径
        s = window_size // 2
        
        for y in range(h):
            for x in range(w):
                # 计算窗口边界
                x1 = max(0, x - s)
                y1 = max(0, y - s)
                x2 = min(w, x + s + 1)
                y2 = min(h, y + s + 1)
                
                # 计算窗口内像素总和
                count = (x2 - x1) * (y2 - y1)
                sum_val = (integral[y2, x2] - integral[y1, x2] - 
                          integral[y2, x1] + integral[y1, x1])
                
                # 计算平均值
                mean = sum_val / count
                
                # 自适应阈值
                if image[y, x] < mean * (1 - threshold):
                    binary[y, x] = 255
                else:
                    binary[y, x] = 0
        
        return binary
    
    def morphological_operations(self, binary: np.ndarray,
                               operation: str = 'open',
                               kernel_size: Tuple[int, int] = (3, 3),
                               iterations: int = 1) -> np.ndarray:
        """
        形态学操作
        
        Args:
            binary: 二值图像
            operation: 操作类型 ('open', 'close', 'erode', 'dilate')
            kernel_size: 核大小
            iterations: 迭代次数
            
        Returns:
            处理后的图像
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        
        if operation == 'open':
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'erode':
            result = cv2.erode(binary, kernel, iterations=iterations)
        elif operation == 'dilate':
            result = cv2.dilate(binary, kernel, iterations=iterations)
        else:
            raise ValueError(f"未知的形态学操作: {operation}")
        
        return result
    
    def fill_holes(self, binary: np.ndarray) -> np.ndarray:
        """
        填充孔洞
        
        Args:
            binary: 二值图像
            
        Returns:
            填充后的图像
        """
        # 创建掩码：找到背景像素
        h, w = binary.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # 从边缘开始填充背景
        cv2.floodFill(binary.copy(), mask, (0, 0), 255)
        
        # 反转掩码得到孔洞
        mask = mask[1:-1, 1:-1]
        filled = binary | (~mask.astype(np.uint8) * 255)
        
        return filled
    
    def morphological_reconstruction(self, binary: np.ndarray,
                                    kernel_size: Tuple[int, int] = (5, 5),
                                    iterations: int = 3) -> np.ndarray:
        """
        形态学重建（简化版）
        
        Args:
            binary: 二值图像
            kernel_size: 核大小
            iterations: 迭代次数
            
        Returns:
            重建后的图像
        """
        # 先进行开运算去噪
        opened = self.morphological_operations(binary, 'open', kernel_size, iterations)
        
        # 填充孔洞
        filled = self.fill_holes(opened)
        
        # 闭运算平滑边界
        closed = self.morphological_operations(filled, 'close', kernel_size, iterations)
        
        return closed
    
    def connected_components_analysis(self, binary: np.ndarray,
                                     min_area: int = 100,
                                     max_area: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
        """
        连通域分析
        
        Args:
            binary: 二值图像
            min_area: 最小区域面积
            max_area: 最大区域面积（可选）
            
        Returns:
            标记图像和区域信息列表
        """
        # 连通域标记
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # 筛选区域
        filtered_mask = np.zeros_like(binary)
        defect_regions = []
        
        for i in range(1, num_labels):  # 跳过背景（标签0）
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 面积筛选
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            
            # 创建该区域的掩码
            region_mask = (labels == i).astype(np.uint8) * 255
            filtered_mask = cv2.bitwise_or(filtered_mask, region_mask)
            
            # 保存区域信息
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            defect_regions.append({
                'label': i,
                'area': int(area),
                'bbox': (x, y, w, h),
                'centroid': (int(centroids[i, 0]), int(centroids[i, 1])),
                'mask': region_mask
            })
        
        self.defect_regions = defect_regions
        return filtered_mask, defect_regions
    
    def segment(self, image: np.ndarray,
               window_size: int = 15,
               threshold: float = 0.15,
               min_area: int = 100,
               use_morphology: bool = True) -> Tuple[np.ndarray, List[dict]]:
        """
        缺陷分割主函数
        
        Args:
            image: 输入图像（增强后的图像）
            window_size: Bradley-Roth窗口大小
            threshold: Bradley-Roth阈值
            min_area: 最小缺陷区域面积
            use_morphology: 是否使用形态学操作
            
        Returns:
            分割掩码和缺陷区域列表
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Bradley-Roth自适应阈值
        binary = self.bradley_roth_threshold(gray, window_size, threshold)
        
        # 2. 形态学操作去噪和填充
        if use_morphology:
            # 开运算去噪
            binary = self.morphological_operations(binary, 'open', (3, 3), 2)
            # 填充孔洞
            binary = self.fill_holes(binary)
            # 形态学重建
            binary = self.morphological_reconstruction(binary, (5, 5), 2)
        
        # 3. 连通域分析
        segmented_mask, defect_regions = self.connected_components_analysis(
            binary, min_area=min_area
        )
        
        self.segmented_mask = segmented_mask
        return segmented_mask, defect_regions
    
    def visualize_segmentation(self, original: np.ndarray, 
                               mask: np.ndarray,
                               defect_regions: List[dict]) -> np.ndarray:
        """
        可视化分割结果
        
        Args:
            original: 原始图像
            mask: 分割掩码
            defect_regions: 缺陷区域列表
            
        Returns:
            可视化图像
        """
        # 创建彩色掩码
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
        # 叠加到原图
        result = cv2.addWeighted(original, 0.7, colored_mask, 0.3, 0)
        
        # 绘制边界框
        for region in defect_regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 标注面积
            cv2.putText(result, f"Area: {region['area']}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result

