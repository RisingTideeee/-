"""
评价指标模块
实现BRISQUE、NIQE、IoU等评价指标
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage
from scipy.special import gamma


class ImageQualityAssessment:
    """图像质量评价类"""
    
    def __init__(self):
        pass
    
    def brisque(self, image: np.ndarray) -> float:
        """
        计算BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) 分数
        分数越低表示质量越好
        
        Args:
            image: 输入灰度图像
            
        Returns:
            BRISQUE分数
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 归一化到0-1
        img = image.astype(np.float64) / 255.0
        
        # 计算局部均值
        mu = cv2.GaussianBlur(img, (7, 7), 7/6)
        mu_sq = mu * mu
        
        # 计算局部方差
        sigma = cv2.GaussianBlur(img * img, (7, 7), 7/6)
        sigma = np.sqrt(np.abs(sigma - mu_sq))
        
        # 计算结构对比度
        structdis = (img - mu) / (sigma + 1e-10)
        
        # 计算特征
        features = []
        
        # 1. 均值
        features.append(np.mean(structdis))
        
        # 2. 方差
        features.append(np.var(structdis))
        
        # 3. 偏度
        features.append(self._skewness(structdis))
        
        # 4. 峰度
        features.append(self._kurtosis(structdis))
        
        # 5. 均值绝对偏差
        features.append(np.mean(np.abs(structdis - np.mean(structdis))))
        
        # 简化版BRISQUE：基于特征计算分数
        # 实际BRISQUE需要训练好的SVR模型，这里使用简化版本
        score = np.mean(features) * 10  # 简化的分数计算
        
        return float(score)
    
    def _skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data.flatten())
        skew = np.sum(((data - mean) / std) ** 3) / n
        return float(skew)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data.flatten())
        kurt = np.sum(((data - mean) / std) ** 4) / n - 3.0
        return float(kurt)
    
    def niqe(self, image: np.ndarray) -> float:
        """
        计算NIQE (Natural Image Quality Evaluator) 分数
        分数越低表示质量越好
        
        Args:
            image: 输入灰度图像
            
        Returns:
            NIQE分数
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 归一化
        img = image.astype(np.float64) / 255.0
        
        # 计算局部均值
        mu = cv2.GaussianBlur(img, (7, 7), 7/6)
        mu_sq = mu * mu
        
        # 计算局部方差
        sigma = cv2.GaussianBlur(img * img, (7, 7), 7/6)
        sigma = np.sqrt(np.abs(sigma - mu_sq))
        
        # 计算结构对比度
        structdis = (img - mu) / (sigma + 1e-10)
        
        # 计算特征（简化版）
        features = []
        features.append(np.mean(structdis))
        features.append(np.var(structdis))
        features.append(self._skewness(structdis))
        features.append(self._kurtosis(structdis))
        
        # 简化版NIQE分数
        score = np.std(features) * 5  # 简化的分数计算
        
        return float(score)
    
    def calculate_quality_score(self, image: np.ndarray, method: str = 'brisque') -> float:
        """
        计算图像质量分数
        
        Args:
            image: 输入图像
            method: 评价方法 ('brisque' 或 'niqe')
            
        Returns:
            质量分数
        """
        if method.lower() == 'brisque':
            return self.brisque(image)
        elif method.lower() == 'niqe':
            return self.niqe(image)
        else:
            raise ValueError(f"未知的评价方法: {method}")


class SegmentationEvaluation:
    """分割评价类"""
    
    def __init__(self):
        pass
    
    def calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        计算IoU (Intersection over Union)
        
        Args:
            pred_mask: 预测的分割掩码（二值图像）
            gt_mask: 真实的分割掩码（二值图像）
            
        Returns:
            IoU值 (0-1之间)
        """
        # 确保是二值图像
        if pred_mask.dtype != np.uint8:
            pred_mask = (pred_mask > 0).astype(np.uint8) * 255
        if gt_mask.dtype != np.uint8:
            gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        
        # 计算交集和并集
        intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
        union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = intersection / union
        return float(iou)
    
    def calculate_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        计算Dice系数
        
        Args:
            pred_mask: 预测的分割掩码
            gt_mask: 真实的分割掩码
            
        Returns:
            Dice系数 (0-1之间)
        """
        if pred_mask.dtype != np.uint8:
            pred_mask = (pred_mask > 0).astype(np.uint8) * 255
        if gt_mask.dtype != np.uint8:
            gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        
        intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
        pred_area = (pred_mask > 0).sum()
        gt_area = (gt_mask > 0).sum()
        
        if pred_area + gt_area == 0:
            return 1.0
        
        dice = 2.0 * intersection / (pred_area + gt_area)
        return float(dice)
    
    def calculate_pixel_accuracy(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        计算像素准确率
        
        Args:
            pred_mask: 预测的分割掩码
            gt_mask: 真实的分割掩码
            
        Returns:
            像素准确率 (0-1之间)
        """
        if pred_mask.dtype != np.uint8:
            pred_mask = (pred_mask > 0).astype(np.uint8) * 255
        if gt_mask.dtype != np.uint8:
            gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        
        correct = (pred_mask == gt_mask).sum()
        total = pred_mask.size
        
        return float(correct / total)


class ClassificationEvaluation:
    """分类评价类"""
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        计算分类评价指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            包含准确率、精确率、召回率、F1-Score的字典
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    def print_metrics(self, metrics: dict):
        """
        打印评价指标
        
        Args:
            metrics: 评价指标字典
        """
        print("\n=== 分类性能评价 ===")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")


class DistortionAnalyzer:
    """图像失真类型分析类"""
    
    def __init__(self):
        pass
    
    def analyze_brightness(self, image: np.ndarray) -> Tuple[float, str]:
        """
        分析图像亮度
        
        Args:
            image: 输入图像
            
        Returns:
            (亮度值, 亮度状态) 亮度值范围0-255，状态: 'low', 'normal', 'high'
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        brightness = np.mean(gray)
        
        if brightness < 80:
            status = 'low'
        elif brightness > 175:
            status = 'high'
        else:
            status = 'normal'
        
        return float(brightness), status
    
    def analyze_contrast(self, image: np.ndarray) -> Tuple[float, str]:
        """
        分析图像对比度
        
        Args:
            image: 输入图像
            
        Returns:
            (对比度值, 对比度状态) 对比度值范围0-255，状态: 'low', 'normal', 'high'
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用标准差作为对比度指标
        contrast = np.std(gray)
        
        if contrast < 30:
            status = 'low'
        elif contrast > 60:
            status = 'high'
        else:
            status = 'normal'
        
        return float(contrast), status
    
    def analyze_noise(self, image: np.ndarray) -> Tuple[float, str]:
        """
        分析图像噪声水平
        
        Args:
            image: 输入图像
            
        Returns:
            (噪声水平, 噪声状态) 噪声水平值，状态: 'low', 'medium', 'high'
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用拉普拉斯算子的方差来估计噪声
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var()
        
        if noise_level < 100:
            status = 'low'
        elif noise_level > 500:
            status = 'high'
        else:
            status = 'medium'
        
        return float(noise_level), status
    
    def analyze_histogram_distribution(self, image: np.ndarray) -> str:
        """
        分析直方图分布特征
        
        Args:
            image: 输入图像
            
        Returns:
            分布类型: 'uniform', 'skewed_left', 'skewed_right', 'normal'
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # 计算直方图的偏度
        mean = np.mean(gray)
        std = np.std(gray)
        if std == 0:
            return 'uniform'
        
        # 计算偏度（简化版）
        skew = np.sum(((gray - mean) / std) ** 3) / gray.size
        
        # 检查直方图是否集中在某一边
        left_sum = np.sum(hist[:85])  # 低灰度值
        middle_sum = np.sum(hist[85:170])  # 中灰度值
        right_sum = np.sum(hist[170:])  # 高灰度值
        
        total = left_sum + middle_sum + right_sum
        if total == 0:
            return 'uniform'
        
        left_ratio = left_sum / total
        right_ratio = right_sum / total
        
        if left_ratio > 0.5:
            return 'skewed_left'  # 图像偏暗
        elif right_ratio > 0.5:
            return 'skewed_right'  # 图像偏亮
        elif abs(skew) < 0.5:
            return 'normal'
        else:
            return 'skewed_left' if skew < 0 else 'skewed_right'
    
    def analyze_distortion(self, image: np.ndarray) -> dict:
        """
        综合分析图像失真类型和程度
        
        Args:
            image: 输入图像
            
        Returns:
            包含失真类型、程度和建议增强方法的字典
        """
        brightness, brightness_status = self.analyze_brightness(image)
        contrast, contrast_status = self.analyze_contrast(image)
        noise_level, noise_status = self.analyze_noise(image)
        hist_dist = self.analyze_histogram_distribution(image)
        
        # 综合评估失真程度
        distortion_score = 0
        distortion_types = []
        
        if brightness_status != 'normal':
            distortion_score += 1
            distortion_types.append(f'亮度异常({brightness_status})')
        
        if contrast_status == 'low':
            distortion_score += 2
            distortion_types.append('对比度不足')
        elif contrast_status == 'high':
            distortion_score += 0.5
            distortion_types.append('对比度过高')
        
        if noise_status == 'high':
            distortion_score += 1.5
            distortion_types.append('噪声较高')
        elif noise_status == 'medium':
            distortion_score += 0.5
        
        if hist_dist in ['skewed_left', 'skewed_right']:
            distortion_score += 1
            distortion_types.append(f'直方图分布不均({hist_dist})')
        
        # 根据失真类型和程度推荐增强方法
        recommended_method = self._recommend_enhancement_method(
            brightness_status, contrast_status, noise_status, hist_dist, distortion_score
        )
        
        return {
            'brightness': brightness,
            'brightness_status': brightness_status,
            'contrast': contrast,
            'contrast_status': contrast_status,
            'noise_level': noise_level,
            'noise_status': noise_status,
            'histogram_distribution': hist_dist,
            'distortion_score': distortion_score,
            'distortion_types': distortion_types,
            'recommended_method': recommended_method,
            'distortion_severity': 'low' if distortion_score < 2 else ('medium' if distortion_score < 4 else 'high')
        }
    
    def _recommend_enhancement_method(self, brightness_status: str, contrast_status: str, 
                                     noise_status: str, hist_dist: str, 
                                     distortion_score: float) -> str:
        """
        根据失真类型推荐最佳增强方法
        
        Args:
            brightness_status: 亮度状态
            contrast_status: 对比度状态
            noise_status: 噪声状态
            hist_dist: 直方图分布
            distortion_score: 失真评分
            
        Returns:
            推荐的增强方法名称
        """
        # 如果失真程度低，使用自适应方法
        if distortion_score < 1.5:
            return 'adaptive'
        
        # 对比度不足是主要问题
        if contrast_status == 'low':
            if brightness_status == 'low':
                # 低亮度 + 低对比度：使用CLAHE（局部自适应）
                return 'clahe'
            elif brightness_status == 'high':
                # 高亮度 + 低对比度：使用对比度拉伸
                return 'contrast_stretch'
            else:
                # 正常亮度 + 低对比度：使用CLAHE
                return 'clahe'
        
        # 亮度问题
        if brightness_status == 'low':
            # 低亮度：使用伽马校正或直方图均衡化
            if hist_dist == 'skewed_left':
                return 'hist_eq'  # 直方图均衡化可以改善整体亮度分布
            else:
                return 'gamma'  # 伽马校正可以提升暗部细节
        elif brightness_status == 'high':
            # 高亮度：使用对比度拉伸或伽马校正
            return 'contrast_stretch'
        
        # 噪声问题
        if noise_status == 'high':
            # 高噪声：使用CLAHE（相比直方图均衡化，CLAHE对噪声更友好）
            return 'clahe'
        
        # 直方图分布不均
        if hist_dist in ['skewed_left', 'skewed_right']:
            return 'hist_eq'
        
        # 默认使用自适应方法
        return 'adaptive'
