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

