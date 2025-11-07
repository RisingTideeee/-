"""
分类模块
实现SVM分类器和特征可视化
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict
import cv2


class DefectClassifier:
    """缺陷分类器类"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        """
        初始化分类器
        
        Args:
            kernel: SVM核类型
            C: 正则化参数
            gamma: 核函数参数
        """
        self.classifier = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.scaler = StandardScaler()
        self.pca = None
        self.is_trained = False
        self.class_names = []
    
    def train(self, X: np.ndarray, y: np.ndarray,
             test_size: float = 0.2,
             random_state: int = 42,
             use_pca: bool = False,
             n_components: int = 50) -> Dict:
        """
        训练分类器
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            test_size: 测试集比例
            random_state: 随机种子
            use_pca: 是否使用PCA降维
            n_components: PCA主成分数量
            
        Returns:
            训练信息字典
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # PCA降维（可选）
        if use_pca:
            self.pca = PCA(n_components=min(n_components, X_train_scaled.shape[1]))
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
        
        # 训练分类器
        self.classifier.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = self.classifier.predict(X_train_scaled)
        y_test_pred = self.classifier.predict(X_test_scaled)
        
        self.is_trained = True
        
        # 获取类别名称
        self.class_names = [f"Class_{i}" for i in np.unique(y)]
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测标签
        """
        if not self.is_trained:
            raise ValueError("分类器尚未训练，请先调用train()方法")
        
        X_scaled = self.scaler.transform(X)
        
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率
        """
        if not self.is_trained:
            raise ValueError("分类器尚未训练，请先调用train()方法")
        
        X_scaled = self.scaler.transform(X)
        
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        return self.classifier.predict_proba(X_scaled)
    
    def visualize_features(self, X: np.ndarray, y: np.ndarray,
                          save_path: Optional[str] = None,
                          n_components: int = 2):
        """
        特征可视化（PCA降维）
        
        Args:
            X: 特征矩阵
            y: 标签向量
            save_path: 保存路径（可选）
            n_components: 降维后的维度（2或3）
        """
        if not self.is_trained:
            # 如果未训练，使用临时scaler和pca
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
            X_reduced = pca.fit_transform(X_scaled)
        else:
            X_scaled = self.scaler.transform(X)
            if self.pca is not None:
                X_reduced = self.pca.transform(X_scaled)
            else:
                pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
                X_reduced = pca.fit_transform(X_scaled)
        
        # 可视化
        unique_labels = np.unique(y)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
        
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            for i, label in enumerate(unique_labels):
                mask = y == label
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                          c=[colors(i)], label=f'Class {label}', alpha=0.6)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('特征分布可视化 (PCA降维到2D)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        elif n_components == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            for i, label in enumerate(unique_labels):
                mask = y == label
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                          c=[colors(i)], label=f'Class {label}', alpha=0.6)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('特征分布可视化 (PCA降维到3D)')
            ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"特征可视化已保存至: {save_path}")
        plt.show()
    
    def visualize_classification_result(self, image: np.ndarray,
                                       defect_regions: List[dict],
                                       predictions: np.ndarray,
                                       class_names: Optional[List[str]] = None,
                                       save_path: Optional[str] = None):
        """
        可视化分类结果
        
        Args:
            image: 原始图像
            defect_regions: 缺陷区域列表
            predictions: 预测标签
            class_names: 类别名称列表（可选）
            save_path: 保存路径（可选）
        """
        result = image.copy()
        
        # 定义颜色
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红
            (0, 255, 255),  # 黄色
        ]
        
        for i, (region, pred) in enumerate(zip(defect_regions, predictions)):
            x, y, w, h = region['bbox']
            color = colors[pred % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # 标注类别
            if class_names:
                label = class_names[pred] if pred < len(class_names) else f"Class {pred}"
            else:
                label = f"Class {pred}"
            
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if save_path:
            cv2.imwrite(save_path, result)
            print(f"分类结果已保存至: {save_path}")
        
        return result

