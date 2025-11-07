"""
基于YOLOv11的缺陷检测与分割模块
使用深度学习模型进行缺陷检测、分割和分类
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
from pathlib import Path


class YOLODefectDetector:
    """基于YOLOv11的缺陷检测器"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 model_size: str = 'n',
                 task: str = 'segment'):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 预训练模型路径（.pt文件），如果为None则使用预训练权重
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')，仅在model_path为None时使用
            task: 任务类型 ('detect' 或 'segment')
        """
        self.task = task
        self.model = None
        self.class_names = []
        
        if model_path and Path(model_path).exists():
            print(f"加载自定义模型: {model_path}")
            self.model = YOLO(model_path)
        else:
            # 使用预训练模型
            if task == 'segment':
                model_name = f'yolo11{model_size}-seg.pt'  # 分割模型
            else:
                model_name = f'yolo11{model_size}.pt'  # 检测模型
            
            print(f"加载预训练模型: {model_name}")
            self.model = YOLO(model_name)
        
        # 获取类别名称
        if hasattr(self.model, 'names'):
            self.class_names = list(self.model.names.values())
        else:
            self.class_names = []
    
    def detect(self, image: np.ndarray,
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45,
               show_labels: bool = True,
               show_conf: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        检测缺陷
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值（用于NMS）
            show_labels: 是否显示标签
            show_conf: 是否显示置信度
            
        Returns:
            检测结果图像和缺陷区域列表
        """
        if self.model is None:
            raise ValueError("模型未初始化")
        
        # 执行检测
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            task=self.task,
            verbose=False
        )
        
        # 解析结果
        defect_regions = []
        result_image = image.copy()
        
        if len(results) > 0:
            result = results[0]
            
            # 获取检测框和掩码
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            
            for i in range(len(boxes)):
                # 获取边界框
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box.astype(int)
                
                # 获取类别和置信度
                cls = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                
                # 获取类别名称
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class_{cls}"
                
                # 获取掩码（如果存在）
                mask = None
                if masks is not None:
                    mask_data = masks.data[i].cpu().numpy()
                    # 将掩码调整到图像尺寸
                    mask = (mask_data * 255).astype(np.uint8)
                    # 可能需要调整掩码尺寸
                    if mask.shape != image.shape[:2]:
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # 保存区域信息
                region_info = {
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'bbox_xyxy': (x1, y1, x2, y2),
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': conf,
                    'area': (x2 - x1) * (y2 - y1),
                    'mask': mask
                }
                defect_regions.append(region_info)
                
                # 绘制边界框
                color = self._get_color(cls)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                if show_labels:
                    label = class_name
                    if show_conf:
                        label += f" {conf:.2f}"
                    
                    # 计算文本大小
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # 绘制文本背景
                    cv2.rectangle(
                        result_image,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    
                    # 绘制文本
                    cv2.putText(
                        result_image,
                        label,
                        (x1, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                
                # 绘制掩码（如果存在）
                if mask is not None:
                    colored_mask = self._apply_mask(image, mask, color)
                    result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
        
        return result_image, defect_regions
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """获取类别对应的颜色"""
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
        ]
        return colors[class_id % len(colors)]
    
    def _apply_mask(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        """应用彩色掩码到图像"""
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        return colored_mask
    
    def create_segmentation_mask(self, image_shape: Tuple[int, int],
                                 defect_regions: List[Dict]) -> np.ndarray:
        """
        从检测结果创建分割掩码
        
        Args:
            image_shape: 图像形状 (height, width)
            defect_regions: 缺陷区域列表
            
        Returns:
            分割掩码
        """
        mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        
        for region in defect_regions:
            if region['mask'] is not None:
                # 使用掩码
                mask = cv2.bitwise_or(mask, region['mask'])
            else:
                # 使用边界框
                x1, y1, w, h = region['bbox']
                x2, y2 = x1 + w, y1 + h
                mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def predict_batch(self, images: List[np.ndarray],
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> List[Tuple[np.ndarray, List[Dict]]]:
        """
        批量检测
        
        Args:
            images: 图像列表
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            结果列表，每个元素为(结果图像, 缺陷区域列表)
        """
        results = []
        for image in images:
            result_image, defect_regions = self.detect(
                image, conf_threshold, iou_threshold
            )
            results.append((result_image, defect_regions))
        return results
    
    def export_model(self, format: str = 'onnx', output_path: Optional[str] = None):
        """
        导出模型
        
        Args:
            format: 导出格式 ('onnx', 'torchscript', 'tflite', etc.)
            output_path: 输出路径
        """
        if self.model is None:
            raise ValueError("模型未初始化")
        
        if output_path is None:
            output_path = f"yolo_model.{format}"
        
        self.model.export(format=format)
        print(f"模型已导出至: {output_path}")


class YOLOTrainer:
    """YOLO模型训练器"""
    
    def __init__(self, model_size: str = 'n', task: str = 'segment'):
        """
        初始化训练器
        
        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            task: 任务类型 ('detect' 或 'segment')
        """
        self.model_size = model_size
        self.task = task
        
        if task == 'segment':
            model_name = f'yolo11{model_size}-seg.pt'
        else:
            model_name = f'yolo11{model_size}.pt'
        
        self.model = YOLO(model_name)
    
    def train(self, data_yaml: str,
              epochs: int = 100,
              imgsz: int = 640,
              batch: int = 16,
              device: str = 'cpu',
              project: str = 'runs',
              name: str = 'defect_detection',
              **kwargs):
        """
        训练模型
        
        Args:
            data_yaml: 数据集配置文件路径（YOLO格式）
            epochs: 训练轮数
            imgsz: 图像尺寸
            batch: 批次大小
            device: 设备 ('cpu', 'cuda', '0', '1', etc.)
            project: 项目目录
            name: 实验名称
            **kwargs: 其他训练参数
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            **kwargs
        )
        
        print(f"训练完成！模型保存在: {project}/{name}")
        return results
    
    def validate(self, data_yaml: str):
        """
        验证模型
        
        Args:
            data_yaml: 数据集配置文件路径
        """
        results = self.model.val(data=data_yaml)
        return results

