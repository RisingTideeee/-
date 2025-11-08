"""
基于YOLOv11的缺陷检测与分割模块
使用深度学习模型进行缺陷检测、分割和分类
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil
import tempfile
from tqdm import tqdm
from image_enhancement import ImageEnhancer


class YOLODefectDetector:
    """基于YOLOv11的缺陷检测器"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 model_size: str = 'n',
                 task: str = 'segment',
                 enhance: bool = False,
                 enhance_method: str = 'adaptive',
                 **enhance_kwargs):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 预训练模型路径（.pt文件），如果为None则使用预训练权重
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')，仅在model_path为None时使用
            task: 任务类型 ('detect' 或 'segment')
            enhance: 是否使用图像增强
            enhance_method: 增强方法 ('hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive')
            **enhance_kwargs: 增强方法特定参数（如clip_limit, gamma等）
        """
        self.task = task
        self.model = None
        self.class_names = []
        self.enhance = enhance
        self.enhance_method = enhance_method
        self.enhance_kwargs = enhance_kwargs
        
        if enhance:
            self.enhancer = ImageEnhancer()
            print(f"图像增强已启用，方法: {enhance_method}")
        
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
        
        # 如果启用增强，先对图像进行增强
        if self.enhance:
            if self.enhance_method == 'adaptive':
                image = self.enhancer.adaptive_enhancement(image)
            else:
                image = self.enhancer.enhance(image, method=self.enhance_method, **self.enhance_kwargs)
        
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
        self.enhancer = ImageEnhancer()
        self.temp_enhanced_dir = None
    
    def _prepare_enhanced_dataset(self, data_yaml: str, 
                                   enhance_method: str = 'adaptive',
                                   **enhance_kwargs) -> str:
        """
        准备增强后的数据集（创建临时增强数据集）
        
        Args:
            data_yaml: 原始数据集配置文件路径
            enhance_method: 增强方法 ('hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive')
            **enhance_kwargs: 增强方法特定参数
            
        Returns:
            增强后数据集的yaml配置文件路径
        """
        print("\n" + "=" * 70)
        print("开始准备增强数据集...")
        print("=" * 70)
        
        # 读取原始配置
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        dataset_path = Path(data_yaml).parent
        original_path = Path(data_config.get('path', dataset_path))
        
        # 创建临时增强数据集目录
        temp_dir = Path(tempfile.mkdtemp(prefix='yolo_enhanced_'))
        self.temp_enhanced_dir = temp_dir
        
        print(f"临时增强数据集目录: {temp_dir}")
        
        # 创建目录结构
        (temp_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # 处理训练集
        train_relative = data_config.get('train', 'images/train')
        train_path = original_path / train_relative if not Path(train_relative).is_absolute() else Path(train_relative)
        train_labels_path = original_path / 'labels' / 'train'
        
        print(f"\n增强训练集图像: {train_path}")
        train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
        print(f"找到 {len(train_images)} 张训练图像")
        
        for img_path in tqdm(train_images, desc="增强训练图像"):
            # 读取并增强图像
            image = cv2.imread(str(img_path))
            if image is not None:
                if enhance_method == 'adaptive':
                    enhanced = self.enhancer.adaptive_enhancement(image)
                else:
                    enhanced = self.enhancer.enhance(image, method=enhance_method, **enhance_kwargs)
                
                # 保存增强后的图像
                output_img = temp_dir / 'images' / 'train' / img_path.name
                cv2.imwrite(str(output_img), enhanced)
                
                # 复制标签文件
                label_file = train_labels_path / (img_path.stem + '.txt')
                if label_file.exists():
                    output_label = temp_dir / 'labels' / 'train' / label_file.name
                    shutil.copy2(label_file, output_label)
        
        # 处理验证集
        val_relative = data_config.get('val', 'images/val')
        val_path = original_path / val_relative if not Path(val_relative).is_absolute() else Path(val_relative)
        val_labels_path = original_path / 'labels' / 'val'
        
        print(f"\n增强验证集图像: {val_path}")
        val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
        print(f"找到 {len(val_images)} 张验证图像")
        
        for img_path in tqdm(val_images, desc="增强验证图像"):
            # 读取并增强图像
            image = cv2.imread(str(img_path))
            if image is not None:
                if enhance_method == 'adaptive':
                    enhanced = self.enhancer.adaptive_enhancement(image)
                else:
                    enhanced = self.enhancer.enhance(image, method=enhance_method, **enhance_kwargs)
                
                # 保存增强后的图像
                output_img = temp_dir / 'images' / 'val' / img_path.name
                cv2.imwrite(str(output_img), enhanced)
                
                # 复制标签文件
                label_file = val_labels_path / (img_path.stem + '.txt')
                if label_file.exists():
                    output_label = temp_dir / 'labels' / 'val' / label_file.name
                    shutil.copy2(label_file, output_label)
        
        # 创建新的yaml配置文件
        enhanced_config = data_config.copy()
        enhanced_config['path'] = str(temp_dir.absolute())
        enhanced_config['train'] = 'images/train'
        enhanced_config['val'] = 'images/val'
        # 保持test集路径（如果存在）
        if 'test' in enhanced_config:
            test_relative = enhanced_config['test']
            if not Path(test_relative).is_absolute():
                enhanced_config['test'] = str((original_path / test_relative).absolute())
        
        enhanced_yaml = temp_dir / 'dataset.yaml'
        with open(enhanced_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(enhanced_config, f, allow_unicode=True)
        
        print(f"\n增强数据集准备完成！")
        print(f"增强方法: {enhance_method}")
        print(f"配置文件: {enhanced_yaml}")
        print("=" * 70)
        
        return str(enhanced_yaml)
    
    def _prepare_enhanced_test_dataset(self, data_yaml: str,
                                       enhance_method: str = 'adaptive',
                                       **enhance_kwargs) -> str:
        """
        准备增强后的测试集（创建临时增强测试集）
        
        Args:
            data_yaml: 原始数据集配置文件路径
            enhance_method: 增强方法 ('hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive')
            **enhance_kwargs: 增强方法特定参数
            
        Returns:
            增强后数据集的yaml配置文件路径
        """
        print("\n" + "=" * 70)
        print("开始准备增强测试集...")
        print("=" * 70)
        
        # 读取原始配置
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        dataset_path = Path(data_yaml).parent
        original_path = Path(data_config.get('path', dataset_path))
        
        # 创建临时增强数据集目录
        temp_dir = Path(tempfile.mkdtemp(prefix='yolo_enhanced_test_'))
        self.temp_enhanced_dir = temp_dir
        
        print(f"临时增强测试集目录: {temp_dir}")
        
        # 创建目录结构
        (temp_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (temp_dir / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        
        # 处理测试集
        if 'test' not in data_config:
            print("警告: 数据集配置中没有test集")
            return data_yaml
        
        test_relative = data_config.get('test', 'images/test')
        test_path = original_path / test_relative if not Path(test_relative).is_absolute() else Path(test_relative)
        test_labels_path = original_path / 'labels' / 'test'
        
        print(f"\n增强测试集图像: {test_path}")
        test_images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
        print(f"找到 {len(test_images)} 张测试图像")
        
        for img_path in tqdm(test_images, desc="增强测试图像"):
            # 读取并增强图像
            image = cv2.imread(str(img_path))
            if image is not None:
                if enhance_method == 'adaptive':
                    enhanced = self.enhancer.adaptive_enhancement(image)
                else:
                    enhanced = self.enhancer.enhance(image, method=enhance_method, **enhance_kwargs)
                
                # 保存增强后的图像
                output_img = temp_dir / 'images' / 'test' / img_path.name
                cv2.imwrite(str(output_img), enhanced)
                
                # 复制标签文件
                label_file = test_labels_path / (img_path.stem + '.txt')
                if label_file.exists():
                    output_label = temp_dir / 'labels' / 'test' / label_file.name
                    shutil.copy2(label_file, output_label)
        
        # 创建新的yaml配置文件
        enhanced_config = data_config.copy()
        enhanced_config['path'] = str(temp_dir.absolute())
        enhanced_config['test'] = 'images/test'
        # 保持train和val路径（如果存在）
        if 'train' in enhanced_config:
            train_relative = enhanced_config['train']
            if not Path(train_relative).is_absolute():
                enhanced_config['train'] = str((original_path / train_relative).absolute())
        if 'val' in enhanced_config:
            val_relative = enhanced_config['val']
            if not Path(val_relative).is_absolute():
                enhanced_config['val'] = str((original_path / val_relative).absolute())
        
        enhanced_yaml = temp_dir / 'dataset.yaml'
        with open(enhanced_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(enhanced_config, f, allow_unicode=True)
        
        print(f"\n增强测试集准备完成！")
        print(f"增强方法: {enhance_method}")
        print(f"配置文件: {enhanced_yaml}")
        print("=" * 70)
        
        return str(enhanced_yaml)
    
    def train(self, data_yaml: str,
              epochs: int = 100,
              imgsz: int = 640,
              batch: int = 16,
              device: str = 'cpu',
              project: str = 'runs',
              name: str = 'defect_detection',
              patience: int = 50,
              enhance: bool = False,
              enhance_method: str = 'adaptive',
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
            patience: 早停耐心值，如果验证指标在patience个epoch内没有提升则停止训练
            enhance: 是否使用图像增强（使用自定义增强算法）
            enhance_method: 增强方法 ('hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive')
            **kwargs: 其他训练参数（如lr0, lrf, weight_decay, dropout等）和增强参数（如clip_limit, gamma等）
        """
        # 分离增强参数和训练参数
        enhance_kwargs = {}
        train_kwargs = {}
        
        enhance_param_names = ['clip_limit', 'tile_grid_size', 'lower_percent', 
                              'upper_percent', 'gamma']
        
        for key, value in kwargs.items():
            if key in enhance_param_names:
                enhance_kwargs[key] = value
            else:
                train_kwargs[key] = value
        
        # 如果启用增强，准备增强数据集
        if enhance:
            enhanced_yaml = self._prepare_enhanced_dataset(
                data_yaml, 
                enhance_method=enhance_method,
                **enhance_kwargs
            )
            data_yaml = enhanced_yaml
        
        # 设置默认参数（如果kwargs中没有指定）
        train_kwargs.update({
            'patience': patience,  # 早停耐心值
            'save': True,  # 保存最佳模型
            'save_period': 10,  # 每10个epoch保存一次
        })
        
        try:
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=project,
                name=name,
                **train_kwargs
            )
            
            print(f"\n训练完成！模型保存在: {project}/{name}")
            print(f"最佳模型: {project}/{name}/weights/best.pt")
            
            return results
        finally:
            # 清理临时增强数据集
            if enhance and self.temp_enhanced_dir and self.temp_enhanced_dir.exists():
                print(f"\n清理临时增强数据集: {self.temp_enhanced_dir}")
                try:
                    shutil.rmtree(self.temp_enhanced_dir)
                    print("临时文件已清理")
                except Exception as e:
                    print(f"清理临时文件时出错: {e}")
                    print(f"请手动删除: {self.temp_enhanced_dir}")
    
    def validate(self, data_yaml: str):
        """
        验证模型
        
        Args:
            data_yaml: 数据集配置文件路径
        """
        results = self.model.val(data=data_yaml)
        return results

