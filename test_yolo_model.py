"""
YOLO模型批量测试脚本
在测试集上验证模型性能，生成详细评估报告和可视化结果
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
import json
import yaml
import shutil
from typing import Dict, List, Tuple
from datetime import datetime

from yolo_segmenter import YOLODefectDetector, YOLOTrainer
from ultralytics import YOLO
from evaluation import SegmentationEvaluation, ClassificationEvaluation


def test_model_validation(model_path: str, data_yaml: str, 
                         device: str = 'cpu', 
                         imgsz: int = 640,
                         conf: float = 0.25,
                         iou: float = 0.45,
                         enhance: bool = False,
                         enhance_method: str = 'adaptive',
                         **enhance_kwargs):
    """
    在测试集上运行验证，获取评估指标
    
    Args:
        model_path: 模型路径
        data_yaml: 数据集配置文件
        device: 设备
        imgsz: 图像尺寸
        conf: 置信度阈值
        iou: IoU阈值
        enhance: 是否使用图像增强
        enhance_method: 增强方法 ('hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive')
        **enhance_kwargs: 增强方法特定参数
        
    Returns:
        验证结果字典
    """
    print("=" * 60)
    print("开始模型验证...")
    if enhance:
        print(f"图像增强已启用，方法: {enhance_method}")
    print("=" * 60)
    
    # 如果启用增强，需要准备增强后的测试集
    trainer = None
    if enhance:
        print("准备增强后的测试集...")
        trainer = YOLOTrainer()
        # 创建增强后的测试集
        enhanced_yaml = trainer._prepare_enhanced_test_dataset(
            data_yaml,
            enhance_method=enhance_method,
            **enhance_kwargs
        )
        data_yaml = enhanced_yaml
    
    # 加载模型
    model = YOLO(model_path)
    
    # 检查是否有test集配置
    data_content = Path(data_yaml).read_text()
    has_test = 'test:' in data_content
    
    # 创建临时配置文件，强制使用test集
    if has_test:
        print("检测到test集配置，创建临时配置文件使用test集进行验证")
        # 读取原始配置
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 将test路径临时设为val路径（YOLO的val方法默认使用val集）
        original_val = data_config.get('val', '')
        test_path = data_config.get('test', '')
        if test_path:
            # 创建临时yaml文件，将test路径设为val
            temp_yaml = Path(data_yaml).parent / 'temp_test_config.yaml'
            data_config['val'] = test_path  # 临时替换val为test
            with open(temp_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(data_config, f, allow_unicode=True)
            data_yaml = str(temp_yaml)
            print(f"使用临时配置文件: {data_yaml}")
    else:
        print("未检测到test集配置，将在val集上验证")
    
    # 运行验证
    val_kwargs = {
        'data': data_yaml,
        'device': device,
        'imgsz': imgsz,
        'conf': conf,
        'iou': iou,
        'save_json': True,
        'plots': True
    }
    
    results = model.val(**val_kwargs)
    
    # 清理临时文件
    temp_files_to_clean = []
    if has_test and Path(data_yaml).name == 'temp_test_config.yaml':
        temp_files_to_clean.append(Path(data_yaml))
    
    # 清理增强测试集的临时目录（如果存在）
    if enhance and trainer is not None:
        if hasattr(trainer, 'temp_enhanced_dir') and trainer.temp_enhanced_dir and trainer.temp_enhanced_dir.exists():
            temp_files_to_clean.append(trainer.temp_enhanced_dir)
    
    for temp_file in temp_files_to_clean:
        try:
            if temp_file.is_file():
                temp_file.unlink()
                print(f"已清理临时配置文件: {temp_file}")
            elif temp_file.is_dir():
                shutil.rmtree(temp_file)
                print(f"已清理临时目录: {temp_file}")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
    
    # 提取关键指标
    metrics = {
        'mAP50_bbox': float(results.box.map50) if hasattr(results, 'box') else 0.0,
        'mAP50_95_bbox': float(results.box.map) if hasattr(results, 'box') else 0.0,
        'mAP50_mask': float(results.seg.map50) if hasattr(results, 'seg') else 0.0,
        'mAP50_95_mask': float(results.seg.map) if hasattr(results, 'seg') else 0.0,
        'precision': float(results.box.mp) if hasattr(results, 'box') else 0.0,
        'recall': float(results.box.mr) if hasattr(results, 'box') else 0.0,
    }
    
    print("\n验证完成！")
    print(f"mAP50 (BBox): {metrics['mAP50_bbox']:.4f}")
    print(f"mAP50-95 (BBox): {metrics['mAP50_95_bbox']:.4f}")
    if metrics['mAP50_mask'] > 0:
        print(f"mAP50 (Mask): {metrics['mAP50_mask']:.4f}")
        print(f"mAP50-95 (Mask): {metrics['mAP50_95_mask']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    return metrics, results


def batch_test_images(model_path: str, 
                     test_images_dir: str,
                     test_labels_dir: str = None,
                     output_dir: str = 'test_results',
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45,
                     device: str = 'cpu',
                     save_images: bool = True,
                     save_json: bool = True,
                     enhance: bool = False,
                     enhance_method: str = 'adaptive',
                     **enhance_kwargs):
    """
    批量处理测试图像，生成可视化结果
    
    Args:
        model_path: 模型路径
        test_images_dir: 测试图像目录
        test_labels_dir: 测试标签目录（可选，用于计算IoU）
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        device: 设备
        save_images: 是否保存可视化图像
        save_json: 是否保存JSON结果
        enhance: 是否使用图像增强
        enhance_method: 增强方法 ('hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive')
        **enhance_kwargs: 增强方法特定参数
        
    Returns:
        测试结果列表
    """
    print("=" * 60)
    print("开始批量测试图像...")
    if enhance:
        print(f"图像增强已启用，方法: {enhance_method}")
    print("=" * 60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_images:
        (output_path / 'images').mkdir(exist_ok=True)
    
    # 加载模型（启用增强）
    detector = YOLODefectDetector(
        model_path=model_path,
        enhance=enhance,
        enhance_method=enhance_method,
        **enhance_kwargs
    )
    
    # 获取测试图像
    test_images_path = Path(test_images_dir)
    image_files = list(test_images_path.glob('*.jpg')) + list(test_images_path.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"错误: 在 {test_images_dir} 中未找到图像文件")
        return []
    
    print(f"找到 {len(image_files)} 张测试图像")
    
    all_results = []
    total_defects = 0
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n处理 [{idx}/{len(image_files)}]: {image_file.name}")
        
        # 读取图像
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  警告: 无法读取图像 {image_file.name}")
            continue
        
        # 执行检测
        result_image, defect_regions = detector.detect(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # 统计信息
        num_defects = len(defect_regions)
        total_defects += num_defects
        
        # 保存结果图像
        if save_images:
            output_image_path = output_path / 'images' / f"{image_file.stem}_result.jpg"
            cv2.imwrite(str(output_image_path), result_image)
        
        # 准备结果数据
        result_data = {
            'image_name': image_file.name,
            'num_defects': int(num_defects),
            'defects': []
        }
        
        for defect in defect_regions:
            # 转换NumPy类型为Python原生类型
            bbox = defect['bbox']
            bbox_xyxy = defect['bbox_xyxy']
            
            # 确保bbox和bbox_xyxy是列表格式
            if hasattr(bbox, 'tolist'):
                bbox = bbox.tolist()
            elif not isinstance(bbox, list):
                bbox = list(bbox)
            
            if hasattr(bbox_xyxy, 'tolist'):
                bbox_xyxy = bbox_xyxy.tolist()
            elif not isinstance(bbox_xyxy, list):
                bbox_xyxy = list(bbox_xyxy)
            
            defect_info = {
                'class_id': int(defect['class_id']),
                'class_name': defect['class_name'],
                'confidence': float(defect['confidence']),
                'bbox': [float(x) for x in bbox],
                'bbox_xyxy': [float(x) for x in bbox_xyxy],
                'area': int(defect['area'])
            }
            result_data['defects'].append(defect_info)
        
        all_results.append(result_data)
        
        print(f"  检测到 {num_defects} 个缺陷")
        for defect in defect_regions:
            print(f"    - {defect['class_name']}: {defect['confidence']:.3f}")
    
    # 保存JSON结果
    if save_json:
        json_path = output_path / 'test_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_path': str(model_path),
                'test_images_dir': str(test_images_dir),
                'total_images': int(len(image_files)),
                'total_defects': int(total_defects),
                'avg_defects_per_image': float(total_defects / len(image_files) if len(image_files) > 0 else 0),
                'conf_threshold': float(conf_threshold),
                'iou_threshold': float(iou_threshold),
                'timestamp': datetime.now().isoformat(),
                'results': all_results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {json_path}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("批量测试统计")
    print("=" * 60)
    print(f"总图像数: {len(image_files)}")
    print(f"总缺陷数: {total_defects}")
    print(f"平均每张图像缺陷数: {total_defects / len(image_files):.2f}" if len(image_files) > 0 else "平均每张图像缺陷数: 0")
    
    # 按类别统计
    class_counts = {}
    for result in all_results:
        for defect in result['defects']:
            class_name = defect['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        print("\n按类别统计:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")
    
    return all_results


def load_yolo_label(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    """
    加载YOLO格式的标签文件（分割格式：多边形坐标）
    
    Args:
        label_path: 标签文件路径
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        标签列表，每个元素包含class_id和归一化的多边形坐标
    """
    labels = []
    if not label_path.exists():
        return labels
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                class_id = int(parts[0])
                # 剩余部分是归一化的多边形坐标 (x1, y1, x2, y2, ...)
                coords = [float(x) for x in parts[1:]]
                
                labels.append({
                    'class_id': class_id,
                    'coords': coords  # 归一化坐标
                })
    except Exception as e:
        print(f"  警告: 读取标签文件失败 {label_path}: {e}")
    
    return labels


def yolo_coords_to_mask(coords: List[float], img_width: int, img_height: int) -> np.ndarray:
    """
    将YOLO格式的归一化多边形坐标转换为分割掩码
    
    Args:
        coords: 归一化的多边形坐标列表 [x1, y1, x2, y2, ...]
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        二值掩码图像
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if len(coords) < 6:  # 至少需要3个点（6个坐标值）
        return mask
    
    # 将归一化坐标转换为像素坐标
    points = []
    for i in range(0, len(coords), 2):
        if i + 1 < len(coords):
            x = int(coords[i] * img_width)
            y = int(coords[i + 1] * img_height)
            points.append([x, y])
    
    if len(points) >= 3:
        # 将点转换为numpy数组并填充多边形
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    
    return mask


def calculate_bbox_iou(bbox1: Tuple[float, float, float, float], 
                       bbox2: Tuple[float, float, float, float]) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        bbox1: (x1, y1, x2, y2) 格式的边界框
        bbox2: (x1, y1, x2, y2) 格式的边界框
        
    Returns:
        IoU值 (0-1之间)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def yolo_coords_to_bbox(coords: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    将YOLO格式的归一化多边形坐标转换为边界框
    
    Args:
        coords: 归一化的多边形坐标列表 [x1, y1, x2, y2, ...]
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        (x1, y1, x2, y2) 格式的边界框
    """
    if len(coords) < 6:
        return (0.0, 0.0, 0.0, 0.0)
    
    # 将归一化坐标转换为像素坐标
    x_coords = []
    y_coords = []
    for i in range(0, len(coords), 2):
        if i + 1 < len(coords):
            x = coords[i] * img_width
            y = coords[i + 1] * img_height
            x_coords.append(x)
            y_coords.append(y)
    
    if not x_coords or not y_coords:
        return (0.0, 0.0, 0.0, 0.0)
    
    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)
    
    return (x1, y1, x2, y2)


def calculate_comprehensive_metrics(model_path: str,
                                   test_images_dir: str,
                                   test_labels_dir: str,
                                   data_yaml: str,
                                   conf_threshold: float = 0.25,
                                   iou_threshold: float = 0.45,
                                   device: str = 'cpu',
                                   enhance: bool = False,
                                   enhance_method: str = 'adaptive',
                                   **enhance_kwargs) -> Dict:
    """
    计算完整的评估指标：IoU、准确率、精确率、召回率、F1-Score
    
    Args:
        model_path: 模型路径
        test_images_dir: 测试图像目录
        test_labels_dir: 测试标签目录
        data_yaml: 数据集配置文件
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        device: 设备
        enhance: 是否使用图像增强
        enhance_method: 增强方法
        **enhance_kwargs: 增强方法特定参数
        
    Returns:
        包含所有评估指标的字典
    """
    print("=" * 60)
    print("开始计算完整评估指标...")
    print("=" * 60)
    
    # 加载数据集配置获取类别名称
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get('names', {})
    if isinstance(class_names, dict):
        class_names = {int(k): v for k, v in class_names.items()}
    
    # 加载模型
    detector = YOLODefectDetector(
        model_path=model_path,
        enhance=enhance,
        enhance_method=enhance_method,
        **enhance_kwargs
    )
    
    # 获取测试图像
    test_images_path = Path(test_images_dir)
    test_labels_path = Path(test_labels_dir) if test_labels_dir else None
    
    image_files = list(test_images_path.glob('*.jpg')) + list(test_images_path.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"错误: 在 {test_images_dir} 中未找到图像文件")
        return {}
    
    print(f"找到 {len(image_files)} 张测试图像")
    
    # 用于计算IoU的变量
    seg_eval = SegmentationEvaluation()
    all_ious = []
    iou_per_image = []
    
    # 用于计算分类指标的变量（基于检测框匹配）
    matched_true_classes = []  # 匹配的真实类别
    matched_pred_classes = []  # 匹配的预测类别
    total_predictions = 0  # 总预测数
    total_ground_truth = 0  # 总真实标注数
    match_iou_threshold = 0.5  # 匹配IoU阈值
    
    # 逐图像处理
    for idx, image_file in enumerate(image_files, 1):
        if idx % 10 == 0:
            print(f"处理进度: [{idx}/{len(image_files)}]")
        
        # 读取图像
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        # 执行检测
        _, defect_regions = detector.detect(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # 加载真实标签
        label_file = None
        if test_labels_path:
            label_file = test_labels_path / f"{image_file.stem}.txt"
        
        gt_labels = []
        if label_file and label_file.exists():
            gt_labels = load_yolo_label(label_file, img_width, img_height)
        
        # 创建预测掩码和真实掩码
        pred_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        gt_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # 准备预测框和真实框用于匹配
        pred_boxes = []  # [(bbox, class_id), ...]
        gt_boxes = []    # [(bbox, class_id), ...]
        
        # 从预测结果创建掩码和收集检测框
        for defect in defect_regions:
            bbox = defect['bbox_xyxy']
            class_id = defect['class_id']
            pred_boxes.append((bbox, class_id))
            
            if 'mask' in defect and defect['mask'] is not None:
                mask = defect['mask']
                # 确保掩码是二值图像
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                # 调整掩码尺寸
                if mask.shape != (img_height, img_width):
                    mask = cv2.resize(mask, (img_width, img_height))
                
                # 二值化掩码
                _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                pred_mask = np.maximum(pred_mask, mask_binary)
            else:
                # 如果没有掩码，使用边界框
                x1, y1, x2, y2 = bbox
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                if x2 > x1 and y2 > y1:
                    pred_mask[y1:y2, x1:x2] = 255
        
        # 从真实标签创建掩码和收集真实框
        for gt_label in gt_labels:
            coords = gt_label['coords']
            mask = yolo_coords_to_mask(coords, img_width, img_height)
            gt_mask = np.maximum(gt_mask, mask)
            
            # 将YOLO坐标转换为边界框
            bbox = yolo_coords_to_bbox(coords, img_width, img_height)
            class_id = gt_label['class_id']
            gt_boxes.append((bbox, class_id))
        
        # 计算IoU
        if np.any(gt_mask > 0) or np.any(pred_mask > 0):
            iou = seg_eval.calculate_iou(pred_mask, gt_mask)
            all_ious.append(iou)
            iou_per_image.append({
                'image': image_file.name,
                'iou': float(iou)
            })
        
        # 基于IoU匹配检测框并计算分类指标
        # 使用贪心匹配：对每个预测框，找到IoU最大的未匹配真实框
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        # 计算所有预测框和真实框之间的IoU
        for pred_idx, (pred_bbox, pred_class) in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, (gt_bbox, gt_class) in enumerate(gt_boxes):
                if gt_idx in matched_gt_indices:
                    continue
                
                iou = calculate_bbox_iou(pred_bbox, gt_bbox)
                if iou > best_iou and iou >= match_iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # 如果找到匹配，记录分类结果
            if best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                matched_pred_indices.add(pred_idx)
                _, gt_class = gt_boxes[best_gt_idx]
                matched_true_classes.append(gt_class)
                matched_pred_classes.append(pred_class)
        
        # 统计总数
        total_predictions += len(pred_boxes)
        total_ground_truth += len(gt_boxes)
    
    # 计算IoU统计
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    iou_above_50 = sum(1 for iou in all_ious if iou > 0.5) / len(all_ious) if all_ious else 0.0
    
    # 计算分类指标（基于匹配的检测框）
    classification_metrics = {}
    
    if matched_true_classes and matched_pred_classes:
        # 计算匹配的检测框的分类准确率
        correct_classifications = sum(1 for true_cls, pred_cls in 
                                     zip(matched_true_classes, matched_pred_classes) 
                                     if true_cls == pred_cls)
        classification_accuracy = correct_classifications / len(matched_true_classes) if matched_true_classes else 0.0
        
        # 计算精确率：匹配的检测框中分类正确的比例
        precision = correct_classifications / len(matched_pred_classes) if matched_pred_classes else 0.0
        
        # 计算召回率：正确分类的检测框占所有真实标注的比例
        recall = correct_classifications / total_ground_truth if total_ground_truth > 0 else 0.0
        
        # 计算F1-Score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        classification_metrics = {
            'accuracy': float(classification_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'matched_detections': len(matched_true_classes),
            'total_predictions': total_predictions,
            'total_ground_truth': total_ground_truth
        }
    else:
        classification_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'matched_detections': 0,
            'total_predictions': total_predictions,
            'total_ground_truth': total_ground_truth
        }
    
    # 汇总结果
    metrics = {
        'segmentation': {
            'mean_iou': float(mean_iou),
            'iou_above_50_percentage': float(iou_above_50 * 100),
            'iou_above_50_count': int(sum(1 for iou in all_ious if iou > 0.5)),
            'total_images': len(all_ious),
            'iou_per_image': iou_per_image
        },
        'classification': classification_metrics,
        'total_images': len(image_files)
    }
    
    print("\n" + "=" * 60)
    print("评估指标计算结果")
    print("=" * 60)
    print(f"\n分割指标 (Segmentation):")
    print(f"  平均IoU: {mean_iou:.4f}")
    print(f"  IoU > 0.5 的比例: {iou_above_50 * 100:.2f}% ({metrics['segmentation']['iou_above_50_count']}/{len(all_ious)})")
    
    if classification_metrics:
        print(f"\n分类指标 (Classification):")
        print(f"  准确率 (Accuracy): {classification_metrics.get('accuracy', 0):.4f}")
        print(f"  精确率 (Precision): {classification_metrics.get('precision', 0):.4f}")
        print(f"  召回率 (Recall): {classification_metrics.get('recall', 0):.4f}")
        print(f"  F1-Score: {classification_metrics.get('f1_score', 0):.4f}")
    
    return metrics


def generate_test_report(model_path: str,
                        validation_metrics: Dict,
                        batch_results: List[Dict],
                        comprehensive_metrics: Dict = None,
                        output_dir: str = 'test_results'):
    """
    生成测试报告
    
    Args:
        model_path: 模型路径
        validation_metrics: 验证指标
        batch_results: 批量测试结果
        output_dir: 输出目录
    """
    report_path = Path(output_dir) / 'test_report.md'
    
    total_images = len(batch_results)
    total_defects = sum(r['num_defects'] for r in batch_results)
    avg_defects = total_defects / total_images if total_images > 0 else 0
    
    report = f"""# YOLO模型测试报告

## 模型信息
- **模型路径**: `{model_path}`
- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 验证指标（在测试集上）

### 边界框检测 (Bounding Box)
- **mAP50**: {validation_metrics.get('mAP50_bbox', 0):.4f}
- **mAP50-95**: {validation_metrics.get('mAP50_95_bbox', 0):.4f}

### 分割掩码 (Mask)
- **mAP50**: {validation_metrics.get('mAP50_mask', 0):.4f}
- **mAP50-95**: {validation_metrics.get('mAP50_95_mask', 0):.4f}

### 其他指标
- **Precision**: {validation_metrics.get('precision', 0):.4f}
- **Recall**: {validation_metrics.get('recall', 0):.4f}

## 批量测试统计

- **测试图像数**: {total_images}
- **检测到缺陷总数**: {total_defects}
- **平均每张图像缺陷数**: {avg_defects:.2f}

## 按类别统计

"""
    
    # 按类别统计
    class_counts = {}
    for result in batch_results:
        for defect in result['defects']:
            class_name = defect['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_defects * 100) if total_defects > 0 else 0
            report += f"- **{class_name}**: {count} ({percentage:.1f}%)\n"
    else:
        report += "- 无缺陷检测\n"
    
    # 添加完整评估指标
    if comprehensive_metrics:
        report += "\n## 定量评估指标\n\n"
        
        # 分割指标
        seg_metrics = comprehensive_metrics.get('segmentation', {})
        if seg_metrics:
            report += "### 分割指标 (Segmentation)\n\n"
            report += f"- **平均IoU**: {seg_metrics.get('mean_iou', 0):.4f}\n"
            report += f"- **IoU > 0.5 的比例**: {seg_metrics.get('iou_above_50_percentage', 0):.2f}% "
            report += f"({seg_metrics.get('iou_above_50_count', 0)}/{seg_metrics.get('total_images', 0)})\n"
            report += f"- **评估图像数**: {seg_metrics.get('total_images', 0)}\n\n"
            
            # 判断是否满足要求
            if seg_metrics.get('mean_iou', 0) > 0.5:
                report += "✅ **IoU满足要求 (> 0.5)**\n\n"
            else:
                report += "❌ **IoU不满足要求 (< 0.5)**\n\n"
        
        # 分类指标
        class_metrics = comprehensive_metrics.get('classification', {})
        if class_metrics:
            report += "### 分类指标 (Classification)\n\n"
            report += f"- **准确率 (Accuracy)**: {class_metrics.get('accuracy', 0):.4f}\n"
            report += f"- **精确率 (Precision)**: {class_metrics.get('precision', 0):.4f}\n"
            report += f"- **召回率 (Recall)**: {class_metrics.get('recall', 0):.4f}\n"
            report += f"- **F1-Score**: {class_metrics.get('f1_score', 0):.4f}\n\n"
            
            report += f"- **匹配的检测数**: {class_metrics.get('matched_detections', 0)}\n"
            report += f"- **总预测数**: {class_metrics.get('total_predictions', 0)}\n"
            report += f"- **总真实标注数**: {class_metrics.get('total_ground_truth', 0)}\n\n"
    
    report += f"""
## 详细结果

详细结果保存在 `test_results.json` 文件中。

## 可视化结果

所有可视化结果保存在 `images/` 目录中。
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n测试报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLO模型批量测试脚本')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径 (.pt文件)')
    parser.add_argument('--data', type=str, default='dataset/dataset.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--test_images', type=str, default='dataset/test/images',
                       help='测试图像目录')
    parser.add_argument('--test_labels', type=str, default='dataset/test/labels',
                       help='测试标签目录（用于计算IoU和分类指标，默认: dataset/test/labels）')
    parser.add_argument('--output', type=str, default='test_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值')
    parser.add_argument('--skip_validation', action='store_true',
                       help='跳过验证步骤，只进行批量测试')
    parser.add_argument('--skip_batch', action='store_true',
                       help='跳过批量测试，只进行验证')
    parser.add_argument('--enhance', action='store_true',
                       help='启用图像增强（与训练时保持一致）')
    parser.add_argument('--enhance-method', type=str, default='adaptive',
                       choices=['hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive'],
                       help='增强方法 (默认: adaptive)')
    parser.add_argument('--clip-limit', type=float, default=2.0,
                       help='CLAHE的对比度限制（仅当--enhance-method=clahe时使用）')
    parser.add_argument('--gamma', type=float, default=1.5,
                       help='伽马值（仅当--enhance-method=gamma时使用）')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    # 检查数据集配置
    if not Path(args.data).exists():
        print(f"错误: 数据集配置文件不存在: {args.data}")
        return
    
    validation_metrics = {}
    batch_results = []
    comprehensive_metrics = {}
    
    # 准备增强参数
    enhance_kwargs = {}
    if args.enhance_method == 'clahe':
        enhance_kwargs['clip_limit'] = args.clip_limit
    elif args.enhance_method == 'gamma':
        enhance_kwargs['gamma'] = args.gamma
    
    # 0. 计算完整评估指标（如果有标签文件）
    if args.test_labels and Path(args.test_labels).exists():
        try:
            comprehensive_metrics = calculate_comprehensive_metrics(
                model_path=args.model,
                test_images_dir=args.test_images,
                test_labels_dir=args.test_labels,
                data_yaml=args.data,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                device=args.device,
                enhance=args.enhance,
                enhance_method=args.enhance_method,
                **enhance_kwargs
            )
        except Exception as e:
            print(f"计算完整评估指标时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("警告: 未提供测试标签目录，跳过完整评估指标计算")
        print("      如需计算IoU和分类指标，请使用 --test_labels 参数")
    
    # 1. 运行验证（如果数据集配置中有test集）
    if not args.skip_validation:
        try:
            # 检查是否有test集配置
            data_content = Path(args.data).read_text()
            if 'test:' in data_content or Path(args.test_images).exists():
                # 临时修改data_yaml添加test路径（如果需要）
                validation_metrics, _ = test_model_validation(
                    model_path=args.model,
                    data_yaml=args.data,
                    device=args.device,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    enhance=args.enhance,
                    enhance_method=args.enhance_method,
                    **enhance_kwargs
                )
            else:
                print("警告: 数据集配置中没有test集，跳过验证步骤")
        except Exception as e:
            print(f"验证过程中出错: {e}")
            print("继续执行批量测试...")
    else:
        print("跳过验证步骤")
    
    # 2. 批量测试图像
    if not args.skip_batch:
        if not Path(args.test_images).exists():
            print(f"错误: 测试图像目录不存在: {args.test_images}")
            return
        
        batch_results = batch_test_images(
            model_path=args.model,
            test_images_dir=args.test_images,
            test_labels_dir=args.test_labels,
            output_dir=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            save_images=True,
            save_json=True,
            enhance=args.enhance,
            enhance_method=args.enhance_method,
            **enhance_kwargs
        )
    else:
        print("跳过批量测试步骤")
    
    # 3. 生成报告
    if validation_metrics or batch_results or comprehensive_metrics:
        generate_test_report(
            model_path=args.model,
            validation_metrics=validation_metrics,
            batch_results=batch_results,
            comprehensive_metrics=comprehensive_metrics,
            output_dir=args.output
        )
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

