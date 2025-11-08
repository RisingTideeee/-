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
from typing import Dict, List, Tuple
from datetime import datetime

from yolo_segmenter import YOLODefectDetector, YOLOTrainer
from ultralytics import YOLO


def test_model_validation(model_path: str, data_yaml: str, 
                         device: str = 'cpu', 
                         imgsz: int = 640,
                         conf: float = 0.25,
                         iou: float = 0.45):
    """
    在测试集上运行验证，获取评估指标
    
    Args:
        model_path: 模型路径
        data_yaml: 数据集配置文件
        device: 设备
        imgsz: 图像尺寸
        conf: 置信度阈值
        iou: IoU阈值
        
    Returns:
        验证结果字典
    """
    print("=" * 60)
    print("开始模型验证...")
    print("=" * 60)
    
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
    if has_test and Path(data_yaml).name == 'temp_test_config.yaml':
        try:
            Path(data_yaml).unlink()
            print("已清理临时配置文件")
        except:
            pass
    
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
                     save_json: bool = True):
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
        
    Returns:
        测试结果列表
    """
    print("=" * 60)
    print("开始批量测试图像...")
    print("=" * 60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_images:
        (output_path / 'images').mkdir(exist_ok=True)
    
    # 加载模型
    detector = YOLODefectDetector(model_path=model_path)
    
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
            'num_defects': num_defects,
            'defects': []
        }
        
        for defect in defect_regions:
            defect_info = {
                'class_id': defect['class_id'],
                'class_name': defect['class_name'],
                'confidence': float(defect['confidence']),
                'bbox': defect['bbox'],
                'bbox_xyxy': defect['bbox_xyxy'],
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
                'model_path': model_path,
                'test_images_dir': test_images_dir,
                'total_images': len(image_files),
                'total_defects': total_defects,
                'avg_defects_per_image': total_defects / len(image_files) if len(image_files) > 0 else 0,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
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


def generate_test_report(model_path: str,
                        validation_metrics: Dict,
                        batch_results: List[Dict],
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
- **平均每张图像缺陷数**: {total_defects / total_images:.2f if total_images > 0 else 0}

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
    parser.add_argument('--test_labels', type=str, default=None,
                       help='测试标签目录（可选）')
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
                    iou=args.iou
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
            save_json=True
        )
    else:
        print("跳过批量测试步骤")
    
    # 3. 生成报告
    if validation_metrics or batch_results:
        generate_test_report(
            model_path=args.model,
            validation_metrics=validation_metrics,
            batch_results=batch_results,
            output_dir=args.output
        )
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

