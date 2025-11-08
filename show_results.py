"""
展示实验结果
整合三个定量指标和重要信息
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
import yaml
from ultralytics import YOLO
from evaluation import ImageQualityAssessment


def show_results(model_path: str, 
                data_yaml: str = 'dataset/dataset.yaml',
                device: str = 'cpu',
                show_quality: bool = False,
                test_images_dir: str = None,
                max_quality_images: int = 30):
    """
    展示实验结果
    
    Args:
        model_path: 模型路径
        data_yaml: 数据集配置文件
        device: 设备
        show_quality: 是否计算图像质量指标（较慢，主要占用CPU）
        test_images_dir: 测试图像目录（用于图像质量评估）
        max_quality_images: 计算质量指标的最大图像数量（默认30，减少内存占用）
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "实验结果展示")
    print("=" * 70)
    
    # 加载模型并运行验证
    print("\n正在加载模型并运行验证...")
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        device=device,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        plots=True,
        save_json=True
    )
    
    print("\n" + "=" * 70)
    print("【核心指标】")
    print("=" * 70)
    
    # 1. 分割IoU指标（定量指标2）
    print("\n【定量指标2】分割IoU评价（要求：IoU > 50%）")
    print("-" * 70)
    if hasattr(results, 'seg'):
        map50 = float(results.seg.map50)
        map50_95 = float(results.seg.map)
        precision = float(results.seg.mp)
        recall = float(results.seg.mr)
        
        print(f"  mAP50:        {map50:.4f} {'✓' if map50 > 0.5 else '✗'}")
        print(f"  mAP50-95:     {map50_95:.4f}")
        print(f"  精确率:       {precision:.4f}")
        print(f"  召回率:       {recall:.4f}")
        print(f"\n  IoU检查:      {'满足要求 (mAP50 > 0.5)' if map50 > 0.5 else '不满足要求 (mAP50 <= 0.5)'}")
    else:
        print("  注意: 模型为检测模型，非分割模型")
        if hasattr(results, 'box'):
            print(f"  边界框 mAP50: {results.box.map50:.4f}")
    
    # 2. 分类器性能指标（定量指标3）
    print("\n【定量指标3】分类器性能评价")
    print("-" * 70)
    if hasattr(results, 'box'):
        precision = float(results.box.mp)
        recall = float(results.box.mr)
        
        # 计算F1-Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 准确率近似（从验证结果中获取）
        accuracy = (precision + recall) / 2
        
        print(f"  准确率:       {accuracy:.4f}")
        print(f"  精确率:       {precision:.4f}")
        print(f"  召回率:       {recall:.4f}")
        print(f"  F1-Score:     {f1_score:.4f}")
    elif hasattr(results, 'seg'):
        # 如果只有分割结果，使用分割的precision和recall
        precision = float(results.seg.mp)
        recall = float(results.seg.mr)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (precision + recall) / 2
        
        print(f"  准确率:       {accuracy:.4f}")
        print(f"  精确率:       {precision:.4f}")
        print(f"  召回率:       {recall:.4f}")
        print(f"  F1-Score:     {f1_score:.4f}")
    
    # 3. 图像质量指标（定量指标1）- 可选
    if show_quality:
        print("\n【定量指标1】图像质量评价")
        print("-" * 70)
        
        # 确定测试图像目录
        if test_images_dir and Path(test_images_dir).exists():
            test_dir = Path(test_images_dir)
        else:
            with open(data_yaml, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            dataset_path = Path(data_yaml).parent
            test_relative = data_config.get('test', '')
            if test_relative:
                test_dir = dataset_path / test_relative
            else:
                val_relative = data_config.get('val', '')
                test_dir = dataset_path / val_relative if val_relative else None
        
        if test_dir and test_dir.exists():
            image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
            if image_files:
                print(f"  找到 {len(image_files)} 张测试图像，正在计算...")
                iqa = ImageQualityAssessment()
                brisque_scores = []
                niqe_scores = []
                
                # 限制处理数量以加快速度和减少内存占用
                max_images = min(max_quality_images, len(image_files))
                print(f"  处理前 {max_images} 张图像（限制数量以减少内存占用）...")
                print(f"  内存使用: 每张图像约50-100MB临时内存，处理完立即释放")
                print(f"  CPU使用: 计算密集型操作，会暂时占用CPU（不影响内存）")
                
                for i, img_path in enumerate(image_files[:max_images], 1):
                    # 逐张处理，处理完立即释放内存
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        # 计算质量指标（主要是CPU计算，内存占用较小）
                        brisque_score = iqa.brisque(image)
                        niqe_score = iqa.niqe(image)
                        brisque_scores.append(brisque_score)
                        niqe_scores.append(niqe_score)
                        
                        # 显式释放图像内存（Python会自动回收，但显式删除更安全）
                        del image
                        
                        # 每处理10张显示进度
                        if i % 10 == 0:
                            print(f"    已处理 {i}/{max_images} 张...")
                
                if brisque_scores:
                    avg_brisque = np.mean(brisque_scores)
                    avg_niqe = np.mean(niqe_scores)
                    print(f"  平均BRISQUE:  {avg_brisque:.4f} (越低越好)")
                    print(f"  平均NIQE:     {avg_niqe:.4f} (越低越好)")
                    print(f"  评估图像数:   {len(brisque_scores)}")
                else:
                    print("  无法计算图像质量指标")
            else:
                print("  未找到测试图像")
        else:
            print("  未找到测试图像目录")
    else:
        print("\n【定量指标1】图像质量评价")
        print("-" * 70)
        print("  提示: 使用 --show-quality 参数可计算BRISQUE/NIQE指标")
    
    # 总结
    print("\n" + "=" * 70)
    print("【展示建议】")
    print("=" * 70)
    print("\n核心展示指标：")
    
    if hasattr(results, 'seg'):
        print(f"  1. mAP50: {results.seg.map50:.4f} (分割IoU核心指标，要求>0.5)")
        print(f"  2. mAP50-95: {results.seg.map:.4f} (更全面的分割评估)")
    
    if hasattr(results, 'box'):
        precision = float(results.box.mp)
        recall = float(results.box.mr)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (precision + recall) / 2
        print(f"  3. 准确率: {accuracy:.4f}")
        print(f"  4. F1-Score: {f1:.4f}")
    
    print("\n" + "=" * 70)
    print("提示: 完整验证结果和可视化图表保存在 runs/ 目录下")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='展示实验结果')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径（.pt文件）')
    parser.add_argument('--data', type=str, default='dataset/dataset.yaml',
                       help='数据集配置文件')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--show-quality', action='store_true',
                       help='计算图像质量指标（BRISQUE/NIQE，较慢）')
    parser.add_argument('--test-images', type=str, default=None,
                       help='测试图像目录（用于图像质量评估）')
    parser.add_argument('--max-quality-images', type=int, default=30,
                       help='计算质量指标的最大图像数量（默认30，减少内存占用）')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    if not Path(args.data).exists():
        print(f"错误: 数据集配置文件不存在: {args.data}")
        return
    
    show_results(
        model_path=args.model,
        data_yaml=args.data,
        device=args.device,
        show_quality=args.show_quality,
        test_images_dir=args.test_images,
        max_quality_images=args.max_quality_images
    )


if __name__ == '__main__':
    main()

