"""
对比增强前后的图像质量
用于判断增强是否有效，从而决定是否需要重新训练模型
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import ImageQualityAssessment


def calculate_quality_metrics(image_path: Path, iqa: ImageQualityAssessment) -> dict:
    """
    计算单张图像的质量指标
    
    Args:
        image_path: 图像路径
        iqa: 图像质量评估器
        
    Returns:
        质量指标字典
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    metrics = {
        'brisque': iqa.brisque(image),
        'niqe': iqa.niqe(image),
    }
    
    # 计算额外的统计信息
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    metrics['mean_brightness'] = float(np.mean(gray))
    metrics['std_contrast'] = float(np.std(gray))
    
    return metrics


def evaluate_image_quality(images_dir: Path, iqa: ImageQualityAssessment, 
                           max_images: int = None) -> dict:
    """
    评估图像目录中所有图像的质量
    
    Args:
        images_dir: 图像目录
        iqa: 图像质量评估器
        max_images: 最大处理图像数量（None表示处理所有）
        
    Returns:
        质量统计字典
    """
    # 查找所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        return None
    
    # 限制处理数量
    if max_images and len(image_files) > max_images:
        import random
        image_files = random.sample(image_files, max_images)
        print(f"  随机选择 {max_images} 张图像进行评估...")
    
    print(f"  找到 {len(image_files)} 张图像，正在计算质量指标...")
    
    all_metrics = {
        'brisque': [],
        'niqe': [],
        'mean_brightness': [],
        'std_contrast': []
    }
    
    failed_count = 0
    
    for img_path in tqdm(image_files, desc="  计算质量指标"):
        metrics = calculate_quality_metrics(img_path, iqa)
        if metrics:
            all_metrics['brisque'].append(metrics['brisque'])
            all_metrics['niqe'].append(metrics['niqe'])
            all_metrics['mean_brightness'].append(metrics['mean_brightness'])
            all_metrics['std_contrast'].append(metrics['std_contrast'])
        else:
            failed_count += 1
    
    if len(all_metrics['brisque']) == 0:
        return None
    
    # 计算统计信息
    stats = {
        'total_images': len(image_files),
        'successful': len(all_metrics['brisque']),
        'failed': failed_count,
        'avg_brisque': float(np.mean(all_metrics['brisque'])),
        'std_brisque': float(np.std(all_metrics['brisque'])),
        'avg_niqe': float(np.mean(all_metrics['niqe'])),
        'std_niqe': float(np.std(all_metrics['niqe'])),
        'avg_brightness': float(np.mean(all_metrics['mean_brightness'])),
        'avg_contrast': float(np.mean(all_metrics['std_contrast'])),
    }
    
    return stats


def compare_quality(original_stats: dict, enhanced_stats: dict, 
                   output_file: str = None) -> dict:
    """
    对比原始图像和增强图像的质量
    
    Args:
        original_stats: 原始图像质量统计
        enhanced_stats: 增强图像质量统计
        output_file: 输出报告文件路径
        
    Returns:
        对比结果字典
    """
    print("\n" + "=" * 70)
    print("图像质量对比分析")
    print("=" * 70)
    
    comparison = {}
    
    # BRISQUE对比（越低越好）
    if 'avg_brisque' in original_stats and 'avg_brisque' in enhanced_stats:
        orig_brisque = original_stats['avg_brisque']
        enh_brisque = enhanced_stats['avg_brisque']
        brisque_improvement = orig_brisque - enh_brisque  # 降低表示改善
        brisque_improvement_percent = (brisque_improvement / orig_brisque * 100) if orig_brisque > 0 else 0
        
        comparison['brisque'] = {
            'original': orig_brisque,
            'enhanced': enh_brisque,
            'improvement': brisque_improvement,
            'improvement_percent': brisque_improvement_percent
        }
        
        print("\n【BRISQUE 指标】（越低越好）")
        print("-" * 70)
        print(f"原始图像: {orig_brisque:.4f}")
        print(f"增强图像: {enh_brisque:.4f}")
        print(f"改善程度: {brisque_improvement:+.4f} ({brisque_improvement_percent:+.2f}%)")
        if brisque_improvement > 0:
            print("  ✓ 质量提升！")
        elif brisque_improvement < -0.5:
            print("  ⚠️  质量下降")
        else:
            print("  → 质量基本不变")
    
    # NIQE对比（越低越好）
    if 'avg_niqe' in original_stats and 'avg_niqe' in enhanced_stats:
        orig_niqe = original_stats['avg_niqe']
        enh_niqe = enhanced_stats['avg_niqe']
        niqe_improvement = orig_niqe - enh_niqe  # 降低表示改善
        niqe_improvement_percent = (niqe_improvement / orig_niqe * 100) if orig_niqe > 0 else 0
        
        comparison['niqe'] = {
            'original': orig_niqe,
            'enhanced': enh_niqe,
            'improvement': niqe_improvement,
            'improvement_percent': niqe_improvement_percent
        }
        
        print("\n【NIQE 指标】（越低越好）")
        print("-" * 70)
        print(f"原始图像: {orig_niqe:.4f}")
        print(f"增强图像: {enh_niqe:.4f}")
        print(f"改善程度: {niqe_improvement:+.4f} ({niqe_improvement_percent:+.2f}%)")
        if niqe_improvement > 0:
            print("  ✓ 质量提升！")
        elif niqe_improvement < -0.5:
            print("  ⚠️  质量下降")
        else:
            print("  → 质量基本不变")
    
    # 亮度对比
    if 'avg_brightness' in original_stats and 'avg_brightness' in enhanced_stats:
        orig_bright = original_stats['avg_brightness']
        enh_bright = enhanced_stats['avg_brightness']
        bright_change = enh_bright - orig_bright
        
        comparison['brightness'] = {
            'original': orig_bright,
            'enhanced': enh_bright,
            'change': bright_change
        }
        
        print("\n【平均亮度】")
        print("-" * 70)
        print(f"原始图像: {orig_bright:.2f}")
        print(f"增强图像: {enh_bright:.2f}")
        print(f"变化: {bright_change:+.2f}")
    
    # 对比度对比
    if 'avg_contrast' in original_stats and 'avg_contrast' in enhanced_stats:
        orig_contrast = original_stats['avg_contrast']
        enh_contrast = enhanced_stats['avg_contrast']
        contrast_change = enh_contrast - orig_contrast
        
        comparison['contrast'] = {
            'original': orig_contrast,
            'enhanced': enh_contrast,
            'change': contrast_change
        }
        
        print("\n【平均对比度】")
        print("-" * 70)
        print(f"原始图像: {orig_contrast:.2f}")
        print(f"增强图像: {enh_contrast:.2f}")
        print(f"变化: {contrast_change:+.2f}")
    
    # 总结和建议
    print("\n" + "=" * 70)
    print("【质量评估总结】")
    print("=" * 70)
    
    # 判断质量是否提升
    quality_improved = False
    significant_improvement = False
    
    if 'brisque' in comparison:
        brisque_imp = comparison['brisque']['improvement']
        if brisque_imp > 1.0:  # BRISQUE降低超过1.0表示明显改善
            significant_improvement = True
            quality_improved = True
        elif brisque_imp > 0:
            quality_improved = True
    
    if 'niqe' in comparison:
        niqe_imp = comparison['niqe']['improvement']
        if niqe_imp > 0.5:  # NIQE降低超过0.5表示明显改善
            if not significant_improvement:
                significant_improvement = True
            quality_improved = True
        elif niqe_imp > 0:
            quality_improved = True
    
    if significant_improvement:
        print("✓ 检测到明显的图像质量提升！")
        print("\n建议: 强烈建议重新训练模型")
        print("原因:")
        print("  1. 增强后的图像质量显著提升（BRISQUE/NIQE明显降低）")
        print("  2. 使用增强后的高质量图像训练，模型性能会更好")
        print("  3. 原始模型在低质量图像上训练，可能没有学到足够的特征")
        recommendation = 'strongly_recommend_retrain'
    elif quality_improved:
        print("→ 检测到图像质量有所提升")
        print("\n建议: 建议重新训练模型")
        print("原因:")
        print("  1. 增强后的图像质量有所改善")
        print("  2. 使用增强后的图像训练可以获得更好的模型性能")
        print("  3. 虽然提升不是特别显著，但重新训练仍然值得")
        recommendation = 'recommend_retrain'
    else:
        print("⚠️  图像质量提升不明显或下降")
        print("\n建议: 可以尝试其他增强方法，或保持现状")
        print("原因:")
        print("  1. 当前增强方法对图像质量改善有限")
        print("  2. 可能需要调整增强参数或尝试其他方法")
        print("  3. 如果质量下降，建议不使用该增强方法")
        recommendation = 'no_retrain_needed'
    
    # 保存对比报告
    if output_file:
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_stats': original_stats,
            'enhanced_stats': enhanced_stats,
            'comparison': comparison,
            'recommendation': recommendation,
            'quality_improved': quality_improved,
            'significant_improvement': significant_improvement
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n对比报告已保存: {output_file}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='对比增强前后的图像质量，判断是否需要重新训练模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 对比原始图像和增强图像的质量
  python scripts/compare_image_quality.py \\
      --original dataset/test/images \\
      --enhanced dataset/test/images_enhanced \\
      --output quality_comparison_report.json \\
      --max-images 50
        """
    )
    
    parser.add_argument('--original', type=str, required=True,
                       help='原始图像目录路径')
    parser.add_argument('--enhanced', type=str, required=True,
                       help='增强后的图像目录路径')
    parser.add_argument('--output', type=str, default='quality_comparison_report.json',
                       help='输出报告文件路径（JSON格式）')
    parser.add_argument('--max-images', type=int, default=None,
                       help='最大处理图像数量（默认：处理所有图像，建议50-100张）')
    
    args = parser.parse_args()
    
    # 检查路径
    original_dir = Path(args.original)
    enhanced_dir = Path(args.enhanced)
    
    if not original_dir.exists():
        print(f"错误: 原始图像目录不存在: {original_dir}")
        return
    
    if not enhanced_dir.exists():
        print(f"错误: 增强图像目录不存在: {enhanced_dir}")
        return
    
    print("=" * 70)
    print("图像质量对比工具")
    print("=" * 70)
    print(f"原始图像目录: {original_dir}")
    print(f"增强图像目录: {enhanced_dir}")
    if args.max_images:
        print(f"最大处理数量: {args.max_images} 张")
    print("=" * 70)
    
    # 初始化质量评估器
    iqa = ImageQualityAssessment()
    
    # 1. 评估原始图像质量
    print("\n【步骤 1/2】评估原始图像质量...")
    original_stats = evaluate_image_quality(original_dir, iqa, args.max_images)
    
    if original_stats is None:
        print("错误: 无法评估原始图像质量")
        return
    
    print(f"\n原始图像质量统计:")
    print(f"  处理图像数: {original_stats['successful']}/{original_stats['total_images']}")
    print(f"  平均 BRISQUE: {original_stats['avg_brisque']:.4f} (越低越好)")
    print(f"  平均 NIQE: {original_stats['avg_niqe']:.4f} (越低越好)")
    print(f"  平均亮度: {original_stats['avg_brightness']:.2f}")
    print(f"  平均对比度: {original_stats['avg_contrast']:.2f}")
    
    # 2. 评估增强图像质量
    print("\n【步骤 2/2】评估增强图像质量...")
    enhanced_stats = evaluate_image_quality(enhanced_dir, iqa, args.max_images)
    
    if enhanced_stats is None:
        print("错误: 无法评估增强图像质量")
        return
    
    print(f"\n增强图像质量统计:")
    print(f"  处理图像数: {enhanced_stats['successful']}/{enhanced_stats['total_images']}")
    print(f"  平均 BRISQUE: {enhanced_stats['avg_brisque']:.4f} (越低越好)")
    print(f"  平均 NIQE: {enhanced_stats['avg_niqe']:.4f} (越低越好)")
    print(f"  平均亮度: {enhanced_stats['avg_brightness']:.2f}")
    print(f"  平均对比度: {enhanced_stats['avg_contrast']:.2f}")
    
    # 3. 对比结果
    comparison = compare_quality(
        original_stats=original_stats,
        enhanced_stats=enhanced_stats,
        output_file=args.output
    )
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    if args.output:
        print(f"详细报告已保存: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()

