"""
批量图像增强脚本
用于批量处理训练数据或测试数据
支持根据失真类型和程度自动选择增强算法
"""
import argparse
from pathlib import Path
import cv2
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_enhancement import ImageEnhancer
from evaluation import ImageQualityAssessment, DistortionAnalyzer


def main():
    parser = argparse.ArgumentParser(description='批量图像增强工具')
    parser.add_argument('--input', type=str, required=True,
                       help='输入目录路径（包含图像文件的目录）')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录路径（增强后的图像保存位置）')
    parser.add_argument('--method', type=str, default='auto',
                       choices=['hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive', 'auto'],
                       help='增强方法（默认：auto，自动根据失真类型选择）')
    parser.add_argument('--preserve_structure', action='store_true', default=True,
                       help='保持目录结构（默认：True）')
    parser.add_argument('--evaluate', action='store_true',
                       help='是否评估增强后的图像质量')
    parser.add_argument('--analyze', action='store_true',
                       help='是否分析图像失真类型（使用auto模式时自动启用）')
    parser.add_argument('--clip_limit', type=float, default=2.0,
                       help='CLAHE的对比度限制（仅当method=clahe时使用）')
    parser.add_argument('--gamma', type=float, default=1.5,
                       help='伽马值（仅当method=gamma时使用）')
    
    args = parser.parse_args()
    
    # 如果使用auto模式，自动启用分析
    if args.method == 'auto':
        args.analyze = True
    
    # 创建增强器和分析器
    enhancer = ImageEnhancer()
    analyzer = DistortionAnalyzer() if args.analyze or args.method == 'auto' else None
    iqa = ImageQualityAssessment() if args.evaluate else None
    
    # 准备参数
    kwargs = {}
    if args.method == 'clahe':
        kwargs['clip_limit'] = args.clip_limit
    elif args.method == 'gamma':
        kwargs['gamma'] = args.gamma
    
    print("=" * 60)
    print("批量图像增强工具")
    print("=" * 60)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"增强方法: {args.method}")
    if args.method == 'auto':
        print("  → 自动模式：根据失真类型和程度选择最佳增强算法")
    print(f"失真分析: {'启用' if args.analyze or args.method == 'auto' else '禁用'}")
    print(f"保持目录结构: {args.preserve_structure}")
    print("=" * 60)
    
    # 执行批量增强
    try:
        if args.method == 'auto':
            # 智能批量增强：根据每张图像的失真类型自动选择方法
            stats = _smart_batch_enhance(
                enhancer, analyzer, iqa,
                args.input, args.output,
                args.preserve_structure
            )
        else:
            # 传统批量增强：使用固定方法
            stats = enhancer.enhance_from_directory(
                input_dir=args.input,
                output_dir=args.output,
                method=args.method,
                preserve_structure=args.preserve_structure,
                **kwargs
            )
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"总计: {stats['total']} 张")
        print(f"成功: {stats['success']} 张")
        print(f"失败: {stats['failed']} 张")
        
        if stats['failed'] > 0:
            print(f"\n失败的文件:")
            for failed_file in stats['failed_files'][:10]:  # 只显示前10个
                print(f"  - {failed_file}")
            if len(stats['failed_files']) > 10:
                print(f"  ... 还有 {len(stats['failed_files']) - 10} 个文件失败")
        
        # 如果启用评估，评估增强后的图像
        if args.evaluate and iqa:
            print("\n" + "=" * 60)
            print("评估增强后的图像质量...")
            print("=" * 60)
            
            # 评估输出目录中的图像
            output_dir = Path(args.output)
            enhanced_files = list(output_dir.rglob('*.jpg')) + list(output_dir.rglob('*.png'))
            
            if len(enhanced_files) > 0:
                # 随机选择一些图像进行评估（最多10张）
                import random
                sample_files = random.sample(enhanced_files, min(10, len(enhanced_files)))
                
                brisque_scores = []
                niqe_scores = []
                
                for img_file in sample_files:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        brisque = iqa.calculate_quality_score(img, 'brisque')
                        niqe = iqa.calculate_quality_score(img, 'niqe')
                        brisque_scores.append(brisque)
                        niqe_scores.append(niqe)
                
                if brisque_scores:
                    avg_brisque = sum(brisque_scores) / len(brisque_scores)
                    avg_niqe = sum(niqe_scores) / len(niqe_scores)
                    print(f"平均 BRISQUE 分数: {avg_brisque:.4f} (越低越好)")
                    print(f"平均 NIQE 分数: {avg_niqe:.4f} (越低越好)")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


def _smart_batch_enhance(enhancer: ImageEnhancer, analyzer: DistortionAnalyzer,
                        iqa: ImageQualityAssessment, input_dir: str, output_dir: str,
                        preserve_structure: bool) -> dict:
    """
    智能批量增强：根据每张图像的失真类型自动选择增强方法
    
    Args:
        enhancer: 图像增强器
        analyzer: 失真分析器
        iqa: 图像质量评估器（可选）
        input_dir: 输入目录
        output_dir: 输出目录
        preserve_structure: 是否保持目录结构
        
    Returns:
        处理结果统计字典
    """
    from pathlib import Path
    from tqdm import tqdm
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        raise ValueError(f"在 {input_dir} 中未找到图像文件")
    
    print(f"找到 {len(image_files)} 张图像")
    print("开始智能分析并增强...\n")
    
    # 统计信息
    stats = {
        'total': len(image_files),
        'success': 0,
        'failed': 0,
        'failed_files': [],
        'method_usage': {},  # 记录各方法使用次数
        'distortion_stats': {
            'low': 0,
            'medium': 0,
            'high': 0
        }
    }
    
    # 批量处理
    for img_path in tqdm(image_files, desc="智能增强中"):
        try:
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                stats['failed'] += 1
                stats['failed_files'].append(str(img_path))
                continue
            
            # 分析失真类型
            distortion_info = analyzer.analyze_distortion(image)
            recommended_method = distortion_info['recommended_method']
            severity = distortion_info['distortion_severity']
            
            # 更新统计
            stats['method_usage'][recommended_method] = stats['method_usage'].get(recommended_method, 0) + 1
            stats['distortion_stats'][severity] = stats['distortion_stats'].get(severity, 0) + 1
            
            # 根据推荐方法增强图像
            if recommended_method == 'adaptive':
                enhanced = enhancer.adaptive_enhancement(image)
            elif recommended_method == 'clahe':
                enhanced = enhancer.enhance(image, method='clahe', clip_limit=2.0)
            elif recommended_method == 'hist_eq':
                enhanced = enhancer.enhance(image, method='hist_eq')
            elif recommended_method == 'contrast_stretch':
                enhanced = enhancer.enhance(image, method='contrast_stretch')
            elif recommended_method == 'gamma':
                enhanced = enhancer.enhance(image, method='gamma', gamma=1.5)
            else:
                enhanced = enhancer.adaptive_enhancement(image)
            
            # 确定输出路径
            if preserve_structure:
                relative_path = img_path.relative_to(input_path)
                output_file = output_path / relative_path
            else:
                output_file = output_path / img_path.name
            
            # 创建输出目录
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存增强后的图像
            cv2.imwrite(str(output_file), enhanced)
            stats['success'] += 1
            
        except Exception as e:
            stats['failed'] += 1
            stats['failed_files'].append(str(img_path))
            print(f"处理 {img_path} 时出错: {e}")
    
    # 打印方法使用统计
    print("\n" + "=" * 60)
    print("增强方法使用统计:")
    print("=" * 60)
    for method, count in sorted(stats['method_usage'].items(), key=lambda x: x[1], reverse=True):
        method_names = {
            'adaptive': '自适应增强',
            'clahe': 'CLAHE',
            'hist_eq': '直方图均衡化',
            'contrast_stretch': '对比度拉伸',
            'gamma': '伽马校正'
        }
        method_name = method_names.get(method, method)
        percentage = (count / stats['success']) * 100 if stats['success'] > 0 else 0
        print(f"  {method_name}: {count} 张 ({percentage:.1f}%)")
    
    print("\n失真程度分布:")
    print("=" * 60)
    for severity, count in stats['distortion_stats'].items():
        severity_names = {'low': '轻微', 'medium': '中等', 'high': '严重'}
        severity_name = severity_names.get(severity, severity)
        percentage = (count / stats['success']) * 100 if stats['success'] > 0 else 0
        print(f"  {severity_name}失真: {count} 张 ({percentage:.1f}%)")
    
    return stats


if __name__ == '__main__':
    main()

