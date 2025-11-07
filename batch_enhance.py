"""
批量图像增强脚本
用于批量处理训练数据或测试数据
"""
import argparse
from pathlib import Path
from image_enhancement import ImageEnhancer
from evaluation import ImageQualityAssessment


def main():
    parser = argparse.ArgumentParser(description='批量图像增强工具')
    parser.add_argument('--input', type=str, required=True,
                       help='输入目录路径（包含图像文件的目录）')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录路径（增强后的图像保存位置）')
    parser.add_argument('--method', type=str, default='adaptive',
                       choices=['hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive'],
                       help='增强方法（默认：adaptive）')
    parser.add_argument('--preserve_structure', action='store_true', default=True,
                       help='保持目录结构（默认：True）')
    parser.add_argument('--evaluate', action='store_true',
                       help='是否评估增强后的图像质量')
    parser.add_argument('--clip_limit', type=float, default=2.0,
                       help='CLAHE的对比度限制（仅当method=clahe时使用）')
    parser.add_argument('--gamma', type=float, default=1.5,
                       help='伽马值（仅当method=gamma时使用）')
    
    args = parser.parse_args()
    
    # 创建增强器
    enhancer = ImageEnhancer()
    
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
    print(f"保持目录结构: {args.preserve_structure}")
    print("=" * 60)
    
    # 执行批量增强
    try:
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
        if args.evaluate:
            print("\n" + "=" * 60)
            print("评估增强后的图像质量...")
            print("=" * 60)
            iqa = ImageQualityAssessment()
            
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
                    import cv2
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


if __name__ == '__main__':
    main()

