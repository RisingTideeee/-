"""
主程序
整合图像增强、缺陷分割、特征提取与分类三个模块
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

from image_enhancement import ImageEnhancer
from defect_segmentation import DefectSegmenter
from yolo_segmenter import YOLODefectDetector
from feature_extraction import FeatureExtractor
from classification import DefectClassifier
from evaluation import ImageQualityAssessment, SegmentationEvaluation, ClassificationEvaluation
from utils import load_image, save_image, create_output_dir, visualize_results


def main():
    parser = argparse.ArgumentParser(description='汽车漆面缺陷检测系统')
    parser.add_argument('--input', type=str, default="D:/Users/CY/Desktop/数字图像处理基础/dataset/test/images/DSC_0296_JPG.rf.cd20e3364e76f2d222e12677de2c55df.jpg", help='输入图像路径')
    parser.add_argument('--output', type=str, default='D:/Users/CY/Desktop/数字图像处理基础/output', help='输出目录')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'enhance', 'segment', 'classify'],
                       help='运行模式: all(全部), enhance(仅增强), segment(仅分割), classify(仅分类)')
    parser.add_argument('--gt_mask', type=str, default=None, 
                       help='真实分割掩码路径（用于计算IoU）')
    parser.add_argument('--train_data', type=str, default=None,
                       help='训练数据目录（用于分类）')
    parser.add_argument('--use_yolo', default=True,
                       help='使用YOLOv11模型进行检测（替代传统方法）')
    parser.add_argument('--yolo_model', type=str, default="D:/Users/CY/Desktop/数字图像处理基础/runs/defect_detection5/weights/best.pt",
                       help='YOLO模型路径（.pt文件），如果未指定则使用预训练模型')
    parser.add_argument('--yolo_size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO模型大小（仅在未指定模型路径时使用）')
    parser.add_argument('--yolo_task', type=str, default='segment',
                       choices=['detect', 'segment'],
                       help='YOLO任务类型：detect(检测) 或 segment(分割)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='YOLO置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                       help='YOLO IoU阈值')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_output_dir(args.output)
    
    # 加载图像
    print(f"加载图像: {args.input}")
    original_image = load_image(args.input)
    
    # ========== 任务一：图像增强 ==========
    if args.mode in ['all', 'enhance']:
        print("\n=== 任务一：图像增强 ===")
        
        enhancer = ImageEnhancer()
        
        # 尝试多种增强方法
        methods = {
            'hist_eq': '直方图均衡化',
            'clahe': 'CLAHE',
            'contrast_stretch': '对比度拉伸',
            'adaptive': '自适应增强'
        }
        
        enhanced_images = {}
        quality_scores = {}
        iqa = ImageQualityAssessment()
        
        for method, name in methods.items():
            if method == 'adaptive':
                enhanced = enhancer.adaptive_enhancement(original_image)
            else:
                enhanced = enhancer.enhance(original_image, method=method)
            
            enhanced_images[method] = enhanced
            
            # 计算质量分数
            brisque_score = iqa.calculate_quality_score(enhanced, 'brisque')
            niqe_score = iqa.calculate_quality_score(enhanced, 'niqe')
            quality_scores[method] = {'brisque': brisque_score, 'niqe': niqe_score}
            
            print(f"{name}:")
            print(f"  BRISQUE分数: {brisque_score:.4f} (越低越好)")
            print(f"  NIQE分数: {niqe_score:.4f} (越低越好)")
            
            # 保存增强结果
            save_image(enhanced, f"{output_dir}/enhanced_{method}.jpg")
        
        # 选择最佳增强方法（BRISQUE分数最低）
        best_method = min(quality_scores.keys(), 
                         key=lambda x: quality_scores[x]['brisque'])
        best_enhanced = enhanced_images[best_method]
        
        print(f"\n最佳增强方法: {methods[best_method]}")
        save_image(best_enhanced, f"{output_dir}/enhanced_best.jpg")
        
        # 可视化增强结果
        visualize_results(
            {'original': original_image, 'enhanced': best_enhanced},
            {'original': '原始图像', 'enhanced': f'增强图像 ({methods[best_method]})'},
            f"{output_dir}/enhancement_comparison.png"
        )
    else:
        # 如果只运行分割或分类，仍需要增强图像
        enhancer = ImageEnhancer()
        best_enhanced = enhancer.adaptive_enhancement(original_image)
    
    # ========== 任务二：缺陷分割 ==========
    if args.mode in ['all', 'segment']:
        print("\n=== 任务二：缺陷分割 ===")
        
        if args.use_yolo:
            # 使用YOLOv11进行检测
            print("使用YOLOv11模型进行缺陷检测...")
            yolo_detector = YOLODefectDetector(
                model_path=args.yolo_model,
                model_size=args.yolo_size,
                task=args.yolo_task
            )
            
            # 执行检测
            result_image, defect_regions = yolo_detector.detect(
                best_enhanced,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold
            )
            
            # 创建分割掩码
            segmented_mask = yolo_detector.create_segmentation_mask(
                best_enhanced.shape[:2],
                defect_regions
            )
            
            print(f"检测到 {len(defect_regions)} 个缺陷区域")
            for i, region in enumerate(defect_regions):
                print(f"  区域 {i+1}: {region['class_name']} "
                      f"(置信度: {region['confidence']:.2f}), "
                      f"边界框={region['bbox']}")
            
            # 保存结果
            save_image(result_image, f"{output_dir}/yolo_detection_result.jpg")
            save_image(segmented_mask, f"{output_dir}/segmented_mask.jpg")
            
            # 可视化
            visualization = result_image
            save_image(visualization, f"{output_dir}/segmentation_result.jpg")
            
        else:
            # 使用传统方法
            print("使用传统图像处理方法进行缺陷分割...")
            segmenter = DefectSegmenter()
            
            # 执行分割
            segmented_mask, defect_regions = segmenter.segment(
                best_enhanced,
                window_size=15,
                threshold=0.15,
                min_area=100
            )
            
            print(f"检测到 {len(defect_regions)} 个缺陷区域")
            for i, region in enumerate(defect_regions):
                print(f"  区域 {i+1}: 面积={region['area']}, 边界框={region['bbox']}")
            
            # 保存分割结果
            save_image(segmented_mask, f"{output_dir}/segmented_mask.jpg")
            
            # 可视化分割结果
            visualization = segmenter.visualize_segmentation(
                original_image, segmented_mask, defect_regions
            )
            save_image(visualization, f"{output_dir}/segmentation_result.jpg")
        
        # 计算IoU（如果有真实掩码）
        if args.gt_mask:
            print("\n计算分割评价指标...")
            gt_mask = load_image(args.gt_mask)
            if len(gt_mask.shape) == 3:
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            
            seg_eval = SegmentationEvaluation()
            iou = seg_eval.calculate_iou(segmented_mask, gt_mask)
            dice = seg_eval.calculate_dice(segmented_mask, gt_mask)
            pixel_acc = seg_eval.calculate_pixel_accuracy(segmented_mask, gt_mask)
            
            print(f"IoU: {iou:.4f} (要求 > 0.5)")
            print(f"Dice系数: {dice:.4f}")
            print(f"像素准确率: {pixel_acc:.4f}")
            
            if iou > 0.5:
                print("✓ IoU满足要求 (> 0.5)")
            else:
                print("✗ IoU不满足要求 (< 0.5)")
    else:
        # 如果只运行分类，需要先进行分割
        if args.use_yolo:
            yolo_detector = YOLODefectDetector(
                model_path=args.yolo_model,
                model_size=args.yolo_size,
                task=args.yolo_task
            )
            _, defect_regions = yolo_detector.detect(
                best_enhanced,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold
            )
            segmented_mask = yolo_detector.create_segmentation_mask(
                best_enhanced.shape[:2],
                defect_regions
            )
        else:
            segmenter = DefectSegmenter()
            segmented_mask, defect_regions = segmenter.segment(best_enhanced)
    
    # ========== 任务三：特征提取与分类 ==========
    if args.mode in ['all', 'classify']:
        print("\n=== 任务三：特征提取与分类 ===")
        
        if len(defect_regions) == 0:
            print("警告: 未检测到缺陷区域，无法进行分类")
            return
        
        # 如果使用YOLO且模型已经进行了分类，直接使用YOLO的分类结果
        if args.use_yolo and len(defect_regions) > 0:
            print("YOLO模型已自动完成分类...")
            predictions = [region['class_id'] for region in defect_regions]
            class_names = [region['class_name'] for region in defect_regions]
            
            # 统计各类别数量
            unique_classes = {}
            for region in defect_regions:
                cls_name = region['class_name']
                if cls_name not in unique_classes:
                    unique_classes[cls_name] = 0
                unique_classes[cls_name] += 1
            
            print("分类结果统计:")
            for cls_name, count in unique_classes.items():
                print(f"  {cls_name}: {count} 个")
            
            # 可视化分类结果（YOLO已经绘制了结果）
            # result_image在分割阶段已经保存
            if args.mode == 'classify':
                # 如果只运行分类模式，需要重新加载结果图像
                result_image_path = f"{output_dir}/yolo_detection_result.jpg"
                if Path(result_image_path).exists():
                    result_image = load_image(result_image_path)
                    save_image(result_image, f"{output_dir}/classification_result.jpg")
            
            # 如果需要计算评价指标，需要真实标签
            if args.train_data:
                print("\n注意: YOLO已自动分类，如需评价需要提供真实标签")
        else:
            # 使用传统方法：特征提取 + 分类器
            feature_extractor = FeatureExtractor()
            
            # 提取所有缺陷区域的特征
            print("提取特征...")
            features_list = []
            for region in defect_regions:
                # 提取缺陷区域
                x, y, w, h = region['bbox']
                roi = best_enhanced[y:y+h, x:x+w]
                
                # 获取掩码
                if 'mask' in region and region['mask'] is not None:
                    mask_roi = region['mask'][y:y+h, x:x+w]
                else:
                    mask_roi = None
                
                # 提取特征
                features = feature_extractor.extract_all_features(
                    roi, mask_roi,
                    use_gabor=True,
                    use_lbp=True,
                    use_geometric=True,
                    use_texture=True
                )
                features_list.append(features)
            
            X = np.array(features_list)
            print(f"特征维度: {X.shape}")
            
            # 如果有训练数据，进行训练和分类
            if args.train_data:
                print("加载训练数据...")
                # 这里需要根据实际数据格式加载
                # 示例：假设有训练数据
                # X_train, y_train = load_training_data(args.train_data)
                # classifier = DefectClassifier()
                # train_info = classifier.train(X_train, y_train)
                # predictions = classifier.predict(X)
                
                # 由于没有实际训练数据，这里使用模拟数据演示
                print("使用模拟数据进行演示...")
                # 生成模拟标签（实际应用中应该从训练数据中获取）
                n_samples = len(defect_regions)
                n_classes = 3  # 假设有3类缺陷
                
                # 模拟训练数据
                X_train = np.random.randn(100, X.shape[1])
                y_train = np.random.randint(0, n_classes, 100)
                
                # 训练分类器
                classifier = DefectClassifier(kernel='rbf', C=1.0)
                train_info = classifier.train(X_train, y_train, use_pca=False)
                
                # 预测
                predictions = classifier.predict(X)
                
                # 评价
                # 注意：这里使用模拟的真实标签，实际应用中需要提供真实标签
                y_true = np.random.randint(0, n_classes, n_samples)  # 模拟真实标签
                
                eval_class = ClassificationEvaluation()
                metrics = eval_class.calculate_metrics(y_true, predictions)
                eval_class.print_metrics(metrics)
                
                # 特征可视化
                print("\n生成特征可视化...")
                classifier.visualize_features(
                    X, predictions,
                    save_path=f"{output_dir}/feature_visualization.png",
                    n_components=2
                )
                
                # 可视化分类结果
                class_names = ['划痕', '漆点', '凹痕']  # 示例类别名称
                classification_viz = classifier.visualize_classification_result(
                    original_image, defect_regions, predictions,
                    class_names=class_names,
                    save_path=f"{output_dir}/classification_result.jpg"
                )
            else:
                print("未提供训练数据，跳过分类步骤")
                print("提示: 使用 --train_data 参数指定训练数据目录")
    
    print(f"\n处理完成！结果保存在: {output_dir}")


if __name__ == '__main__':
    main()

