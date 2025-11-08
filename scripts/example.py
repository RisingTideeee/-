"""
示例脚本：演示如何使用汽车漆面缺陷检测系统
"""
import cv2
import numpy as np
from image_enhancement import ImageEnhancer
from defect_segmentation import DefectSegmenter
from feature_extraction import FeatureExtractor
from classification import DefectClassifier
from evaluation import ImageQualityAssessment, SegmentationEvaluation, ClassificationEvaluation
from utils import load_image, save_image, create_output_dir, visualize_results


def example_enhancement():
    """示例1：图像增强"""
    print("=== 示例1：图像增强 ===")
    
    # 加载图像（请替换为实际图像路径）
    # image = load_image("path/to/your/image.jpg")
    
    # 创建模拟图像用于演示
    image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    enhancer = ImageEnhancer()
    iqa = ImageQualityAssessment()
    
    # 计算原始图像质量
    original_brisque = iqa.calculate_quality_score(image, 'brisque')
    print(f"原始图像 BRISQUE分数: {original_brisque:.4f}")
    
    # 应用CLAHE增强
    enhanced = enhancer.enhance(image, method='clahe', clip_limit=2.0)
    enhanced_brisque = iqa.calculate_quality_score(enhanced, 'brisque')
    print(f"增强后 BRISQUE分数: {enhanced_brisque:.4f}")
    
    print("图像增强完成！\n")


def example_segmentation():
    """示例2：缺陷分割"""
    print("=== 示例2：缺陷分割 ===")
    
    # 创建模拟图像（包含模拟缺陷）
    image = np.ones((400, 600, 3), dtype=np.uint8) * 200
    # 添加一些模拟缺陷（暗色区域）
    cv2.rectangle(image, (100, 100), (200, 150), (50, 50, 50), -1)
    cv2.ellipse(image, (400, 250), (60, 40), 45, 0, 360, (60, 60, 60), -1)
    
    segmenter = DefectSegmenter()
    
    # 执行分割
    mask, regions = segmenter.segment(
        image,
        window_size=15,
        threshold=0.15,
        min_area=50
    )
    
    print(f"检测到 {len(regions)} 个缺陷区域")
    for i, region in enumerate(regions):
        print(f"  区域 {i+1}: 面积={region['area']}, 边界框={region['bbox']}")
    
    # 可视化
    visualization = segmenter.visualize_segmentation(image, mask, regions)
    
    print("缺陷分割完成！\n")


def example_classification():
    """示例3：特征提取与分类"""
    print("=== 示例3：特征提取与分类 ===")
    
    # 创建模拟数据
    n_samples = 50
    n_features = 100
    
    # 生成模拟特征
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # 3个类别
    
    # 训练分类器
    classifier = DefectClassifier(kernel='rbf', C=1.0)
    train_info = classifier.train(X, y, test_size=0.2)
    
    # 评价
    eval_class = ClassificationEvaluation()
    metrics = eval_class.calculate_metrics(train_info['y_test'], train_info['y_test_pred'])
    eval_class.print_metrics(metrics)
    
    # 特征可视化
    classifier.visualize_features(X, y, n_components=2)
    
    print("特征提取与分类完成！\n")


def example_full_pipeline():
    """示例4：完整流程"""
    print("=== 示例4：完整检测流程 ===")
    
    # 创建模拟图像
    image = np.ones((400, 600, 3), dtype=np.uint8) * 180
    # 添加模拟缺陷
    cv2.rectangle(image, (150, 120), (250, 180), (40, 40, 40), -1)
    cv2.circle(image, (450, 300), 50, (50, 50, 50), -1)
    
    output_dir = create_output_dir("example_output")
    
    # 1. 图像增强
    print("步骤1: 图像增强...")
    enhancer = ImageEnhancer()
    enhanced = enhancer.adaptive_enhancement(image)
    iqa = ImageQualityAssessment()
    score = iqa.calculate_quality_score(enhanced, 'brisque')
    print(f"  增强后质量分数: {score:.4f}")
    
    # 2. 缺陷分割
    print("步骤2: 缺陷分割...")
    segmenter = DefectSegmenter()
    mask, regions = segmenter.segment(enhanced, min_area=50)
    print(f"  检测到 {len(regions)} 个缺陷区域")
    
    # 3. 特征提取
    print("步骤3: 特征提取...")
    feature_extractor = FeatureExtractor()
    if len(regions) > 0:
        features_list = []
        for region in regions:
            x, y, w, h = region['bbox']
            roi = enhanced[y:y+h, x:x+w]
            mask_roi = region['mask'][y:y+h, x:x+w]
            features = feature_extractor.extract_all_features(roi, mask_roi)
            features_list.append(features)
        
        X = np.array(features_list)
        print(f"  特征维度: {X.shape}")
        
        # 4. 分类（使用模拟训练数据）
        print("步骤4: 分类...")
        X_train = np.random.randn(30, X.shape[1])
        y_train = np.random.randint(0, 3, 30)
        
        classifier = DefectClassifier()
        classifier.train(X_train, y_train)
        predictions = classifier.predict(X)
        
        print(f"  预测结果: {predictions}")
    
    print("\n完整流程执行完成！")
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    print("汽车漆面缺陷检测系统 - 示例演示\n")
    print("=" * 50)
    
    # 运行各个示例
    example_enhancement()
    example_segmentation()
    example_classification()
    example_full_pipeline()
    
    print("=" * 50)
    print("所有示例运行完成！")

