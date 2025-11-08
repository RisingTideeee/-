"""
系统测试脚本
验证各个模块是否正常工作
"""
import numpy as np
import cv2
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """测试导入"""
    print("测试模块导入...")
    try:
        from image_enhancement import ImageEnhancer
        from defect_segmentation import DefectSegmenter
        from feature_extraction import FeatureExtractor
        from classification import DefectClassifier
        from evaluation import ImageQualityAssessment, SegmentationEvaluation, ClassificationEvaluation
        from utils import load_image, save_image, create_output_dir
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_enhancement():
    """测试图像增强模块"""
    print("\n测试图像增强模块...")
    try:
        from image_enhancement import ImageEnhancer
        from evaluation import ImageQualityAssessment
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        enhancer = ImageEnhancer()
        enhanced = enhancer.enhance(test_image, method='clahe')
        
        iqa = ImageQualityAssessment()
        score = iqa.calculate_quality_score(enhanced, 'brisque')
        
        print(f"✓ 图像增强测试通过 (BRISQUE: {score:.4f})")
        return True
    except Exception as e:
        print(f"✗ 图像增强测试失败: {e}")
        return False


def test_segmentation():
    """测试缺陷分割模块"""
    print("\n测试缺陷分割模块...")
    try:
        from defect_segmentation import DefectSegmenter
        
        # 创建测试图像（包含模拟缺陷）
        test_image = np.ones((200, 300, 3), dtype=np.uint8) * 200
        cv2.rectangle(test_image, (50, 50), (150, 100), (50, 50, 50), -1)
        
        segmenter = DefectSegmenter()
        mask, regions = segmenter.segment(test_image, min_area=10)
        
        print(f"✓ 缺陷分割测试通过 (检测到 {len(regions)} 个区域)")
        return True
    except Exception as e:
        print(f"✗ 缺陷分割测试失败: {e}")
        return False


def test_feature_extraction():
    """测试特征提取模块"""
    print("\n测试特征提取模块...")
    try:
        from feature_extraction import FeatureExtractor
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(test_mask, (20, 20), (80, 80), 255, -1)
        
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(test_image, test_mask)
        
        print(f"✓ 特征提取测试通过 (特征维度: {features.shape})")
        return True
    except Exception as e:
        print(f"✗ 特征提取测试失败: {e}")
        return False


def test_classification():
    """测试分类模块"""
    print("\n测试分类模块...")
    try:
        from classification import DefectClassifier
        from evaluation import ClassificationEvaluation
        
        # 创建模拟数据
        X = np.random.randn(50, 20)
        y = np.random.randint(0, 3, 50)
        
        classifier = DefectClassifier()
        train_info = classifier.train(X, y, test_size=0.2)
        
        predictions = classifier.predict(train_info['X_test'])
        
        eval_class = ClassificationEvaluation()
        metrics = eval_class.calculate_metrics(train_info['y_test'], predictions)
        
        print(f"✓ 分类测试通过 (准确率: {metrics['accuracy']:.4f})")
        return True
    except Exception as e:
        print(f"✗ 分类测试失败: {e}")
        return False


def test_evaluation():
    """测试评价模块"""
    print("\n测试评价模块...")
    try:
        from evaluation import SegmentationEvaluation
        
        # 创建测试掩码
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(pred_mask, (10, 10), (50, 50), 255, -1)
        cv2.rectangle(gt_mask, (15, 15), (55, 55), 255, -1)
        
        eval_seg = SegmentationEvaluation()
        iou = eval_seg.calculate_iou(pred_mask, gt_mask)
        dice = eval_seg.calculate_dice(pred_mask, gt_mask)
        
        print(f"✓ 评价模块测试通过 (IoU: {iou:.4f}, Dice: {dice:.4f})")
        return True
    except Exception as e:
        print(f"✗ 评价模块测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 50)
    print("汽车漆面缺陷检测系统 - 系统测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_enhancement,
        test_segmentation,
        test_feature_extraction,
        test_classification,
        test_evaluation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败，请检查错误信息")
        return 1


if __name__ == '__main__':
    sys.exit(main())

