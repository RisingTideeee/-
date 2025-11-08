"""分析训练结果，找出最佳epoch"""
import pandas as pd
import sys
from pathlib import Path

def analyze_training_results(results_csv_path):
    """分析训练结果"""
    df = pd.read_csv(results_csv_path)
    
    print("=" * 60)
    print("训练结果分析 - 最佳验证指标")
    print("=" * 60)
    
    # 边界框检测最佳指标
    print("\n【边界框检测 (Bounding Box)】")
    best_bbox_map50 = df.loc[df['metrics/mAP50(B)'].idxmax()]
    best_bbox_map50_95 = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
    print(f"  最佳 mAP50(B):     {best_bbox_map50['metrics/mAP50(B)']:.4f} @ Epoch {int(best_bbox_map50['epoch'])}")
    print(f"  最佳 mAP50-95(B):  {best_bbox_map50_95['metrics/mAP50-95(B)']:.4f} @ Epoch {int(best_bbox_map50_95['epoch'])}")
    
    # 分割掩码最佳指标
    print("\n【分割掩码 (Mask)】")
    best_mask_map50 = df.loc[df['metrics/mAP50(M)'].idxmax()]
    best_mask_map50_95 = df.loc[df['metrics/mAP50-95(M)'].idxmax()]
    print(f"  最佳 mAP50(M):     {best_mask_map50['metrics/mAP50(M)']:.4f} @ Epoch {int(best_mask_map50['epoch'])}")
    print(f"  最佳 mAP50-95(M):  {best_mask_map50_95['metrics/mAP50-95(M)']:.4f} @ Epoch {int(best_mask_map50_95['epoch'])}")
    
    # 验证损失最低
    print("\n【验证损失最低】")
    min_val_box = df.loc[df['val/box_loss'].idxmin()]
    min_val_seg = df.loc[df['val/seg_loss'].idxmin()]
    min_val_cls = df.loc[df['val/cls_loss'].idxmin()]
    print(f"  最低 val/box_loss: {min_val_box['val/box_loss']:.4f} @ Epoch {int(min_val_box['epoch'])}")
    print(f"  最低 val/seg_loss: {min_val_seg['val/seg_loss']:.4f} @ Epoch {int(min_val_seg['epoch'])}")
    print(f"  最低 val/cls_loss: {min_val_cls['val/cls_loss']:.4f} @ Epoch {int(min_val_cls['epoch'])}")
    
    # 综合最佳（mAP50-95(M)通常是最重要的指标）
    print("\n【综合推荐】")
    print(f"  推荐使用 Epoch {int(best_mask_map50_95['epoch'])} 的模型")
    print(f"  - mAP50(M): {best_mask_map50_95['metrics/mAP50(M)']:.4f}")
    print(f"  - mAP50-95(M): {best_mask_map50_95['metrics/mAP50-95(M)']:.4f}")
    print(f"  - val/box_loss: {best_mask_map50_95['val/box_loss']:.4f}")
    print(f"  - val/seg_loss: {best_mask_map50_95['val/seg_loss']:.4f}")
    
    # 最后5个epoch对比
    print("\n" + "=" * 60)
    print("最后5个Epoch对比（检查过拟合）")
    print("=" * 60)
    last_5 = df[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50(M)', 
                 'metrics/mAP50-95(M)', 'val/box_loss', 'val/seg_loss']].tail(5)
    print(last_5.to_string(index=False))
    
    # 检查过拟合迹象
    print("\n" + "=" * 60)
    print("过拟合分析")
    print("=" * 60)
    last_epoch = df.iloc[-1]
    best_epoch = df.loc[df['metrics/mAP50-95(M)'].idxmax()]
    
    if last_epoch['val/box_loss'] > best_epoch['val/box_loss']:
        print(f"[警告] 检测到过拟合：")
        print(f"   - 最后epoch (Epoch {int(last_epoch['epoch'])}) 的验证损失高于最佳epoch")
        print(f"   - 最佳epoch: {int(best_epoch['epoch'])}")
        print(f"   - 验证损失增加: {last_epoch['val/box_loss'] - best_epoch['val/box_loss']:.4f}")
    else:
        print("[OK] 未检测到明显过拟合")

if __name__ == '__main__':
    results_path = Path('runs/defect_detection6/results.csv')
    if not results_path.exists():
        print(f"错误: 找不到结果文件: {results_path}")
        sys.exit(1)
    
    analyze_training_results(results_path)

