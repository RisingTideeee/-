"""
YOLO模型K折交叉验证训练脚本
用于在有限数据集上获得更可靠的模型评估
"""
import argparse
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
from yolo_segmenter import YOLOTrainer
from ultralytics import YOLO


def split_dataset_kfold(data_dir: Path, k: int = 5, random_state: int = 42):
    """
    将数据集划分为K折
    
    Args:
        data_dir: 数据集根目录（包含images/train和labels/train）
        k: 折数
        random_state: 随机种子
        
    Returns:
        folds: K折划分列表，每个元素为(train_indices, val_indices)
        image_files: 图像文件列表
    """
    train_images_dir = data_dir / 'images' / 'train'
    train_labels_dir = data_dir / 'labels' / 'train'
    
    # 获取所有图像文件
    image_files = sorted(list(train_images_dir.glob('*.jpg')))
    
    if len(image_files) == 0:
        raise ValueError(f"在 {train_images_dir} 中未找到图像文件")
    
    # 创建KFold划分器
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # 生成K折划分
    folds = []
    indices = np.arange(len(image_files))
    
    for train_idx, val_idx in kf.split(indices):
        folds.append((train_idx, val_idx))
    
    return folds, image_files


def create_fold_dataset(data_dir: Path, image_files: list, train_indices: np.ndarray, 
                       val_indices: np.ndarray, fold_num: int, output_dir: Path):
    """
    为当前fold创建临时数据集
    
    Args:
        data_dir: 原始数据集目录
        image_files: 所有图像文件列表
        train_indices: 训练集索引
        val_indices: 验证集索引
        fold_num: 当前fold编号
        output_dir: 输出目录
    """
    fold_dir = output_dir / f'fold_{fold_num}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建目录结构
    (fold_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (fold_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (fold_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (fold_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 复制训练集
    train_images_dir = data_dir / 'images' / 'train'
    train_labels_dir = data_dir / 'labels' / 'train'
    
    for idx in train_indices:
        img_file = image_files[idx]
        label_file = train_labels_dir / (img_file.stem + '.txt')
        
        # 复制图像
        shutil.copy2(img_file, fold_dir / 'images' / 'train' / img_file.name)
        
        # 复制标签
        if label_file.exists():
            shutil.copy2(label_file, fold_dir / 'labels' / 'train' / label_file.name)
    
    # 复制验证集
    for idx in val_indices:
        img_file = image_files[idx]
        label_file = train_labels_dir / (img_file.stem + '.txt')
        
        # 复制图像
        shutil.copy2(img_file, fold_dir / 'images' / 'val' / img_file.name)
        
        # 复制标签
        if label_file.exists():
            shutil.copy2(label_file, fold_dir / 'labels' / 'val' / label_file.name)
    
    # 创建dataset.yaml
    original_yaml = data_dir / 'dataset.yaml'
    if original_yaml.exists():
        with open(original_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 更新路径为相对路径
        data_config['path'] = str(fold_dir.absolute())
        data_config['train'] = 'images/train'
        data_config['val'] = 'images/val'
        # 保持test集不变（如果存在）
        if 'test' in data_config:
            # test集使用原始路径
            original_test = data_config['test']
            if not Path(original_test).is_absolute():
                data_config['test'] = str((data_dir / original_test).absolute())
        
        fold_yaml = fold_dir / 'dataset.yaml'
        with open(fold_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True)
        
        return str(fold_yaml)
    else:
        raise FileNotFoundError(f"未找到数据集配置文件: {original_yaml}")


def train_kfold(data_yaml: str, k: int = 5, model_size: str = 'n', 
                task: str = 'segment', epochs: int = 100, imgsz: int = 640,
                batch: int = 16, device: str = 'cpu', project: str = 'runs', 
                name: str = 'kfold_cv', random_state: int = 42,
                enhance: bool = False, enhance_method: str = 'adaptive', **kwargs):
    """
    执行K折交叉验证训练
    
    Args:
        data_yaml: 原始数据集配置文件
        k: 折数
        model_size: 模型大小
        task: 任务类型
        epochs: 训练轮数
        imgsz: 图像尺寸
        batch: 批次大小
        device: 设备
        project: 项目目录
        name: 实验名称
        random_state: 随机种子
        enhance: 是否使用图像增强
        enhance_method: 增强方法
        **kwargs: 其他训练参数（如lr0, lrf, weight_decay, dropout, cos_lr等）和增强参数
    """
    data_dir = Path(data_yaml).parent
    output_dir = Path(project) / name
    
    print("=" * 70)
    print(" " * 20 + "K折交叉验证训练")
    print("=" * 70)
    print(f"数据集: {data_yaml}")
    print(f"折数: {k}")
    print(f"模型大小: {model_size}")
    print(f"任务类型: {task}")
    print(f"训练轮数: {epochs}")
    print(f"图像尺寸: {imgsz}")
    print(f"批次大小: {batch}")
    print(f"设备: {device}")
    if enhance:
        print(f"图像增强: 启用 ({enhance_method})")
    else:
        print("图像增强: 未启用（使用原始图像）")
    if kwargs:
        print(f"其他参数: {kwargs}")
    print("=" * 70)
    
    # 划分数据集
    print("\n正在划分数据集...")
    folds, image_files = split_dataset_kfold(data_dir, k=k, random_state=random_state)
    print(f"数据集总数: {len(image_files)} 张")
    print(f"已划分为 {k} 折")
    
    # 存储每折的结果
    fold_results = []
    fold_metrics = []
    
    # 训练每一折
    for fold_num, (train_idx, val_idx) in enumerate(folds, 1):
        print("\n" + "=" * 70)
        print(f"Fold {fold_num}/{k}")
        print("=" * 70)
        print(f"训练集: {len(train_idx)} 张 ({len(train_idx)/len(image_files)*100:.1f}%)")
        print(f"验证集: {len(val_idx)} 张 ({len(val_idx)/len(image_files)*100:.1f}%)")
        
        # 创建当前fold的数据集
        print("正在创建fold数据集...")
        fold_yaml = create_fold_dataset(
            data_dir, image_files, train_idx, val_idx, 
            fold_num, output_dir
        )
        
        # 训练当前fold
        print(f"开始训练 Fold {fold_num}...")
        trainer = YOLOTrainer(model_size=model_size, task=task)
        
        fold_name = f"{name}_fold{fold_num}"
        results = trainer.train(
            data_yaml=fold_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=fold_name,
            enhance=enhance,
            enhance_method=enhance_method,
            **kwargs
        )
        
        # 提取指标
        if hasattr(results, 'seg'):
            metrics = {
                'fold': fold_num,
                'mAP50': float(results.seg.map50),
                'mAP50_95': float(results.seg.map),
                'precision': float(results.seg.mp),
                'recall': float(results.seg.mr),
            }
        elif hasattr(results, 'box'):
            metrics = {
                'fold': fold_num,
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
            }
        else:
            metrics = {'fold': fold_num}
        
        fold_results.append(results)
        fold_metrics.append(metrics)
        
        print(f"\nFold {fold_num} 完成:")
        print(f"  mAP50: {metrics.get('mAP50', 0):.4f}")
        print(f"  mAP50-95: {metrics.get('mAP50_95', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
    
    # 计算平均指标
    print("\n" + "=" * 70)
    print("K折交叉验证结果汇总")
    print("=" * 70)
    
    if fold_metrics and 'mAP50' in fold_metrics[0]:
        avg_map50 = np.mean([m['mAP50'] for m in fold_metrics])
        avg_map50_95 = np.mean([m['mAP50_95'] for m in fold_metrics])
        avg_precision = np.mean([m['precision'] for m in fold_metrics])
        avg_recall = np.mean([m['recall'] for m in fold_metrics])
        
        std_map50 = np.std([m['mAP50'] for m in fold_metrics])
        std_map50_95 = np.std([m['mAP50_95'] for m in fold_metrics])
        std_precision = np.std([m['precision'] for m in fold_metrics])
        std_recall = np.std([m['recall'] for m in fold_metrics])
        
        print(f"\n平均指标 (均值 ± 标准差):")
        print(f"  mAP50:      {avg_map50:.4f} ± {std_map50:.4f}")
        print(f"  mAP50-95:   {avg_map50_95:.4f} ± {std_map50_95:.4f}")
        print(f"  Precision:  {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"  Recall:     {avg_recall:.4f} ± {std_recall:.4f}")
        
        print(f"\n各折详细结果:")
        for m in fold_metrics:
            print(f"  Fold {m['fold']}: mAP50={m['mAP50']:.4f}, "
                  f"mAP50-95={m['mAP50_95']:.4f}, "
                  f"P={m['precision']:.4f}, R={m['recall']:.4f}")
        
        # 找出最佳fold
        best_fold = max(fold_metrics, key=lambda x: x['mAP50'])
        print(f"\n最佳Fold: Fold {best_fold['fold']} (mAP50={best_fold['mAP50']:.4f})")
        print(f"最佳模型路径: {project}/{name}_fold{best_fold['fold']}/weights/best.pt")
    
    print("\n" + "=" * 70)
    print("K折交叉验证完成！")
    print("=" * 70)
    print(f"\n所有fold的模型保存在: {project}/")
    print(f"临时数据集保存在: {output_dir}/")
    print("\n建议:")
    print("  1. 使用最佳fold的模型作为最终模型")
    print("  2. 或者集成所有fold的模型（需要额外实现）")
    print("  3. 清理临时数据集目录以节省空间")
    
    return fold_results, fold_metrics


def main():
    parser = argparse.ArgumentParser(description='YOLO模型K折交叉验证训练')
    parser.add_argument('--data', type=str, default='dataset/dataset.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--k', type=int, default=5,
                       help='折数（默认5）')
    parser.add_argument('--model_size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小')
    parser.add_argument('--task', type=str, default='segment',
                       choices=['detect', 'segment'],
                       help='任务类型')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cpu',
                       help='训练设备 (cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--project', type=str, default='runs',
                       help='项目目录')
    parser.add_argument('--name', type=str, default='kfold_cv',
                       help='实验名称')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值（如果验证指标在patience个epoch内没有提升则停止训练）')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子（用于数据划分）')
    parser.add_argument('--lr0', type=float, default=None,
                       help='初始学习率（默认使用YOLO默认值0.01）')
    parser.add_argument('--lrf', type=float, default=None,
                       help='最终学习率因子（最终学习率=lr0*lrf，默认0.01）')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='权重衰减（默认0.0005）')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout比率（默认0.0，建议0.1-0.2防止过拟合）')
    parser.add_argument('--cos_lr', action='store_true',
                       help='使用余弦学习率调度（推荐）')
    parser.add_argument('--enhance', action='store_true',
                       help='使用自定义图像增强算法（在训练前对图像进行增强）')
    parser.add_argument('--enhance-method', type=str, default='adaptive',
                       choices=['hist_eq', 'clahe', 'contrast_stretch', 'gamma', 'adaptive'],
                       help='图像增强方法（默认：adaptive，自适应增强）')
    parser.add_argument('--clip-limit', type=float, default=2.0,
                       help='CLAHE的对比度限制（仅当enhance-method=clahe时使用）')
    parser.add_argument('--gamma', type=float, default=1.5,
                       help='伽马值（仅当enhance-method=gamma时使用）')
    
    args = parser.parse_args()
    
    # 准备训练参数
    train_kwargs = {}
    if args.lr0 is not None:
        train_kwargs['lr0'] = args.lr0
    if args.lrf is not None:
        train_kwargs['lrf'] = args.lrf
    if args.weight_decay is not None:
        train_kwargs['weight_decay'] = args.weight_decay
    if args.dropout is not None:
        train_kwargs['dropout'] = args.dropout
    if args.cos_lr:
        train_kwargs['cos_lr'] = True
    
    # 准备增强参数
    if args.enhance:
        if args.enhance_method == 'clahe':
            train_kwargs['clip_limit'] = args.clip_limit
        elif args.enhance_method == 'gamma':
            train_kwargs['gamma'] = args.gamma
    
    train_kfold(
        data_yaml=args.data,
        k=args.k,
        model_size=args.model_size,
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        random_state=args.random_state,
        enhance=args.enhance,
        enhance_method=args.enhance_method,
        **train_kwargs
    )


if __name__ == '__main__':
    main()

