"""
YOLOv11模型训练脚本
用于训练汽车漆面缺陷检测模型
"""
import argparse
from pathlib import Path
from yolo_segmenter import YOLOTrainer


def main():
    parser = argparse.ArgumentParser(description='训练YOLOv11缺陷检测模型')
    parser.add_argument('--data', type=str, default="D:/Users/CY/Desktop/数字图像处理基础/dataset/dataset.yaml",
                       help='数据集配置文件路径（YOLO格式的yaml文件）')
    parser.add_argument('--model_size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小')
    parser.add_argument('--task', type=str, default='segment',
                       choices=['detect', 'segment'],
                       help='任务类型：detect(检测) 或 segment(分割)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备 (cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--project', type=str, default='runs',
                       help='项目目录')
    parser.add_argument('--name', type=str, default='defect_detection',
                       help='实验名称')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值（如果验证指标在patience个epoch内没有提升则停止训练，默认15）')
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
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not Path(args.data).exists():
        print(f"错误: 数据配置文件不存在: {args.data}")
        print("\nYOLO数据格式说明:")
        print("需要创建一个yaml文件，格式如下:")
        print("""
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # 可选

names:
  0: 划痕
  1: 漆点
  2: 凹痕
  3: 水渍
  4: 污点
        """)
        return
    
    print("=" * 50)
    print("YOLOv11 缺陷检测模型训练")
    print("=" * 50)
    print(f"数据集配置: {args.data}")
    print(f"模型大小: {args.model_size}")
    print(f"任务类型: {args.task}")
    print(f"训练轮数: {args.epochs}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"批次大小: {args.batch}")
    print(f"设备: {args.device}")
    print(f"早停耐心值: {args.patience}")
    print("=" * 50)
    
    # 创建训练器
    trainer = YOLOTrainer(model_size=args.model_size, task=args.task)
    
    # 准备训练参数
    train_kwargs = {}
    if args.lr0 is not None:
        train_kwargs['lr0'] = args.lr0
        print(f"初始学习率: {args.lr0}")
    if args.lrf is not None:
        train_kwargs['lrf'] = args.lrf
        print(f"最终学习率因子: {args.lrf}")
    if args.weight_decay is not None:
        train_kwargs['weight_decay'] = args.weight_decay
        print(f"权重衰减: {args.weight_decay}")
    if args.dropout is not None:
        train_kwargs['dropout'] = args.dropout
        print(f"Dropout: {args.dropout}")
    if args.cos_lr:
        train_kwargs['cos_lr'] = True
        print("使用余弦学习率调度")
    
    # 开始训练
    print("\n开始训练...")
    try:
        results = trainer.train(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            **train_kwargs
        )
        
        print("\n训练完成！")
        print(f"模型保存在: {args.project}/{args.name}/weights/best.pt")
        print(f"训练结果保存在: {args.project}/{args.name}/")
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

