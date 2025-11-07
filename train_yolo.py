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
    print("=" * 50)
    
    # 创建训练器
    trainer = YOLOTrainer(model_size=args.model_size, task=args.task)
    
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
            name=args.name
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

