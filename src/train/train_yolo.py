import argparse
from ultralytics import YOLO

def train_yolo(data_path, weights_path, img_size=640,  epochs=100, batch_size=16, device='cuda', project='beads/train', name='exp'):
    """Train YOLO model."""
    # Load model
    model = YOLO(weights_path)
    # Train model
    train_results = model.train(data=data_path, epochs=epochs, batch=batch_size, imgsz=img_size, device=device, project=project, name=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data configuration file')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the pre-trained weights file')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')
    parser.add_argument('--project', type=str, default='beads/train', help='Project name for saving training results')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name for saving training results')

    args = parser.parse_args()

    train_yolo(
        data_path=args.data_path,
        weights_path=args.weights_path,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        project=args.project,
        name=args.name
    )