# Description: This script is used to track objects in a video using YOLO model.
from ultralytics import YOLO
from src.utils.common_utils import yolo_results_to_dataframe


def get_yolo_trakcing_results(weights,tracker, source, project,name, save_video=False):
    """Track objects in a video using YOLO model."""
    # Load model
    model = YOLO(weights)
    results = model.track(source=source, save=save_video, project=project, name=name, verbose=True, tracker=tracker)
    results_path = project + f'/{name}/results.csv'
    df = yolo_results_to_dataframe(results)
    df.to_csv(results_path, index=False)
    return df


if __name__ == '__main__':
    weights = r'C:\Users\97254\Desktop\git\FluoroVision\src\weights\yolo11n_bead_det_best_301224.pt'
    source = r'C:\Users\97254\Desktop\Resources\Technion\exploratory_resaerach\data\dataset\val_video.mp4'
    project = r'C:\Users\97254\Desktop\git\FluoroVision\runs\detect'
    tracker = r'C:\Users\97254\Desktop\git\FluoroVision\src\config\bytetrack.yaml'
    name = 'bytetrack'
    results = get_yolo_trakcing_results(
        weights, tracker, source, project, name, save_video=True)
