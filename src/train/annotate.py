import cv2
import numpy as np
import os
from glob import glob
from tifffile import TiffFile
from matplotlib.pyplot import imsave
from tqdm import tqdm
from matplotlib import colormaps

# Global variables for rectangle drawing
drawing = False
top_left_pt = (0, 0)
bottom_right_pt = (0, 0)
rectangles = []
current_class = 0  # Default class ID
num_classes = 2  # Number of classes

def draw_rectangle(event, x, y, flags, userdata):
    """Mouse callback function to handle rectangle drawing."""
    global drawing, top_left_pt, bottom_right_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bottom_right_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        rectangles.append((top_left_pt, bottom_right_pt))

def annotate_image(image_path):
    """Annotate a single image and save annotations in YOLO format."""
    global rectangles, top_left_pt, bottom_right_pt

    rectangles = []
    print(f"Annotating: {os.path.basename(image_path)}")

    # Load and resize the image
    image = cv2.imread(image_path)
    screen_res = (1920, 1080)
    scale = min(screen_res[0] / image.shape[1], screen_res[1] / image.shape[0])
    resized_image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    clone = resized_image.copy()

    # Setup OpenCV window
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowTitle("image", f"Annotating: {os.path.basename(image_path)}")
    cv2.setMouseCallback("image", draw_rectangle)

    while True:
        display_image = clone.copy()
        
        # Draw completed rectangles
        for i, rect in enumerate(rectangles):
            color = (100, 255, 100) if i % 2 == 0 else (100, 100, 255)
            cv2.rectangle(display_image, rect[0], rect[1], color, 1)
        
        # Draw the currently active rectangle
        if drawing:
            color = (100, 255, 100) if len(rectangles) % 2 == 0 else (100, 100, 255)
            cv2.rectangle(display_image, top_left_pt, bottom_right_pt, color, 1)
        
        cv2.imshow("image", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("w") and rectangles:
            rectangles.pop()

    cv2.destroyAllWindows()
    print(f"{len(rectangles)} bead(s) annotated")

    # Save annotations in YOLO format
    img_height, img_width = clone.shape[:2]
    annotations = []
    for rect in rectangles:
        x_min, x_max = sorted([rect[0][0], rect[1][0]])
        y_min, y_max = sorted([rect[0][1], rect[1][1]])
        x_center = round((x_min + x_max) / 2.0 / img_width, 4)
        y_center = round((y_min + y_max) / 2.0 / img_height, 4)
        width = round((x_max - x_min) / img_width, 4)
        height = round((y_max - y_min) / img_height, 4)
        annotations.append(f"0 {x_center} {y_center} {width} {height}")

    label_file = image_path.replace('images', 'labels').replace('.png', '.txt')
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, 'w') as f:
        f.write('\n'.join(annotations) + '\n')

def draw_rectanglev2(event, x, y, flags, userdata):
    """Mouse callback function to handle rectangle drawing."""
    global drawing, top_left_pt, bottom_right_pt, current_class

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bottom_right_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        rectangles.append((top_left_pt, bottom_right_pt, current_class))

def annotate_imagev2(image_path):
    """Annotate a single image and save annotations in YOLO format."""
    global rectangles, top_left_pt, bottom_right_pt, current_class

    rectangles = []
    print(f"Annotating: {os.path.basename(image_path)}")

    # Load and resize the image
    image = cv2.imread(image_path)
    screen_res = (1920, 1080)
    scale = min(screen_res[0] / image.shape[1], screen_res[1] / image.shape[0])
    resized_image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    clone = resized_image.copy()

    # Setup OpenCV window
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowTitle("image", f"Annotating: {os.path.basename(image_path)}")
    cv2.setMouseCallback("image", draw_rectanglev2)

    while True:
        display_image = clone.copy()
        
        # Draw completed rectangles
        for i, rect in enumerate(rectangles):
            color = (100, 255, 100) if i % 2 == 0 else (100, 100, 255)
            cv2.rectangle(display_image, rect[0], rect[1], color, 1)
            #cv2.putText(display_image, str(rect[2]), rect[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw the currently active rectangle
        if drawing:
            color = (100, 255, 100) if current_class == 0 else (100, 100, 255)
            cv2.rectangle(display_image, top_left_pt, bottom_right_pt, color, 1)
        
        cv2.putText(display_image, f"Class: {current_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("image", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("w") and rectangles:
            rectangles.pop()
        elif key == ord("c"):
            current_class = (current_class + 1) % num_classes  # Toggle class

    cv2.destroyAllWindows()
    print(f"{len(rectangles)} annotation(s) saved")

    # Save annotations in YOLO format
    img_height, img_width = clone.shape[:2]
    annotations = []
    for rect in rectangles:
        x_min, x_max = sorted([rect[0][0], rect[1][0]])
        y_min, y_max = sorted([rect[0][1], rect[1][1]])
        x_center = round((x_min + x_max) / 2.0 / img_width, 4)
        y_center = round((y_min + y_max) / 2.0 / img_height, 4)
        width = round((x_max - x_min) / img_width, 4)
        height = round((y_max - y_min) / img_height, 4)
        annotations.append(f"{rect[2]} {x_center} {y_center} {width} {height}")

    label_file = image_path.replace('images', 'labels').replace('.png', '.txt')
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, 'w') as f:
        f.write('\n'.join(annotations) + '\n')

def create_annotations_for_dataset(dataset_dir):
    """Create annotations for all images in a dataset."""
    for image_dir in ["train", "val"]:
        image_path = os.path.join(dataset_dir, 'images', image_dir)
        label_path = os.path.join(dataset_dir, 'labels', image_dir)
        os.makedirs(label_path, exist_ok=True)
        
        for image_file in glob(os.path.join(image_path, "*.png")):
            label_file = os.path.join(label_path, os.path.basename(image_file).replace('.png', '.txt'))
            if not os.path.exists(label_file):
                annotate_imagev2(image_file)

def load_tif_to_frames(tif_path):
    """Load frames from a TIF file and normalize them to RGB format."""
    with TiffFile(tif_path) as tif:
        frames = []
        for page in tif.pages:
            normalized = np.clip((page.asarray().astype(np.float32) - (-100)) / (7000 - (-100)), 0, 1)
            colormap = colormaps['viridis']
            colored_frame = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)
            frames.append(colored_frame)
    return frames

def save_tif_frames(tif_dir, out_dir):
    """Save frames from TIF files as individual PNG images."""
    os.makedirs(out_dir, exist_ok=True)
    for tif_file in glob(os.path.join(tif_dir, '*.tif')):
        print(f"Processing: {tif_file}")
        frames = load_tif_to_frames(tif_file)
        for i, frame in enumerate(tqdm(frames)):
            out_file = os.path.join(out_dir, os.path.basename(tif_file).replace('.tif', f'_{i+1}.png'))
            imsave(out_file, frame)
    print("Processing complete!")

if __name__ == "__main__":
    video_path = r"D:\\Beads\\YOLO 20.2.25\\image_processing\\2025.01.22_RA22_4.tif"
    # save_tif_frames(video_path, f"{video_path}\\frames")
    create_annotations_for_dataset("C:\\Users\\97254\\Desktop\\Resources\\Technion\\exploratory_resaerach\\fluorovision\\data\\datasetv2")
