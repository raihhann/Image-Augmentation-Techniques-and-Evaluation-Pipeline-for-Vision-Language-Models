import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load Pose model once globally
pose_model = YOLO('yolov8n-pose.pt')

def run_pose_estimation(image, save_output=False, output_path=None):
    """
    Run YOLOv8 pose estimation and create a vertically stacked collage.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        save_output (bool): Whether to save the annotated skeleton image.
        output_path (str): Path to save image if save_output=True.

    Returns:
        PIL.Image.Image: Collage of original and pose-annotated image.
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()

    # Run YOLOv8 inference
    # verbose=False keeps the console clean during batch processing
    results = pose_model(original_image, verbose=False)[0]
    
    # Generate the annotated image (skeleton plot)
    # boxes=False keeps the focus on keypoints/kinematics
    annotated_image = results.plot(boxes=False)

    # Save if requested
    if save_output and output_path:
        cv2.imwrite(output_path, annotated_image)

# -------- COLLAGE PART --------

    h, w = original_image.shape[:2]

    # Resize both images to half height while keeping width
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(annotated_image, (w, h // 2))

    # Vertical collage with same final resolution as original
    collage = np.vstack((orig_resized, annot_resized))

    collage = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    print("Pose estimation augmentation completed.")
    return Image.fromarray(collage)