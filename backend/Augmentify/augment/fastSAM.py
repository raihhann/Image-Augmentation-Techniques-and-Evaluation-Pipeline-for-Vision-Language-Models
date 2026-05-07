import cv2
import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import FastSAM

# Load FastSAM once globally (avoid reloading every call)
fastsam_model = FastSAM('FastSAM-x.pt')

def run_fastsam(image, save_output=False, output_path=None):
    """
    Run FastSAM segmentation and annotate the image.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        save_output (bool): Whether to save annotated image.
        output_path (str): Path to save image if save_output=True.

    Returns:
        PIL.Image.Image: Collage of original and annotated image.
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()

    # Run FastSAM
    results = fastsam_model(image, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)[0]

    # Convert to Supervision detections
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.area > 400]  # remove tiny masks

    # Annotators
    mask_annotator = sv.MaskAnnotator(opacity=0.3)
    box_annotator = sv.BoxAnnotator(thickness=1, color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_position=sv.Position.CENTER,
        color_lookup=sv.ColorLookup.INDEX
    )

    labels = [f"{i}" for i in range(len(detections))]

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

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
    print("FastSAM augmentation completed.")
    return Image.fromarray(collage)