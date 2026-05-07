# augmentation.py
# This module will later contain real augmentation logic

from PIL import Image
from Augmentify.augment.fastSAM import run_fastsam
from Augmentify.augment.midas import run_midas_depth
from Augmentify.augment.mobileSAM import run_mobilesam
from Augmentify.augment.yolo_world_annotate import annotate
from Augmentify.augment.ContextAware_Zoom import run_automated_zoom
from Augmentify.augment.detr import run_rtdetr
from Augmentify.augment.depth_anything import run_depth_anything
from Augmentify.augment.geometric_segmentation import run_geometric_segmentation
from Augmentify.augment.Illumination import run_gamma_correction, run_clahe, run_retinex_ssr
from Augmentify.augment.pose_estimation import run_pose_estimation
from Augmentify.augment.rembg import run_rembg
from Augmentify.augment.SailencyCrop import run_saliency_crop
from Augmentify.augment.surface_normalization import run_surface_normals
from Augmentify.augment.zoe_depth import run_zoe_depth

def apply_augmentation(image: Image.Image, method: str, prompt: str) -> Image.Image:
    print(f"Applying augmentation: {method} with prompt: {prompt}")
    """
    TODO:
    - Implement real augmentations here.
    - Possibly allow parameters like rotation angle later.
    """

    if method == "FastSAM":
        return run_fastsam(image)
    
    elif method == "MobileSAM":
        return run_mobilesam(image)
    
    elif method == "MIDAS":
        return run_midas_depth(image)

    elif method == "Yolo_World_Annotate":
        return annotate(image)

    elif method == "ContextAwareZoom":
        return run_automated_zoom(image, prompt)

    elif method == "RT-DETR":
        return run_rtdetr(image)

    elif method == "DepthAnything":
        return run_depth_anything(image)

    elif method == "ZoeDepth":
        return run_zoe_depth(image)

    elif method == "GeometricSegmentation":
        return run_geometric_segmentation(image)

    elif method == "GammaCorrection":
        return run_gamma_correction(image)

    elif method == "CLAHE":
        return run_clahe(image)

    elif method == "RetinexSSR":
        return run_retinex_ssr(image)

    elif method == "PoseEstimation":
        return run_pose_estimation(image)
    
    elif method == "Rembg":
        return run_rembg(image)
    
    elif method == "SaliencyCrop":
        return run_saliency_crop(image, prompt)

    elif method == "SurfaceNormalization":
        return run_surface_normals(image)

    elif method == "none":
        return image

    else:
        return image