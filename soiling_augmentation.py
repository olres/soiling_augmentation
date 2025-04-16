import os
import json
import random
import numpy as np
import cv2
from glob import glob

def load_soiling_data(annotation_dir):
    """Load soiling images and their corresponding JSON label files."""
    soiling_data = []
    
    # Get all JSON files
    json_files = glob(os.path.join(annotation_dir, "*.json"))
    
    for json_file in json_files:
        img_file = json_file.replace('.json', '.png')
        if os.path.exists(img_file):
            with open(json_file, 'r') as f:
                annotation = json.load(f)
            
            soiling_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            if soiling_img is None:
                print(f"Warning: Could not read image {img_file}")
                continue
                
            soiling_data.append({
                'image': soiling_img,
                'annotation': annotation,
                'file_name': os.path.basename(img_file)
            })
    
    print(f"Loaded {len(soiling_data)} soiling samples")
    return soiling_data

def load_camera_images(camera_dir, num_images=5):
    """Load a limited number of camera images from the target directory."""
    image_files = glob(os.path.join(camera_dir, "*.png"))
    
    # Randomly select only the number of files we need
    if len(image_files) > num_images:
        image_files = random.sample(image_files, num_images)
    
    camera_images = []
    
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            continue
            
        camera_images.append({
            'image': img,
            'file_name': os.path.basename(img_file)
        })
    
    print(f"Loaded {len(camera_images)} camera images")
    return camera_images

def apply_soiling_to_camera(camera_img, soiling_data, output_dir):
    """Extract soiling patterns from source image and apply to camera image."""
    # Create a copy of the camera image
    result_img = camera_img['image'].copy()
    camera_height, camera_width = result_img.shape[:2]
    
    # Get soiling image and convert to RGBA
    soiling_img = soiling_data['image'].copy()
    soiling_height, soiling_width = soiling_img.shape[:2]
    
    # Create alpha channel (4th channel)
    alpha_channel = np.zeros((soiling_height, soiling_width), dtype=np.uint8)
    
    # Updated annotation with scaled polygons
    new_annotation = {
        "version": soiling_data['annotation'].get("version", "4.5.6"),
        "flags": soiling_data['annotation'].get("flags", {}),
        "shapes": []
    }
    
    # Draw each polygon from the soiling data
    for shape in soiling_data['annotation']["shapes"]:
        label = shape.get("label", "unknown")
        points = np.array(shape["points"], dtype=np.int32)
        
        # Create a new shape for the annotation with the same points
        new_shape = {
            "line_color": shape.get("line_color", None),
            "fill_color": shape.get("fill_color", None),
            "label": label,
            "points": [[float(p[0]), float(p[1])] for p in points],
            "group_id": shape.get("group_id", None),
            "shape_type": shape.get("shape_type", "polygon"),
            "flags": shape.get("flags", {})
        }
        new_annotation["shapes"].append(new_shape)
        
        # Create mask for this polygon
        mask = np.zeros((soiling_height, soiling_width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # Set alpha value based on label
        if label == "opaque":
            alpha_channel[mask > 0] = 255  # Fully opaque
        elif label == "transparent":
            alpha_channel[mask > 0] = 128  # Semi-transparent (50%)
    
    # Create a region of interest in the camera image
    roi_height = min(soiling_height, camera_height)
    roi_width = min(soiling_width, camera_width)
    
    # Get the region of interest
    roi = result_img[:roi_height, :roi_width].copy()
    
    # Get the alpha values for the ROI
    alpha = alpha_channel[:roi_height, :roi_width].astype(float) / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=2)  # Replicate for 3 channels
    
    # Extract soiling pixels for the ROI
    soiling_roi = soiling_img[:roi_height, :roi_width]
    
    # Apply alpha blending
    result_img[:roi_height, :roi_width] = (alpha * soiling_roi + (1.0 - alpha) * roi).astype(np.uint8)
    
    # Create output filenames
    base_name = os.path.splitext(camera_img['file_name'])[0]
    soiling_name = os.path.splitext(soiling_data['file_name'])[0]
    output_img_file = os.path.join(output_dir, f"{base_name}_soiled_{soiling_name}.png")
    output_json_file = os.path.join(output_dir, f"{base_name}_soiled_{soiling_name}.json")
    
    # Save the result
    cv2.imwrite(output_img_file, result_img)
    
    # Save the annotation
    with open(output_json_file, 'w') as f:
        json.dump(new_annotation, f, indent=2)
    
    return output_img_file

def main():
    # Paths
    annotation_dir = "/home/dxdxxd/projects/AnyDoor/test/labeled_ubuntu/dataset_train/train_annotated"
    camera_dir = "/home/dxdxxd/projects/AnyDoor/test/camera"
    output_dir = "/home/dxdxxd/projects/AnyDoor/test/augmented_soiling"
    
    # Number of camera images to augment
    num_images = 5
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    soiling_data = load_soiling_data(annotation_dir)
    camera_images = load_camera_images(camera_dir, num_images)
    
    if not soiling_data or not camera_images:
        print("Error: No data loaded")
        return
    
    # No need to sample again - we've already loaded exactly what we need
    # Process each selected camera image
    for camera_img in camera_images:
        # Randomly select a soiling pattern
        soiling = random.choice(soiling_data)
        
        # Apply soiling to camera image
        output_file = apply_soiling_to_camera(camera_img, soiling, output_dir)
        print(f"Created augmented image: {output_file}")

if __name__ == "__main__":
    main() 
