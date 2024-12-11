# Link for the Hugging Space based deployement - https://huggingface.co/spaces/huchen2005/_123

# BCCD Dataset Object Detection with YOLOv5

This project demonstrates how to train a YOLOv5 object detection model on the BCCD (Blood Cell Count Dataset) for detecting blood cells.

## Steps

1. **Cloning the Dataset**: The BCCD dataset is cloned from GitHub. It contains images and XML annotations.

2. **Converting Annotations**: The VOC XML annotations are converted to YOLO format, which includes class labels and normalized bounding box coordinates.

3. **Data Augmentation**: Augmentation techniques such as rotation, flipping, scaling, and blurring are applied to the images to improve model generalization.

4. **Training the Model**: A pre-trained YOLOv5s model is fine-tuned on the BCCD dataset. Training is done with a batch size of 16 for 20 epochs.

5. **Inference**: The fine-tuned model is used to make predictions on new images, detecting and labeling the cells in the images.

## Requirements

- Python 3.x
- PyTorch
- YOLOv5
- Albumentations
- OpenCV
- PIL

## Results

The model detects three types of blood cells:
- RBC (Red Blood Cells)
- WBC (White Blood Cells)
- Platelets

Bounding boxes and labels are drawn on the predicted images.

## Features

- **Upload Image**: Allows users to upload images for blood cell detection.
  
- **Object Detection**: Detects and labels Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets in the uploaded image using a fine-tuned YOLOv5 model.
  
- **Bounding Boxes & Labels**: Displayed on the image with confidence scores.
  
- **Cell Information**: Provides descriptions for each detected cell type.
  
- **Precision & Recall**: Displays precision and recall scores for each class (RBC, WBC, Platelets).
  
- **Downloadable Images**: Users can download images with bounding boxes or detailed text labels.
  
- **Metrics Summary**: Displays key performance metrics (average precision, recall, loss, etc.) across all epochs.
