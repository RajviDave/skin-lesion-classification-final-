<!-- # Skin Lesion Classification

This project implements a Convolutional Neural Network (CNN) using PyTorch for classifying skin lesion images into three categories: normal, abnormal, and melanoma.

## Project Structure
- `skin_lesion_classifier_pytorch.py`: Main Python script containing the CNN implementation
- `requirements.txt`: List of Python dependencies
- `PH2_dataset.xlsx`: Dataset annotations
- `dataset/`: Directory containing the image dataset

## Features
- Custom Dataset class for handling skin lesion images
- CNN architecture with 4 convolutional blocks
- Data augmentation and normalization
- Training and validation pipeline
- Model performance visualization
- Model checkpointing

## Requirements
```
pip install -r requirements.txt
```

## Usage
1. Extract the dataset from `Final Dataset.zip`
2. Ensure the dataset annotations are properly set in `PH2_dataset.xlsx`
3. Run the training script:
```python
python skin_lesion_classifier_pytorch.py
```

## Model Architecture
The CNN model consists of:
- 4 convolutional blocks with batch normalization and ReLU activation
- Max pooling layers
- Dropout for regularization
- Fully connected layers
- Softmax output for 3-class classification

## Author
Rajvi Dave  -->