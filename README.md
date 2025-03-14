# Plant Disease Classification

A deep learning model for classifying plant diseases using VGG16 architecture with custom modifications.

## 📋 Overview

This project implements a CNN-based approach for classifying 38 different plant diseases using transfer learning with VGG16. The model achieves high accuracy through enhanced data augmentation, custom architecture modifications, and optimized training techniques.

## 🌱 Dataset

The project uses the Plant Village Dataset, which contains images of healthy and diseased plant leaves across multiple plant species including:
- Apple
- Corn
- Grape
- Potato
- Tomato
- And many more

Each image is classified into one of 38 disease categories or as healthy.

## 🧠 Model Architecture

- **Base**: Pre-trained VGG16 model
- **Customizations**:
  - Additional fully connected layers (4096 → 2048 → 1024 → 38)
  - Dropout layers (0.5) for regularization
  - Xavier initialization for better convergence
  - Fine-tuning of later convolutional layers
  - Batch normalization for improved training stability

## 🔍 Key Features

- **Enhanced Data Augmentation**: Horizontal/vertical flips, rotation, color jittering
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rate
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Saves the best model based on validation accuracy
- **Optimized Training**: NAdam optimizer with custom parameters

## 📊 Results

- **Validation Accuracy**: 99.89%
- **Test Accuracy**: 99.78%
- **Training Time**: Approximately 5 hours on a Tesla T4 GPU (40 epochs)

## 🚀 Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

Run the Jupyter notebook `notebooks/plant_disease_classification.ipynb` or use the script:

```bash
python scripts/train.py
```

### Prediction

```bash
python scripts/predict.py --image path/to/plant/image.jpg
```

Example output:
```
Image: tomato_leaf.jpg
Prediction: Tomato_Late_blight
Confidence: 99.8%
```


## 🔄 Model Loading

To load the saved model for inference:

```python
# Load the complete model
model = torch.load('models/plant_disease_model_complete.pth')
model.eval()

# OR load from checkpoint
checkpoint = torch.load('models/plant_disease_model_checkpoint.pth')
model = Plant_Disease_VGG16()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 📈 Performance Visualization

The model's performance can be visualized using the included functions:

```python
plot_accuracies(history)
plot_losses(history)
plot_confusion_matrix(model, test_loader, classes)
```

## 🛠️ Future Improvements

- Implement Grad-CAM visualization for model interpretability
- Explore additional architectures (ResNet, EfficientNet)
- Develop a web application/mobile app for real-time disease detection
- Add support for more plant species and diseases

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Plant Village Dataset]([https://www.kaggle.com/datasets/snikhilrao/crop-disease-detection-dataset])
- [PyTorch](https://pytorch.org/)
- [Kaggle](https://www.kaggle.com/) for providing the computational resources
