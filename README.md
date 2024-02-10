# Project--Potatoes-Diseases-Detection
- The main goal of this project is used to detect the stage of disease or infection in the potato leaf. 
- Data downloaded from: https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset?select=Potato]

# Description
- There are **2152 images belonging to 3 classes - Healthy, Early blight, Late blight.**
- Preprocessing the data using TensorFlow.Dataset by creating training, validation & testing sets.
- Also augment the data for better prediction.
- I have created a **CNN model with 6 Conv2D layers (one 32 units & 5 64 units with all 3x3 kernel with activation function RELU), 6 MaxPooling layers of poolsize 2**.
- **Flattening the output before passing it into Dense layer of 64 units.**
- **Final layer is a softmax layer that will output the probabilities of 3 classes.**
- Predicting the images.
- Deploying model to StreamLit.

# Steamlit Interface

![image](https://github.com/sahilkadu96/Project--Potatoes-Diseases-Detection/assets/106151994/7d00a1bc-eb08-49bb-a6fd-254f16d35182)
