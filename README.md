# Glioma_AI_Detector_Project

# Description: This project focuses on developing a deep-learning model to detect glioma tumors from MRI images. The model utilizes convolutional neural networks (CNN) and data augmentation techniques to improve its accuracy and generalization. The project is implemented using TensorFlow and Keras, and it runs on Google Colab for training and testing.

# -----------

# The Algorithm: The algorithm is based on a Convolutional Neural Network (CNN) architecture. Below is a brief explanation of how it works:

# Data Augmentation: The algorithm uses the ImageDataGenerator class from Keras to apply random transformations to the training images, such as rotations, shifts, and flips. This helps in making the model more robust and less prone to overfitting.

# Model Architecture: The model uses multiple convolutional layers with batch normalization and pooling layers to extract image features. This is followed by a global average pooling layer and fully connected layers to classify the images into Glioma and No Tumor categories.

# Training: The model is trained using the augmented training data, and its performance is validated on a separate validation set. The training process includes compiling the model with the Adam optimizer and binary cross-entropy loss function.

# Inference: After training, the model is used to classify new, unseen images. The model predicts the class of each image and outputs the confidence level.

# ----------

# Running the project:
 #  1) Download the Data.zip and upload it to your Google Drive 
#   2) Download the Glioma_AI_detector.ipynb and upload it to your Google Drive
#   3) Download the Train_Classification_Model_for_Glioma and upload it to your Google Drive
#   4) Upload the zip to the Glioma_AI_detector.ipynb and unzip it by running the second cell. After unzipping, comment the entire second cell and run all the cells
#   5) Open the provided Jupyter notebook (Train_Classification_Model_for_Glioma.ipynb) in Google Colab or your local environment and run all the cells until the 9th cell. 
#   6) After running the 9th cell, you should get an onnx file like 'resnet18.onnx'
#   7) In your Nvidia Jetson, add a folder called Glioma_AI_detector
#   8) Unzip the Data.zip and drag and drop it inside the folder
#   9) Download, and drag and drop it inside the folder for 'resnet18.onnx'
#   10) Download, and drag and drop it inside the folder for 'labels.txt'
#   11) Download, and drag and drop it inside the folder for 'run_onnx_inference.py'

# ---------

#   1) After setting up everything to run the project, cd into Glioma_AI_detector
#   2) This is an example of the command that should look like, make sure to change the output filename every time you run it, in this case, “result”, and add the relative path of the image you want to test from the dataset and replace the existing path in this demo code

# imagenet --model=resnet18.onnx --input_blob=input_0 --output_blob=output_0 --headless --labels=labels.txt Data/Testing/glioma/Te-gl_0010.jpg result.jpg 
