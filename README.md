Brain Tumor Classifier with Gradio & TensorFlow
This project is a Brain Tumor Classification model built using TensorFlow/Keras, designed to classify brain MRI images into four categories: glioma, meningioma, pituitary, and no tumor. The model uses a Convolutional Neural Network (CNN) to process and predict the type of tumor based on the input MRI image. To make the classification accessible, a Gradio interface is included, allowing users to easily upload an image and get real-time predictions directly from their browser.

The goal of this project is to provide a simple, effective way of identifying brain tumors from MRI images. The model is trained using a dataset that contains labeled images for each of the four categories, which are processed, resized, and normalized before being fed into the network for training. The trained model can be used for classification tasks on new, unseen images.

To run this project, you first need to install the required dependencies, including TensorFlow, OpenCV, NumPy, scikit-learn, and Gradio. After installing the dependencies, you can easily upload your MRI images to Google Drive, which is where the dataset is stored. The model can be trained on Google Colab by mounting the Google Drive and then running the training script. During training, the model uses a series of Convolutional and MaxPooling layers, followed by BatchNormalization and GlobalAveragePooling, which help improve the performance of the network. The model is optimized using the Adam optimizer and is trained over multiple epochs.

Once the model is trained, a Gradio interface will allow users to upload an image to classify it. The interface shows the classification result along with the confidence score for each tumor type, helping users quickly understand the predicted result. This project can serve as a helpful tool in the early stages of tumor detection or assist in educational purposes related to machine learning applications in medical imaging.

The folder structure for the project includes the training data for each tumor type, the Python script for training the model, and the notebook for running everything in Google Colab. The model uses a simple but effective CNN architecture with three Conv2D layers, followed by MaxPooling and Dropout for regularization. Finally, a Dense layer with a Softmax activation function outputs the prediction for the given MRI image.

This project is a good starting point for anyone interested in applying machine learning to medical imaging or exploring deep learning techniques for image classification tasks. You can use this model to classify brain tumors based on MRI images, and further improvements can be made by exploring more complex architectures or using additional data augmentation techniques.

To run this project, you can clone the repository, install the required libraries, and start with training the model. You can test the model through the Gradio interface to upload images and get predictions instantly.

The project is licensed under the MIT License, so feel free to modify and use the code as per your requirements.
