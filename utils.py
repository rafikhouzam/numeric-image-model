import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess(image, label):
    print("Image shape:", image.shape)
    print("Label:", label)
    image = tf.image.resize(image, [224, 224])  # Resize images to match MobileNetV2 input
    image = tf.image.grayscale_to_rgb(image)  # Convert grayscale to RGB
    image = preprocess_input(image)
    return image, label