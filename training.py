import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from utils import preprocess

#load & preprocess data
mnist_data = tfds.load("mnist", as_supervised=True)
mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]

mnist_train = mnist_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
mnist_test = mnist_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Create a simpler model based on MobileNetV2 for MNIST
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')  # Load MobileNetV2 without the top layer
base_model.trainable = False  # Freeze the base model

# Create a new top layer for the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#to make sure training only occurs when this program is run
if __name__ == '__main__':
    # Train the model
    model.fit(mnist_train, epochs=10, validation_data=mnist_test)
    # Evaluate the model
    loss, accuracy = model.evaluate(mnist_test)
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    model.save('my_mnist_model')
