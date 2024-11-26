import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from utils import preprocess

# Load & preprocess data
mnist_data = tfds.load("mnist", as_supervised=True)
mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]

mnist_train = mnist_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
mnist_test = mnist_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Create a model based on MobileNetV2 for MNIST
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Create a new top layer for the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
csv_logger = CSVLogger('training_log.csv')
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Train the model
model.fit(mnist_train, epochs=10, validation_data=mnist_test, callbacks=[checkpoint, csv_logger, tensorboard_callback])

# Evaluate the model
loss, accuracy = model.evaluate(mnist_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Save the final model
model.save('my_mnist_model2')
