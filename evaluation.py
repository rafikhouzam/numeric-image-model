import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from utils import preprocess

# Load the model
model = tf.keras.models.load_model('my_mnist_model')

# Assuming you are using tfds for the MNIST dataset
test_data = tfds.load('mnist', split='test', as_supervised=True)
test_data = test_data.map(preprocess).batch(32)

# Making predictions
test_images, test_labels = next(iter(test_data)) 
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

test_labels = np.array([label.numpy() for label in test_labels])

# Evaluate the model
print("Evaluation results:")
print(classification_report(test_labels, predicted_classes))

# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(test_labels, predicted_classes)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
