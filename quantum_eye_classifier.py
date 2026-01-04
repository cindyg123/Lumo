import numpy as np
import pennylane as qml
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from google.colab import files
import zipfile
import os

def upload_and_unzip(subdir): #
    print(f"Please upload your zipped dataset of {subdir} eyes.") #
    uploaded = files.upload() #
    for filename in uploaded.keys(): #
        with zipfile.ZipFile(filename, 'r') as zip_ref: #
            zip_ref.extractall(f"dataset/{subdir}") #

upload_and_unzip("closed") #
upload_and_unzip("open") #
dataset_dir = "dataset"  # directory

def add_noise(images, noise_factor=0.2):
    noisy_images = images + noise_factor * tf.random.normal(shape=images.shape)
    noisy_images = tf.clip_by_value(noisy_images, 0.0, 1.0)
    return noisy_images

#  training 80 and testing 20
train_dataset = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(28, 28),
    batch_size=32
)
test_dataset = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(28, 28),
    batch_size=32
)
plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

input_dim = 27
wires = qml.wires.Wires(['0', '1', '2'])
wire_map = {'0': 0, '1': 1, '2': 2}
n_layers = 2
num_qubits = 3
weight_shapes = {"weights": (n_layers, num_qubits)}
mapping_wires = wires.map(wire_map)
dev = qml.device("default.qubit", wires=wires)
@qml.qnode(dev, interface="tf")
def qnode(inputs, weights):
    qml.AngleEmbedding(features=inputs, wires=range(num_qubits), rotation='Z')
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=10)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# predicts test
for images, labels in test_dataset.take(1):
    predictions = model.predict(images)
    print(predictions)
def add_noise(images, noise_factor=0.2):
    noisy_images = images + noise_factor * tf.random.normal(shape=images.shape)
    noisy_images = tf.clip_by_value(noisy_images, 0.0, 1.0)
    return noisy_images