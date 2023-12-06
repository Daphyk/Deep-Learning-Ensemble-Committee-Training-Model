import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import classification_report, accuracy_score

def load_and_preprocess_data():
    # Check if the "data" folder exists, and create it if not
    if not os.path.exists('data'):
        os.makedirs('data')

        # Load and save Fashion MNIST dataset only if it doesn't exist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        np.save('data/train_images.npy', train_images)
        np.save('data/train_labels.npy', train_labels)
        np.save('data/test_images.npy', test_images)
        np.save('data/test_labels.npy', test_labels)

    else:
        # Load data from saved files
        train_images = np.load('data/train_images.npy')
        train_labels = np.load('data/train_labels.npy')
        test_images = np.load('data/test_images.npy')
        test_labels = np.load('data/test_labels.npy')

    # Preprocess the data
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels

# Load and preprocess data
train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

# Define Model 1: Simple Convolutional Neural Network
model1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Define Model 2: Long Short-Term Memory Neural Network
model2 = models.Sequential([
    layers.LSTM(64, input_shape=(28, 28), activation='relu', return_sequences=True),
    layers.LSTM(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Define Model 3: Basic Feedforward Neural Network
model3 = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile models
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model3.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Train models
model1.fit(train_images[..., np.newaxis], train_labels, epochs=5)
model2.fit(train_images[..., np.newaxis], train_labels, epochs=5)
model3.fit(train_images, train_labels, epochs=5)

# Evaluate individual models
def evaluate_model(model, test_images, test_labels):
    predictions = np.argmax(model.predict(test_images), axis=1)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, target_names=[str(i) for i in range(10)])
    return accuracy, report

# Evaluate each model on the test data
accuracy1, report1 = evaluate_model(model1, test_images[..., np.newaxis], test_labels)
accuracy2, report2 = evaluate_model(model2, test_images[..., np.newaxis], test_labels)
accuracy3, report3 = evaluate_model(model3, test_images, test_labels)

# Combine models into a committee
committee_predictions = np.mean([model1.predict(test_images[..., np.newaxis]),
                                model2.predict(test_images[..., np.newaxis]),
                                model3.predict(test_images)], axis=0)
committee_predictions = np.argmax(committee_predictions, axis=1)
committee_accuracy = accuracy_score(test_labels, committee_predictions)
committee_report = classification_report(test_labels, committee_predictions, target_names=[str(i) for i in range(10)])

# Print individual accuracies and committee accuracy
print(f'Model 1 Accuracy: {accuracy1}\n{report1}\n')
print(f'Model 2 Accuracy: {accuracy2}\n{report2}\n')
print(f'Model 3 Accuracy: {accuracy3}\n{report3}\n')
print(f'Committee Accuracy: {committee_accuracy}\n{committee_report}')
