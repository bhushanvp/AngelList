import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os
import cv2
import glob
import random
import json


class CustomDataset:
    def __init__(self):
        self.num_classes = 0  # Number of classes in your dataset
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.class_names = []

    def load_data(self, image_dir, image_resize_width, image_resize_height):
        # Find class folders
        class_folders = glob.glob(os.path.join(image_dir, "*"))
        self.num_classes = len(class_folders)

        # Load images and labels from each class folder
        X, y = [], []

        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)
            self.class_names.append(class_name)

            # Load images from folder
            class_images = glob.glob(os.path.join(class_folder, "*.jpg")) + glob.glob(os.path.join(class_folder, "*.png"))

            # Shuffle the images randomly
            random.shuffle(class_images)

            # Take 2000 images from each class
            class_images = class_images[:1500]

            for image_file in class_images:
                image = cv2.imread(image_file)
                if image is not None:
                    image = cv2.resize(image, (image_resize_width, image_resize_height))  # Resize the image
                    X.append(image)
                    y.append(len(self.class_names) - 1)  # Assign label based on class index

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Shuffle the data
        combined = list(zip(X, y))
        random.shuffle(combined)
        X, y = zip(*combined)

        # Split the data into training and testing sets
        split_index = int(0.8 * len(X))
        self.X_train = np.array(X[:split_index])
        self.y_train = np.array(y[:split_index])
        self.X_test = np.array(X[split_index:])
        self.y_test = np.array(y[split_index:])

        print('Size of y_train:', self.y_train.shape)
        print('Size of y_test:', self.y_test.shape)
        print('Number of classes:', self.num_classes)

        # Perform data augmentation on training data
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            channel_shift_range=0.2,
            brightness_range=(0.5, 1.5),
            fill_mode='nearest'
        )
        self.X_train = self.apply_data_augmentation(datagen, self.X_train)

        # Perform any necessary data preprocessing here
        # For example, normalizing the data:
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0

        self.convert_labels_to_categorical()

    def apply_data_augmentation(self, datagen, X):
        augmented_data = []
        for image in X:
            augmented_image = datagen.random_transform(image)
            augmented_data.append(augmented_image)
        augmented_data = np.array(augmented_data)
        return augmented_data

    def convert_labels_to_categorical(self):
        print('Size of y_train before conversion:', self.y_train.shape)
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
        print('Size of y_train after conversion:', self.y_train.shape)


# Load the configuration from config.json
with open("config.json") as config_file:
    config = json.load(config_file)

# Set the image directory path
image_dir = config["image_dir"]

# Load and preprocess the dataset
dataset = CustomDataset()
dataset.load_data(image_dir, config["image_resize_width"], config["image_resize_height"])

# Define the AutoKeras classifier with max_trials set in the config
clf = ak.ImageClassifier(overwrite=True, max_trials=config["max_trials"])

# Compile the classifier with accuracy metrics for training and validation
# clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=1, restore_best_weights=True
)

# Fit the classifier on the training data with the EarlyStopping callback
history = clf.fit(dataset.X_train, dataset.y_train, epochs=config["epochs"], validation_split=config["validation_split"], callbacks=[early_stopping])

# Evaluate the classifier on the test data
acc = clf.evaluate(dataset.X_test, dataset.y_test)
print('Accuracy on test data:', acc)

# Make predictions on the test data
y_pred = clf.predict(dataset.X_test)

# Convert predictions from one-hot encoding to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert ground truth from one-hot encoding to class labels
y_true_labels = np.argmax(dataset.y_test, axis=1)

# Save the best model with a unique name
model_name = "best_model.h5"
model_path = os.path.join(".", model_name)
clf.export_model().save(model_path)

print("Best model saved as:", model_name)




# Generate a classification report and confusion matrix
report = classification_report(y_true_labels, y_pred_labels)
confusion = confusion_matrix(y_true_labels, y_pred_labels)



print('Classification Report:')
print(report)

print('Confusion Matrix:')
print(confusion)


# Plot ROC curve and calculate AUC for each class
n_classes = dataset.num_classes
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(dataset.y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='Class {0} (AUC = {1:.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Save metrics in a text file
with open("metrics.txt", "w") as file:
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\n\nConfusion Matrix:\n")
    file.write(str(confusion))
    file.write("\n\nAccuracy on test data: " + str(acc))
    file.write("\n\n")

    # Write ROC AUC for each class
    file.write("ROC AUC:\n")
    for i in range(n_classes):
        file.write("Class {0}: {1:.2f}\n".format(i, roc_auc[i]))


# Display images of predicted vs actual
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(dataset.X_test[i])
    ax.set_title(f"Actual: {dataset.class_names[y_true_labels[i]]}\nPredicted: {dataset.class_names[y_pred_labels[i]]}")
    ax.axis('off')
plt.show()

# Plot train vs test results graph
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot graphs to verify overfitting
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
