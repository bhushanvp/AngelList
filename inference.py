import numpy as np
import tensorflow as tf
import autokeras as ak
import cv2
from keras.models import load_model
import json

class ImageClassifier:
    def __init__(self, model_path, image_resize_width, image_resize_height):
        self.model_path = model_path
        self.model = None
        self.image_resize_width = image_resize_width
        self.image_resize_height = image_resize_height

    def load_model(self):
        self.model = load_model(self.model_path)

    

    def preprocess_image(self, image):
        # Resize the image to match the input size of the model
        image = cv2.resize(image, (self.image_resize_width, self.image_resize_height))
        # Convert the image to float32 and normalize its values
        image = image.astype('float32') / 255.0
        # Expand dimensions to create a batch of size 1
        image = np.expand_dims(image, axis=0)
        return image

    def classify_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        # Make predictions using the loaded model
        predictions = self.model.predict(preprocessed_image)
        # Convert predictions from one-hot encoding to class labels
        predicted_class = np.argmax(predictions[0])
        return predicted_class

    def run(self, image_path):
        # Load the model
        self.load_model()
        # Read the image from file
        image = cv2.imread(image_path)
        if image is not None:
            # Classify the image
            predicted_class = self.classify_image(image)
            return predicted_class
        else:
            print('Failed to load image.')
            return None

# Load the configuration from the config.json file
with open('config.json') as config_file:
    config = json.load(config_file)

# Path to the trained model file
model_path = config["model_path"]

# Path to the image you want to classify
image_path = config["image_path"]

# Create an instance of the ImageClassifier
classifier = ImageClassifier(
    model_path,
    config["image_resize_width"],
    config["image_resize_height"]
)

# Run the image classification
predicted_class = classifier.run(image_path)

if predicted_class is not None:
    print('Predicted class:', predicted_class)
else:
    print('Image classification failed.')
