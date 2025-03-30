import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define class labels
CLASS_LABELS = ['ABNORMAL', 'NORMAL', 'PNEUMONIA']

# CLAHE Preprocessing Function
def clahe_preprocessing(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read the image file '{image_path}'. Check file path and integrity.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = [clahe.apply(image[:, :, i]) for i in range(3)]
    clahe_image = np.stack(channels, axis=-1)
    clahe_image = cv2.resize(clahe_image, (224, 224))  # Resize to model input size
    clahe_image = clahe_image.astype(np.float32) / 255.0  # Normalize
    clahe_image = np.expand_dims(clahe_image, axis=0)  # Add batch dimension
    return clahe_image

# Define custom layers
class PrimaryCaps(tf.keras.layers.Layer):
    def __init__(self, num_capsules, dim_capsules, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(filters=self.num_capsules * self.dim_capsules,
                                           kernel_size=3,
                                           strides=1,
                                           padding='same')
        super(PrimaryCaps, self).build(input_shape)

    def call(self, inputs):
        output = self.conv(inputs)
        output = tf.reshape(output, (-1, output.shape[1] * output.shape[2] * self.num_capsules, self.dim_capsules))
        return self.squash(output)

    def squash(self, s):
        s_squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return scale * s

    def get_config(self):
        config = super(PrimaryCaps, self).get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
        })
        return config

class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules, dim_capsules, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings

    def build(self, input_shape):
        self.W = self.add_weight(shape=[input_shape[-1], self.num_capsules * self.dim_capsules],
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        inputs_hat = tf.einsum('bij,jk->bik', inputs, self.W)
        inputs_hat = tf.reshape(inputs_hat, (-1, inputs.shape[1], self.num_capsules, self.dim_capsules))
        b = tf.zeros_like(inputs_hat[:, :, :, 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            outputs = self.squash(tf.reduce_sum(c[:, :, :, tf.newaxis] * inputs_hat, axis=1))
            if i < self.routings - 1:
                b += tf.reduce_sum(inputs_hat * outputs[:, tf.newaxis, :, :], axis=-1)
        return outputs

    def squash(self, s):
        s_squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return scale * s

    def get_config(self):
        config = super(CapsuleLayer, self).get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "routings": self.routings,
        })
        return config

# Load model with custom objects
model_path = "fold_Fold_3_complete_model.h5"
custom_objects = {'PrimaryCaps': PrimaryCaps, 'CapsuleLayer': CapsuleLayer}
model = load_model(model_path, custom_objects=custom_objects)

# Function to compute highest contribution factor based on predefined scaling
def compute_highest_contribution_factor(cnn_probabilities):
    max_prob_index = np.argmax(cnn_probabilities)
    max_prob = cnn_probabilities[max_prob_index]

    # Define scaling based on class
    if CLASS_LABELS[max_prob_index] == "NORMAL":
        contribution_factor = 0.25 * max_prob  # 25% of probability for Normal
    elif CLASS_LABELS[max_prob_index] == "ABNORMAL":
        contribution_factor = 0.50 * max_prob  # 50% of probability for Abnormal
    else:  # PNEUMONIA
        contribution_factor = 0.75 * max_prob  # 75% of probability for Pneumonia

    return {CLASS_LABELS[max_prob_index]: contribution_factor}

# Function to predict confidence scores
def predict_confidence(image_path):
    processed_image = clahe_preprocessing(image_path)
    predictions = model.predict(processed_image)[0]  # Extract confidence scores
    confidence_results = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
    contribution_factor = compute_highest_contribution_factor(predictions)
    return confidence_results, contribution_factor

# Example usage
if __name__ == "__main__":
    image_path = "test_image_2.png"  # Replace with actual image path
    confidence_scores, contribution_factor = predict_confidence(image_path)
    
    print("🔍 **CNN Confidence Scores:**")
    for label, score in confidence_scores.items():
        print(f"  - {label}: {score:.4f}")
    
    print("\n📊 **Final Contribution Factor Applied to Fuzzy Inference:**")
    print(contribution_factor)
