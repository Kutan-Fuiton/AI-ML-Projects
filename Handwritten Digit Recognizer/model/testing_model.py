# testing_model run the model on a custom image

import matplotlib
matplotlib.use('Agg')
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# confirming trained model exists
model = load_model("model/mnist_cnn.h5")
print("Model loaded successfully!")

# Load MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Preprocess test data
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_test = to_categorical(y_test, 10)

# Evaluate model performance
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Load image
img = Image.open("model/images/image2.png").convert("L")
img = img.resize((28, 28))
img_array = np.array(img).astype("float32") / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
print(f"Predicted Digit: {predicted_digit}")

plt.imshow(img_array.reshape(28, 28), cmap="gray")
plt.title(f"Prediction: {predicted_digit}")
plt.axis("off")
plt.savefig("predicted_digit.png")  # saves to file
print("Image saved as predicted_digit.png")