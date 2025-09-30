import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np

# Load trained model
model = load_model("model/mnist_cnn.h5")
print("✅ Model loaded successfully!")

# Load MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Preprocess test data
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_test = to_categorical(y_test, 10)

# Evaluate model performance
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"📊 Test Accuracy: {acc * 100:.2f}%")

# Pick a sample image
sample_index = 0  
sample_image = x_test[sample_index].reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(sample_image)
predicted_digit = np.argmax(prediction)

print(f"🔢 Predicted Digit: {predicted_digit}")
