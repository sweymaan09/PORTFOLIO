# MNIST Digit Recognition using Keras

# 1. Import libraries

!pip install tensorflow==2.13
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 2. Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data (make values between 0 and 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 4. Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train model
model.fit(x_train, y_train, epochs=5)

# 6. Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# 7. Predict on sample image
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Prediction: {model.predict(x_test[:1]).argmax()}")
plt.show()
