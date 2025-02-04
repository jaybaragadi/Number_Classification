import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

# Load the dataset and normalize it (commented out since you already have the trained model)
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build and train the model (commented out since you already have the trained model)
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=3)
# model.save('handwritten_model')

# Load the pre-trained model
model = tf.keras.models.load_model('handwritten_model')

image_number = 1
while os.path.isfile(r"C:\Users\jayas\OneDrive\Desktop\number_classification\digits\digit{image_number}.png"):
    try:
        img = cv2.imread(r"C:\Users\jayas\OneDrive\Desktop\number_classification\digits\digit{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        img = img / 255.0  # Normalize the image
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error! {e}")
    finally:
        image_number += 1

os.path.isfile(r"C:\Users\jayas\OneDrive\Desktop\number_classification\digits\digit2.png")
img = cv2.imread(r"C:\Users\jayas\OneDrive\Desktop\number_classification\digits\digit2.png")[:, :, 0]
img = np.invert(np.array([img]))
img = img / 255.0  # Normalize the image
prediction = model.predict(img)
print(f"This digit is probably a {np.argmax(prediction)}")
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()