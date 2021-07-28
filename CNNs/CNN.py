from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Template model

# Preprocessing the Training set
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Convoultion
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
        activation='relu', input_shape=[64, 64, 3]))

# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Second Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten
cnn.add(tf.keras.layers.Flatten())

# Connecting before our output
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Implementing the output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train, but also evaluate on the testing set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Single prediction
test_image = image.load_img(
    'dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'a'
else:
    prediction = 'b'
print(prediction)
