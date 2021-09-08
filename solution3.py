"""
Computer Vision related project

Using TFDS build a CNN that will classify images from the Beans dataset.

The output layer is given for you according to the dataset.
Images are RGB and shape should be 124x124.

Dataset info: https://github.com/AI-Lab-Makerere/ibean/
"""

import tensorflow as tf
import tensorflow_datasets as tfds

DATASET_NAME = 'beans'
(train_data, test_data), info = tfds.load(name=DATASET_NAME, split=[tfds.Split.TRAIN, tfds.Split.TEST], with_info=True)


def preprocess(data):

    features = data['image']
    label = data['label']

    features = tf.cast(features, tf.float32)
    label = tf.cast(label, tf.float32)

    features /= 255.0
    features = tf.image.resize(features, (124, 124))

    return features, label


def solution_model():
    train_dataset = train_data.map(preprocess).batch(32)
    test_dataset = test_data.map(preprocess).batch(32)

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(124, 124, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.fit(train_dataset, epochs=20, validation_data=test_dataset)

    return model


# save your model in the .h5 format.
if __name__ == "__main__":
    model = solution_model()
    model.save("beans.h5")
