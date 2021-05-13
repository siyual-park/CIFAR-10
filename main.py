import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


def create_model(num_classes: int) -> tf.keras.models.Sequential:
    model: tf.keras.models.Sequential = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-06),
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    return model


def show_history(history) -> None:
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_val = x_train[40000:]
    y_val = y_train[40000:]
    x_train = x_train[:40000]
    y_train = y_train[:40000]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
    x_train = x_train / 255.0
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 3)
    x_val = x_val / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
    x_test = x_test / 255.0

    y_train = y_train.flatten()
    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_val = y_val.flatten()
    y_val = tf.one_hot(y_val.astype(np.int32), depth=10)
    y_test = y_test.flatten()
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

    batch_size = 32
    num_classes = 10
    epochs = 2

    model = create_model(num_classes)

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val)
    )

    show_history(history)




