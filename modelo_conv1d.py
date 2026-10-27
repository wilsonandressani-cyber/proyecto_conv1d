
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def crear_modelo_conv1d(n_caracteristicas=10):
    """
    Construye una red neuronal convolucional Conv1D
    para clasificacion binaria de datos tabulares.
    """
    model = Sequential([
        tf.keras.Input(shape=(n_caracteristicas, 1)),
        Conv1D(filters=32, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
