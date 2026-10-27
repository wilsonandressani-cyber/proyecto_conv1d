
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

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


if __name__ == "__main__":
    modelo = crear_modelo_conv1d(n_caracteristicas=10)
    modelo.summary()
    print("Modelo Conv1D construido correctamente.")


    # Generación de datos sintéticos
    np.random.seed(42)

    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

    
    # División de datos: 80% entrenamiento y 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Estandarización de las variables
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Adaptación de los datos para Conv1D
    X_train = X_train.reshape(-1, 10, 1)
    X_test = X_test.reshape(-1, 10, 1)

    
    # Entrenamiento de la red neuronal
    historial = modelo.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.20,
        verbose=1
    )

    # Evaluación del modelo
    perdida, exactitud = modelo.evaluate(
        X_test, y_test, verbose=0
    )

    print(f"\nExactitud en prueba: {exactitud:.4f}")

    # Predicciones
    probabilidades = modelo.predict(X_test, verbose=0)
    predicciones = (probabilidades >= 0.5).astype(int).ravel()

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, predicciones))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, predicciones))

