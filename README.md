
# Modelo de Red Neuronal Convolucional Conv1D

## Descripción

Este proyecto presenta la construcción de una red neuronal convolucional unidimensional (Conv1D) utilizando Python y TensorFlow/Keras.

El modelo está diseñado para realizar tareas de clasificación binaria a partir de datos tabulares con diez características de entrada.

## Arquitectura del modelo

La red neuronal incorpora las siguientes capas:

- Input: recepción de las características.
- Conv1D: extracción de patrones mediante filtros convolucionales.
- MaxPooling1D: reducción de la dimensión de los datos.
- Dropout: regularización para reducir el sobreajuste.
- Flatten: transformación de los resultados a un vector.
- Dense: procesamiento de las características extraídas.
- Dense de salida: clasificación binaria.

## Herramientas utilizadas

- Python 3.12
- TensorFlow y Keras
- Visual Studio Code
- Git y GitHub

## Estado del proyecto

Se ha implementado la arquitectura inicial de la red neuronal Conv1D y se ha verificado su construcción mediante el resumen del modelo.

La arquitectura contiene 10.561 parámetros entrenables.

El entrenamiento y la evaluación predictiva quedan pendientes.

## Ejecución

Instalar las dependencias necesarias:

```bash
python -m pip install tensorflow numpy pandas scikit-learn matplotlib
```

Ejecutar el archivo principal:

```bash
python modelo_conv1d.py
```

## Objetivo académico

Aplicar los fundamentos de aprendizaje profundo mediante la implementación de una red neuronal convolucional para clasificación binaria.
