
import numpy as np

def generar_datos(n_muestras=1000, n_caracteristicas=10):
    np.random.seed(42)
    datos = np.random.randn(n_muestras, n_caracteristicas)
    return datos
