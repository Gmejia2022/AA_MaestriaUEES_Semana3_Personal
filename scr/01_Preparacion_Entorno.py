"""
01 - Preparacion del Entorno
Proyecto: Segmentacion de Clientes usando Modelos No Supervisados
Maestria en IA - UEES - Semana 3
Alumno: Ingeniero Gonzalo Mejia Alcivar
"""

# === Importacion de librerias ===

# Manejo de datos
import pandas as pd
import numpy as np

# Visualizacion
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento
from sklearn.preprocessing import StandardScaler

# Clustering
from sklearn.cluster import KMeans, DBSCAN

# Reduccion de dimensionalidad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Utilidades
import os
import warnings

warnings.filterwarnings('ignore')


def verificar_entorno():
    """Verifica que todas las librerias esten instaladas correctamente."""
    librerias = {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'matplotlib': plt.matplotlib.__version__,
        'seaborn': sns.__version__,
        'scikit-learn': __import__('sklearn').__version__,
    }

    print("=" * 50)
    print("  VERIFICACION DEL ENTORNO")
    print("=" * 50)
    for lib, version in librerias.items():
        print(f"  {lib:20s} -> v{version}")
    print("=" * 50)
    print("  Todas las librerias instaladas correctamente.")
    print("=" * 50)


def verificar_estructura_proyecto():
    """Verifica que la estructura de carpetas del proyecto exista."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpetas = ['Data', 'Models', 'notebooks', 'results', 'scr']

    print("\n  ESTRUCTURA DEL PROYECTO")
    print("=" * 50)
    for carpeta in carpetas:
        ruta = os.path.join(base_path, carpeta)
        existe = os.path.exists(ruta)
        estado = "OK" if existe else "NO ENCONTRADA"
        print(f"  {carpeta:20s} -> {estado}")
    print("=" * 50)


def cargar_datos():
    """Carga el dataset y muestra informacion basica."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_datos = os.path.join(base_path, 'Data', 'Mall_Customers.csv')

    if not os.path.exists(ruta_datos):
        print(f"  ERROR: No se encontro el archivo en {ruta_datos}")
        return None

    df = pd.read_csv(ruta_datos)

    print("\n  INFORMACION DEL DATASET")
    print("=" * 50)
    print(f"  Filas:    {df.shape[0]}")
    print(f"  Columnas: {df.shape[1]}")
    print(f"  Columnas: {list(df.columns)}")
    print("=" * 50)

    print("\n  PRIMERAS 5 FILAS")
    print("-" * 50)
    print(df.head().to_string(index=False))

    print("\n  TIPOS DE DATOS")
    print("-" * 50)
    for col in df.columns:
        print(f"  {col:30s} -> {df[col].dtype}")

    print("\n  ESTADISTICAS DESCRIPTIVAS")
    print("-" * 50)
    print(df.describe().to_string())

    print("\n  VALORES NULOS")
    print("-" * 50)
    print(f"  Total valores nulos: {df.isnull().sum().sum()}")

    return df


if __name__ == '__main__':
    verificar_entorno()
    verificar_estructura_proyecto()
    df = cargar_datos()
