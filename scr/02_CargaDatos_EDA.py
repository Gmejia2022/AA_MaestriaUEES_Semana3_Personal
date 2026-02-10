"""
02 - Carga de Datos y Analisis Exploratorio (EDA)
Proyecto: Segmentacion de Clientes usando Modelos No Supervisados
Maestria en IA - UEES - Semana 3
Alumno: Ingeniero Gonzalo Mejia Alcivar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Rutas del proyecto ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)


def cargar_datos():
    """Carga el dataset Mall_Customers.csv."""
    ruta = os.path.join(DATA_DIR, 'Mall_Customers.csv')
    df = pd.read_csv(ruta)
    print("=" * 60)
    print("  DATASET CARGADO EXITOSAMENTE")
    print("=" * 60)
    print(f"  Filas:    {df.shape[0]}")
    print(f"  Columnas: {df.shape[1]}")
    print(f"  Nombres:  {list(df.columns)}")
    print("=" * 60)
    return df


def analisis_exploratorio(df):
    """Realiza el analisis exploratorio y guarda resultados en results/."""
    reporte = []
    reporte.append("=" * 60)
    reporte.append("  ANALISIS EXPLORATORIO DE DATOS (EDA)")
    reporte.append("=" * 60)

    # --- Primeras filas ---
    reporte.append("\n--- PRIMERAS 5 FILAS ---")
    reporte.append(df.head().to_string())

    # --- Tipos de datos ---
    reporte.append("\n--- TIPOS DE DATOS ---")
    for col in df.columns:
        reporte.append(f"  {col:30s} -> {df[col].dtype}")

    # --- Estadisticas descriptivas ---
    reporte.append("\n--- ESTADISTICAS DESCRIPTIVAS ---")
    reporte.append(df.describe().to_string())

    # --- Valores nulos ---
    reporte.append("\n--- VALORES NULOS ---")
    nulos = df.isnull().sum()
    reporte.append(nulos.to_string())
    reporte.append(f"\n  Total valores nulos: {nulos.sum()}")

    # --- Distribucion por genero ---
    reporte.append("\n--- DISTRIBUCION POR GENERO ---")
    reporte.append(df['Genre'].value_counts().to_string())

    # --- Correlacion entre variables numericas ---
    cols_numericas = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    reporte.append("\n--- MATRIZ DE CORRELACION ---")
    reporte.append(df[cols_numericas].corr().to_string())

    reporte.append("\n" + "=" * 60)

    # Imprimir en consola
    texto_reporte = "\n".join(reporte)
    print(texto_reporte)

    # Guardar reporte en archivo
    ruta_reporte = os.path.join(RESULTS_DIR, '02_EDA_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(texto_reporte)
    print(f"\n  Reporte guardado en: {ruta_reporte}")

    return df


def generar_visualizaciones(df):
    """Genera y guarda las visualizaciones del EDA."""
    cols_numericas = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    # --- 1. Pairplot ---
    print("\n  Generando Pairplot...")
    g = sns.pairplot(df[cols_numericas], diag_kind='hist')
    g.figure.suptitle('Pairplot - Variables Numericas', y=1.02, fontsize=14)
    ruta_pairplot = os.path.join(RESULTS_DIR, '02_pairplot.png')
    g.savefig(ruta_pairplot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta_pairplot}")

    # --- 2. Histogramas individuales ---
    print("  Generando Histogramas...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(cols_numericas):
        axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='black')
        axes[i].set_title(f'Distribucion de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
    fig.suptitle('Histogramas de Variables Numericas', fontsize=14)
    plt.tight_layout()
    ruta_hist = os.path.join(RESULTS_DIR, '02_histogramas.png')
    fig.savefig(ruta_hist, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta_hist}")

    # --- 3. Boxplots ---
    print("  Generando Boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(cols_numericas):
        axes[i].boxplot(df[col], vert=True)
        axes[i].set_title(f'Boxplot de {col}')
        axes[i].set_ylabel(col)
    fig.suptitle('Boxplots de Variables Numericas', fontsize=14)
    plt.tight_layout()
    ruta_box = os.path.join(RESULTS_DIR, '02_boxplots.png')
    fig.savefig(ruta_box, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta_box}")

    # --- 4. Matriz de correlacion ---
    print("  Generando Matriz de Correlacion...")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[cols_numericas].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title('Matriz de Correlacion', fontsize=14)
    plt.tight_layout()
    ruta_corr = os.path.join(RESULTS_DIR, '02_correlacion.png')
    fig.savefig(ruta_corr, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta_corr}")

    # --- 5. Distribucion por genero ---
    print("  Generando Distribucion por Genero...")
    fig, ax = plt.subplots(figsize=(6, 4))
    df['Genre'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'], ax=ax)
    ax.set_title('Distribucion por Genero', fontsize=14)
    ax.set_xlabel('Genero')
    ax.set_ylabel('Cantidad')
    plt.tight_layout()
    ruta_genero = os.path.join(RESULTS_DIR, '02_distribucion_genero.png')
    fig.savefig(ruta_genero, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta_genero}")

    print("\n  Todas las visualizaciones generadas exitosamente.")


if __name__ == '__main__':
    df = cargar_datos()
    analisis_exploratorio(df)
    generar_visualizaciones(df)
