"""
03 - Preprocesamiento
Proyecto: Segmentacion de Clientes usando Modelos No Supervisados
Maestria en IA - UEES - Semana 3
Alumno: Ingeniero Gonzalo Mejia Alcivar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# === Rutas del proyecto ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Variables numericas a escalar
COLS_NUMERICAS = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']


def cargar_datos():
    """Carga el dataset Mall_Customers.csv."""
    ruta = os.path.join(DATA_DIR, 'Mall_Customers.csv')
    df = pd.read_csv(ruta)
    print("  Dataset cargado exitosamente.")
    return df


def aplicar_standard_scaler(df):
    """Aplica StandardScaler a las variables numericas."""
    X = df[COLS_NUMERICAS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=COLS_NUMERICAS)

    return X, df_scaled, scaler


def generar_reporte(X_original, df_scaled, scaler):
    """Genera reporte comparativo antes y despues del escalado."""
    lineas = []
    lineas.append("=" * 60)
    lineas.append("  ETAPA 3: PREPROCESAMIENTO - StandardScaler")
    lineas.append("=" * 60)

    lineas.append("\n--- ESTADISTICAS ANTES DEL ESCALADO ---")
    lineas.append(X_original.describe().to_string())

    lineas.append("\n--- ESTADISTICAS DESPUES DEL ESCALADO ---")
    lineas.append(df_scaled.describe().to_string())

    lineas.append("\n--- PARAMETROS DEL SCALER ---")
    for i, col in enumerate(COLS_NUMERICAS):
        lineas.append(f"  {col}:")
        lineas.append(f"    Media (mean):           {scaler.mean_[i]:.4f}")
        lineas.append(f"    Desviacion estandar:    {scaler.scale_[i]:.4f}")

    lineas.append("\n--- VERIFICACION ---")
    lineas.append(f"  Media despues del escalado (esperada ~0):")
    for col in COLS_NUMERICAS:
        lineas.append(f"    {col}: {df_scaled[col].mean():.6f}")
    lineas.append(f"  Desv. estandar despues del escalado (esperada ~1):")
    for col in COLS_NUMERICAS:
        lineas.append(f"    {col}: {df_scaled[col].std():.6f}")

    lineas.append("\n" + "=" * 60)

    texto = "\n".join(lineas)
    print(texto)

    ruta_reporte = os.path.join(RESULTS_DIR, '03_preprocesamiento_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(f"\n  Reporte guardado en: {ruta_reporte}")


def generar_visualizaciones(X_original, df_scaled):
    """Genera graficos comparativos antes y despues del escalado."""

    # --- 1. Comparacion de distribuciones (antes vs despues) ---
    print("\n  Generando comparacion de distribuciones...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for i, col in enumerate(COLS_NUMERICAS):
        axes[0, i].hist(X_original[col], bins=20, color='steelblue', edgecolor='black')
        axes[0, i].set_title(f'Original: {col}')
        axes[0, i].set_ylabel('Frecuencia')

        axes[1, i].hist(df_scaled[col], bins=20, color='salmon', edgecolor='black')
        axes[1, i].set_title(f'Escalado: {col}')
        axes[1, i].set_ylabel('Frecuencia')

    fig.suptitle('Comparacion de Distribuciones: Original vs Escalado (StandardScaler)', fontsize=14)
    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '03_comparacion_distribuciones.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 2. Boxplots comparativos ---
    print("  Generando boxplots comparativos...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].boxplot([X_original[col] for col in COLS_NUMERICAS], labels=['Age', 'Income', 'Score'])
    axes[0].set_title('Boxplot - Datos Originales')
    axes[0].set_ylabel('Valor')

    axes[1].boxplot([df_scaled[col] for col in COLS_NUMERICAS], labels=['Age', 'Income', 'Score'])
    axes[1].set_title('Boxplot - Datos Escalados (StandardScaler)')
    axes[1].set_ylabel('Valor estandarizado')

    fig.suptitle('Comparacion de Escala: Original vs StandardScaler', fontsize=14)
    plt.tight_layout()
    ruta = os.path.join(RESULTS_DIR, '03_comparacion_boxplots.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 3. Pairplot datos escalados ---
    print("  Generando pairplot de datos escalados...")
    g = sns.pairplot(df_scaled, diag_kind='hist')
    g.figure.suptitle('Pairplot - Datos Escalados (StandardScaler)', y=1.02, fontsize=14)
    ruta = os.path.join(RESULTS_DIR, '03_pairplot_escalado.png')
    g.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    print("\n  Todas las visualizaciones generadas exitosamente.")


if __name__ == '__main__':
    df = cargar_datos()
    X_original, df_scaled, scaler = aplicar_standard_scaler(df)
    generar_reporte(X_original, df_scaled, scaler)
    generar_visualizaciones(X_original, df_scaled)
