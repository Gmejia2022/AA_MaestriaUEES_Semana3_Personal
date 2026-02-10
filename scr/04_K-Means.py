"""
04 - K-Means Clustering
Proyecto: Segmentacion de Clientes usando Modelos No Supervisados
Maestria en IA - UEES - Semana 3
Alumno: Ingeniero Gonzalo Mejia Alcivar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# === Rutas del proyecto ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Variables numericas
COLS_NUMERICAS = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']


def cargar_y_escalar():
    """Carga el dataset y aplica StandardScaler."""
    ruta = os.path.join(DATA_DIR, 'Mall_Customers.csv')
    df = pd.read_csv(ruta)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[COLS_NUMERICAS])

    print("  Dataset cargado y escalado exitosamente.")
    return df, X_scaled


def metodo_del_codo(X_scaled):
    """Ejecuta el metodo del codo para determinar el K optimo."""
    print("\n  Ejecutando metodo del codo (k=1 a k=9)...")
    inertias = []
    rango_k = range(1, 10)

    for k in rango_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"    k={k} -> Inercia: {km.inertia_:.2f}")

    # Grafico del codo
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rango_k, inertias, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axvline(x=4, color='red', linestyle='--', label='K optimo = 4')
    ax.set_xlabel('Numero de Clusters (K)', fontsize=12)
    ax.set_ylabel('Inercia', fontsize=12)
    ax.set_title('Metodo del Codo para K-Means', fontsize=14)
    ax.set_xticks(rango_k)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '04_metodo_del_codo.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    return inertias


def entrenar_kmeans(df, X_scaled, k=4):
    """Entrena el modelo K-Means con el K optimo."""
    print(f"\n  Entrenando K-Means con k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

    print(f"  Modelo entrenado. Clusters asignados:")
    print(df['KMeans_Cluster'].value_counts().sort_index().to_string())

    return df, kmeans


def generar_visualizaciones(df, X_scaled):
    """Genera los scatterplots de los clusters."""

    # --- 1. Scatterplot Income vs Spending Score ---
    print("\n  Generando scatterplot Income vs Spending Score...")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        X_scaled[:, 1], X_scaled[:, 2],
        c=df['KMeans_Cluster'], cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    ax.set_xlabel('Annual Income (escalado)', fontsize=12)
    ax.set_ylabel('Spending Score (escalado)', fontsize=12)
    ax.set_title('Segmentacion por K-Means (K=4) - Income vs Spending', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '04_kmeans_income_vs_spending.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 2. Scatterplot Age vs Spending Score ---
    print("  Generando scatterplot Age vs Spending Score...")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        X_scaled[:, 0], X_scaled[:, 2],
        c=df['KMeans_Cluster'], cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    ax.set_xlabel('Age (escalado)', fontsize=12)
    ax.set_ylabel('Spending Score (escalado)', fontsize=12)
    ax.set_title('Segmentacion por K-Means (K=4) - Age vs Spending', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '04_kmeans_age_vs_spending.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 3. Scatterplot Age vs Income ---
    print("  Generando scatterplot Age vs Income...")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        X_scaled[:, 0], X_scaled[:, 1],
        c=df['KMeans_Cluster'], cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    ax.set_xlabel('Age (escalado)', fontsize=12)
    ax.set_ylabel('Annual Income (escalado)', fontsize=12)
    ax.set_title('Segmentacion por K-Means (K=4) - Age vs Income', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '04_kmeans_age_vs_income.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    print("\n  Todas las visualizaciones generadas exitosamente.")


def generar_reporte(df, kmeans, inertias):
    """Genera el reporte con medias por cluster y resultados."""
    lineas = []
    lineas.append("=" * 60)
    lineas.append("  ETAPA 4: K-MEANS CLUSTERING")
    lineas.append("=" * 60)

    lineas.append("\n--- METODO DEL CODO (Inercia por K) ---")
    for k, inercia in enumerate(inertias, start=1):
        marcador = "  <-- K optimo" if k == 4 else ""
        lineas.append(f"  K={k} -> Inercia: {inercia:.2f}{marcador}")

    lineas.append(f"\n--- MODELO FINAL: K=4 ---")
    lineas.append(f"  Inercia del modelo: {kmeans.inertia_:.2f}")
    lineas.append(f"  Iteraciones: {kmeans.n_iter_}")

    lineas.append("\n--- DISTRIBUCION DE CLIENTES POR CLUSTER ---")
    conteo = df['KMeans_Cluster'].value_counts().sort_index()
    for cluster, cantidad in conteo.items():
        lineas.append(f"  Cluster {cluster}: {cantidad} clientes ({cantidad/len(df)*100:.1f}%)")

    lineas.append("\n--- MEDIAS POR CLUSTER ---")
    medias = df.groupby('KMeans_Cluster')[COLS_NUMERICAS].mean()
    lineas.append(medias.to_string())

    lineas.append("\n--- INTERPRETACION DE PERFILES ---")
    for idx, row in medias.iterrows():
        edad = row['Age']
        ingreso = row['Annual Income (k$)']
        gasto = row['Spending Score (1-100)']

        if ingreso > 70 and gasto > 60:
            perfil = "Premium: Alto ingreso y alto gasto"
        elif ingreso > 70 and gasto < 40:
            perfil = "Ahorradores: Alto ingreso pero bajo gasto"
        elif edad > 45:
            perfil = "Conservadores: Edad mayor, ingreso y gasto moderados"
        else:
            perfil = "Jovenes activos: Bajo ingreso pero gasto medio-alto"

        lineas.append(f"  Cluster {idx}: Edad={edad:.0f}, Ingreso={ingreso:.0f}k$, Gasto={gasto:.0f} -> {perfil}")

    lineas.append("\n" + "=" * 60)

    texto = "\n".join(lineas)
    print(texto)

    ruta_reporte = os.path.join(RESULTS_DIR, '04_kmeans_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    df, X_scaled = cargar_y_escalar()
    inertias = metodo_del_codo(X_scaled)
    df, kmeans = entrenar_kmeans(df, X_scaled, k=4)
    generar_visualizaciones(df, X_scaled)
    generar_reporte(df, kmeans, inertias)
