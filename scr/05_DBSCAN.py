"""
05 - DBSCAN Clustering
Proyecto: Segmentacion de Clientes usando Modelos No Supervisados
Maestria en IA - UEES - Semana 3
Alumno: Ingeniero Gonzalo Mejia Alcivar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
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


def entrenar_dbscan(df, X_scaled, eps=0.6, min_samples=5):
    """Entrena el modelo DBSCAN."""
    print(f"\n  Entrenando DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
    n_ruido = (df['DBSCAN_Cluster'] == -1).sum()

    print(f"  Clusters encontrados: {n_clusters}")
    print(f"  Puntos de ruido: {n_ruido}")
    print(f"  Distribucion:")
    print(df['DBSCAN_Cluster'].value_counts().sort_index().to_string())

    return df, dbscan, n_clusters, n_ruido


def generar_visualizaciones(df, X_scaled):
    """Genera los scatterplots de DBSCAN."""

    # Colores: -1 (ruido) en gris, clusters en colores
    clusters = df['DBSCAN_Cluster'].values
    colores_unicos = sorted(df['DBSCAN_Cluster'].unique())
    cmap = plt.cm.tab10
    color_map = {}
    idx_color = 0
    for c in colores_unicos:
        if c == -1:
            color_map[c] = 'gray'
        else:
            color_map[c] = cmap(idx_color)
            idx_color += 1
    colores = [color_map[c] for c in clusters]

    # --- 1. Scatterplot Income vs Spending Score ---
    print("\n  Generando scatterplot Income vs Spending Score...")
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in colores_unicos:
        mask = clusters == c
        label = f'Ruido ({(mask).sum()})' if c == -1 else f'Cluster {c} ({(mask).sum()})'
        ax.scatter(
            X_scaled[mask, 1], X_scaled[mask, 2],
            c=[color_map[c]], s=60, alpha=0.7, edgecolors='black', linewidth=0.5, label=label
        )
    ax.set_xlabel('Annual Income (escalado)', fontsize=12)
    ax.set_ylabel('Spending Score (escalado)', fontsize=12)
    ax.set_title('Segmentacion por DBSCAN - Income vs Spending', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '05_dbscan_income_vs_spending.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 2. Scatterplot Age vs Spending Score ---
    print("  Generando scatterplot Age vs Spending Score...")
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in colores_unicos:
        mask = clusters == c
        label = f'Ruido ({(mask).sum()})' if c == -1 else f'Cluster {c} ({(mask).sum()})'
        ax.scatter(
            X_scaled[mask, 0], X_scaled[mask, 2],
            c=[color_map[c]], s=60, alpha=0.7, edgecolors='black', linewidth=0.5, label=label
        )
    ax.set_xlabel('Age (escalado)', fontsize=12)
    ax.set_ylabel('Spending Score (escalado)', fontsize=12)
    ax.set_title('Segmentacion por DBSCAN - Age vs Spending', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '05_dbscan_age_vs_spending.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 3. Comparacion K-Means vs DBSCAN ---
    print("  Generando comparacion K-Means vs DBSCAN...")

    # Cargar K-Means para comparar
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # K-Means
    scatter1 = axes[0].scatter(
        X_scaled[:, 1], X_scaled[:, 2],
        c=kmeans_labels, cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    axes[0].set_xlabel('Annual Income (escalado)', fontsize=11)
    axes[0].set_ylabel('Spending Score (escalado)', fontsize=11)
    axes[0].set_title('K-Means (K=4)', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # DBSCAN
    for c in colores_unicos:
        mask = clusters == c
        label = 'Ruido' if c == -1 else f'Cluster {c}'
        axes[1].scatter(
            X_scaled[mask, 1], X_scaled[mask, 2],
            c=[color_map[c]], s=60, alpha=0.7, edgecolors='black', linewidth=0.5, label=label
        )
    axes[1].set_xlabel('Annual Income (escalado)', fontsize=11)
    axes[1].set_ylabel('Spending Score (escalado)', fontsize=11)
    axes[1].set_title('DBSCAN (eps=0.6, min_samples=5)', fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Comparacion: K-Means vs DBSCAN', fontsize=14)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '05_comparacion_kmeans_vs_dbscan.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    print("\n  Todas las visualizaciones generadas exitosamente.")


def generar_reporte(df, n_clusters, n_ruido):
    """Genera el reporte de DBSCAN."""
    lineas = []
    lineas.append("=" * 60)
    lineas.append("  ETAPA 5: DBSCAN CLUSTERING")
    lineas.append("=" * 60)

    lineas.append("\n--- PARAMETROS ---")
    lineas.append("  eps = 0.6 (radio maximo para considerar vecinos)")
    lineas.append("  min_samples = 5 (minimo de puntos para formar cluster)")

    lineas.append(f"\n--- RESULTADOS ---")
    lineas.append(f"  Clusters encontrados: {n_clusters}")
    lineas.append(f"  Puntos de ruido (outliers): {n_ruido} ({n_ruido/len(df)*100:.1f}%)")

    lineas.append("\n--- DISTRIBUCION POR CLUSTER ---")
    conteo = df['DBSCAN_Cluster'].value_counts().sort_index()
    for cluster, cantidad in conteo.items():
        etiqueta = "Ruido" if cluster == -1 else f"Cluster {cluster}"
        lineas.append(f"  {etiqueta}: {cantidad} clientes ({cantidad/len(df)*100:.1f}%)")

    lineas.append("\n--- MEDIAS POR CLUSTER (incluyendo ruido) ---")
    medias = df.groupby('DBSCAN_Cluster')[COLS_NUMERICAS].mean()
    lineas.append(medias.to_string())

    lineas.append("\n--- MEDIAS POR CLUSTER (sin ruido) ---")
    df_sin_ruido = df[df['DBSCAN_Cluster'] != -1]
    if len(df_sin_ruido) > 0:
        medias_sin_ruido = df_sin_ruido.groupby('DBSCAN_Cluster')[COLS_NUMERICAS].mean()
        lineas.append(medias_sin_ruido.to_string())

    lineas.append("\n--- INTERPRETACION ---")
    lineas.append("  Cluster -1 (Ruido): Clientes con patrones atipicos que no encajan")
    lineas.append("    en ninguna agrupacion densa. Suelen tener ingresos altos y gasto bajo.")
    for idx, row in medias.iterrows():
        if idx == -1:
            continue
        edad = row['Age']
        ingreso = row['Annual Income (k$)']
        gasto = row['Spending Score (1-100)']
        lineas.append(f"  Cluster {idx}: Edad={edad:.0f}, Ingreso={ingreso:.0f}k$, Gasto={gasto:.0f}")

    lineas.append("\n--- COMPARACION CON K-MEANS ---")
    lineas.append("  K-Means: 4 clusters, sin deteccion de outliers")
    lineas.append("  DBSCAN:  2 clusters + ruido, detecta outliers automaticamente")
    lineas.append("  DBSCAN es mas conservador pero identifica clientes atipicos")

    lineas.append("\n" + "=" * 60)

    texto = "\n".join(lineas)
    print(texto)

    ruta_reporte = os.path.join(RESULTS_DIR, '05_dbscan_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    df, X_scaled = cargar_y_escalar()
    df, dbscan, n_clusters, n_ruido = entrenar_dbscan(df, X_scaled, eps=0.6, min_samples=5)
    generar_visualizaciones(df, X_scaled)
    generar_reporte(df, n_clusters, n_ruido)
