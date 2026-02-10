"""
07 - t-SNE Reduccion de Dimensionalidad
Proyecto: Segmentacion de Clientes usando Modelos No Supervisados
Maestria en IA - UEES - Semana 3
Alumno: Ingeniero Gonzalo Mejia Alcivar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# === Rutas del proyecto ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Variables numericas
COLS_NUMERICAS = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']


def cargar_escalar_y_clusterizar():
    """Carga el dataset, escala y aplica K-Means (k=4)."""
    ruta = os.path.join(DATA_DIR, 'Mall_Customers.csv')
    df = pd.read_csv(ruta)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[COLS_NUMERICAS])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

    print("  Dataset cargado, escalado y clusterizado (K-Means k=4).")
    return df, X_scaled


def aplicar_tsne(X_scaled):
    """Aplica t-SNE reduciendo de 3 a 2 dimensiones."""
    print("\n  Aplicando t-SNE (3 -> 2 dimensiones)...")
    print("  Parametros: perplexity=30, learning_rate=200, random_state=42")

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    print(f"  KL divergence final: {tsne.kl_divergence_:.4f}")
    print(f"  Iteraciones: {tsne.n_iter_}")
    print("  t-SNE aplicado exitosamente.")

    return X_tsne, tsne


def generar_visualizaciones(df, X_scaled, X_tsne):
    """Genera los graficos de t-SNE."""

    perfiles = {0: 'Conservadores', 1: 'Premium', 2: 'Jovenes activos', 3: 'Ahorradores'}
    colores_perfil = {0: 'steelblue', 1: 'gold', 2: 'limegreen', 3: 'tomato'}

    # --- 1. t-SNE con colores de K-Means ---
    print("\n  Generando t-SNE con clusters K-Means...")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=df['KMeans_Cluster'], cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Visualizacion t-SNE de Clusters K-Means (K=4)', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '07_tsne_kmeans_clusters.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 2. t-SNE con perfiles de clientes ---
    print("  Generando t-SNE con perfiles de clientes...")
    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster_id, nombre in perfiles.items():
        mask = df['KMeans_Cluster'] == cluster_id
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=colores_perfil[cluster_id], s=60, alpha=0.7,
            edgecolors='black', linewidth=0.5, label=nombre
        )
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE - Perfiles de Clientes (K-Means K=4)', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '07_tsne_perfiles_clientes.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 3. Comparacion PCA vs t-SNE ---
    print("  Generando comparacion PCA vs t-SNE...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA
    for cluster_id, nombre in perfiles.items():
        mask = df['KMeans_Cluster'] == cluster_id
        axes[0].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colores_perfil[cluster_id], s=60, alpha=0.7,
            edgecolors='black', linewidth=0.5, label=nombre
        )
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    axes[0].set_title('PCA (lineal)', fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # t-SNE
    for cluster_id, nombre in perfiles.items():
        mask = df['KMeans_Cluster'] == cluster_id
        axes[1].scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=colores_perfil[cluster_id], s=60, alpha=0.7,
            edgecolors='black', linewidth=0.5, label=nombre
        )
    axes[1].set_xlabel('t-SNE Dim 1', fontsize=11)
    axes[1].set_ylabel('t-SNE Dim 2', fontsize=11)
    axes[1].set_title('t-SNE (no lineal)', fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Comparacion: PCA vs t-SNE con Clusters K-Means', fontsize=14)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '07_comparacion_pca_vs_tsne.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    print("\n  Todas las visualizaciones generadas exitosamente.")


def generar_reporte(df, X_tsne, tsne):
    """Genera el reporte de t-SNE."""
    lineas = []
    lineas.append("=" * 60)
    lineas.append("  ETAPA 7: t-SNE - REDUCCION DE DIMENSIONALIDAD")
    lineas.append("=" * 60)

    lineas.append("\n--- CONFIGURACION ---")
    lineas.append("  Dimensiones originales: 3 (Age, Annual Income, Spending Score)")
    lineas.append("  Dimensiones reducidas:  2 (Dim 1, Dim 2)")
    lineas.append("  perplexity:     30 (vecinos cercanos considerados)")
    lineas.append("  learning_rate:  200 (velocidad de aprendizaje)")
    lineas.append("  random_state:   42")

    lineas.append(f"\n--- METRICAS ---")
    lineas.append(f"  KL divergence: {tsne.kl_divergence_:.4f}")
    lineas.append(f"  Iteraciones:   {tsne.n_iter_}")

    lineas.append("\n--- MEDIAS POR CLUSTER EN ESPACIO t-SNE ---")
    df_tsne = df.copy()
    df_tsne['tSNE_Dim1'] = X_tsne[:, 0]
    df_tsne['tSNE_Dim2'] = X_tsne[:, 1]
    medias_tsne = df_tsne.groupby('KMeans_Cluster')[['tSNE_Dim1', 'tSNE_Dim2']].mean()
    lineas.append(medias_tsne.to_string())

    lineas.append("\n--- COMPARACION PCA vs t-SNE ---")
    lineas.append("  PCA:")
    lineas.append("    - Metodo lineal")
    lineas.append("    - Preserva varianza global")
    lineas.append("    - Rapido de calcular")
    lineas.append("    - Componentes son interpretables (combinacion lineal)")
    lineas.append("  t-SNE:")
    lineas.append("    - Metodo no lineal")
    lineas.append("    - Preserva estructura local (vecinos cercanos)")
    lineas.append("    - Mas lento de calcular")
    lineas.append("    - Mejor separacion visual de clusters densos")

    lineas.append("\n--- INTERPRETACION ---")
    lineas.append("  t-SNE genera agrupaciones mas compactas y separadas que PCA.")
    lineas.append("  Los 4 perfiles de clientes se distinguen claramente:")
    lineas.append("    Cluster 0: Conservadores (edad mayor, gasto moderado)")
    lineas.append("    Cluster 1: Premium (alto ingreso y alto gasto)")
    lineas.append("    Cluster 2: Jovenes activos (bajo ingreso, gasto medio-alto)")
    lineas.append("    Cluster 3: Ahorradores (alto ingreso, bajo gasto)")

    lineas.append("\n" + "=" * 60)

    texto = "\n".join(lineas)
    print(texto)

    ruta_reporte = os.path.join(RESULTS_DIR, '07_tsne_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    df, X_scaled = cargar_escalar_y_clusterizar()
    X_tsne, tsne = aplicar_tsne(X_scaled)
    generar_visualizaciones(df, X_scaled, X_tsne)
    generar_reporte(df, X_tsne, tsne)
