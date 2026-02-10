"""
06 - PCA Reduccion de Dimensionalidad
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


def aplicar_pca(X_scaled):
    """Aplica PCA reduciendo de 3 a 2 componentes."""
    print("\n  Aplicando PCA (3 -> 2 componentes)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"  Varianza explicada por PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    print(f"  Varianza explicada por PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    print(f"  Varianza total explicada:   {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)")

    return X_pca, pca


def generar_visualizaciones(df, X_pca, pca):
    """Genera los graficos de PCA."""

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    # --- 1. PCA con colores de K-Means ---
    print("\n  Generando PCA con clusters K-Means...")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=df['KMeans_Cluster'], cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    ax.set_xlabel(f'PC1 ({var_pc1:.1f}% varianza)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_pc2:.1f}% varianza)', fontsize=12)
    ax.set_title('Visualizacion PCA de Clusters K-Means (K=4)', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '06_pca_kmeans_clusters.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 2. Varianza explicada por componente ---
    print("  Generando grafico de varianza explicada...")
    fig, ax = plt.subplots(figsize=(8, 5))

    componentes = ['PC1', 'PC2', 'PC3 (descartado)']
    # PCA completo para mostrar las 3 componentes
    pca_full = PCA(n_components=3)
    pca_full.fit(StandardScaler().fit_transform(df[COLS_NUMERICAS]))
    varianzas = pca_full.explained_variance_ratio_ * 100
    acumulada = np.cumsum(varianzas)

    bars = ax.bar(componentes, varianzas, color=['steelblue', 'salmon', 'lightgray'], edgecolor='black')
    ax.plot(componentes, acumulada, marker='o', color='darkred', linewidth=2, label='Acumulada')

    for i, (v, a) in enumerate(zip(varianzas, acumulada)):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('Varianza Explicada (%)', fontsize=12)
    ax.set_title('Varianza Explicada por Componente Principal', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '06_pca_varianza_explicada.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    # --- 3. PCA con etiquetas de cluster por perfil ---
    print("  Generando PCA con perfiles de clientes...")
    perfiles = {0: 'Conservadores', 1: 'Premium', 2: 'Jovenes activos', 3: 'Ahorradores'}
    colores_perfil = {0: 'steelblue', 1: 'gold', 2: 'limegreen', 3: 'tomato'}

    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster_id, nombre in perfiles.items():
        mask = df['KMeans_Cluster'] == cluster_id
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colores_perfil[cluster_id], s=60, alpha=0.7,
            edgecolors='black', linewidth=0.5, label=nombre
        )
    ax.set_xlabel(f'PC1 ({var_pc1:.1f}% varianza)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_pc2:.1f}% varianza)', fontsize=12)
    ax.set_title('PCA - Perfiles de Clientes (K-Means K=4)', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(RESULTS_DIR, '06_pca_perfiles_clientes.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

    print("\n  Todas las visualizaciones generadas exitosamente.")


def generar_reporte(df, X_pca, pca):
    """Genera el reporte de PCA."""
    lineas = []
    lineas.append("=" * 60)
    lineas.append("  ETAPA 6: PCA - REDUCCION DE DIMENSIONALIDAD")
    lineas.append("=" * 60)

    lineas.append("\n--- CONFIGURACION ---")
    lineas.append("  Componentes originales: 3 (Age, Annual Income, Spending Score)")
    lineas.append("  Componentes reducidos:  2 (PC1, PC2)")

    lineas.append("\n--- VARIANZA EXPLICADA ---")
    lineas.append(f"  PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    lineas.append(f"  PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    lineas.append(f"  Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")

    lineas.append("\n--- COMPOSICION DE COMPONENTES ---")
    lineas.append("  Cada componente es una combinacion lineal de las variables originales:")
    for i, comp in enumerate(pca.components_):
        lineas.append(f"  PC{i+1} = {comp[0]:.4f}*Age + {comp[1]:.4f}*Income + {comp[2]:.4f}*Spending")

    lineas.append("\n--- MEDIAS POR CLUSTER EN ESPACIO PCA ---")
    df_pca = df.copy()
    df_pca['PC1'] = X_pca[:, 0]
    df_pca['PC2'] = X_pca[:, 1]
    medias_pca = df_pca.groupby('KMeans_Cluster')[['PC1', 'PC2']].mean()
    lineas.append(medias_pca.to_string())

    lineas.append("\n--- MEDIAS POR CLUSTER EN VARIABLES ORIGINALES ---")
    medias = df.groupby('KMeans_Cluster')[COLS_NUMERICAS].mean()
    lineas.append(medias.to_string())

    lineas.append("\n--- INTERPRETACION ---")
    lineas.append("  PCA reduce las 3 variables a 2 componentes principales preservando")
    lineas.append(f"  el {sum(pca.explained_variance_ratio_)*100:.1f}% de la varianza original.")
    lineas.append("  La visualizacion confirma que los 4 clusters de K-Means estan")
    lineas.append("  bien separados en el espacio reducido.")

    lineas.append("\n" + "=" * 60)

    texto = "\n".join(lineas)
    print(texto)

    ruta_reporte = os.path.join(RESULTS_DIR, '06_pca_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    df, X_scaled = cargar_escalar_y_clusterizar()
    X_pca, pca = aplicar_pca(X_scaled)
    generar_visualizaciones(df, X_pca, pca)
    generar_reporte(df, X_pca, pca)
