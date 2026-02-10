"""
08 - Analisis Basico y Resumen Final
Proyecto: Segmentacion de Clientes usando Modelos No Supervisados
Maestria en IA - UEES - Semana 3
Alumno: Ingeniero Gonzalo Mejia Alcivar
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
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


def ejecutar_pipeline():
    """Ejecuta todo el pipeline y retorna los resultados."""
    # Carga y escalado
    ruta = os.path.join(DATA_DIR, 'Mall_Customers.csv')
    df = pd.read_csv(ruta)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[COLS_NUMERICAS])

    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

    # DBSCAN
    dbscan = DBSCAN(eps=0.6, min_samples=5)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    print("  Pipeline completo ejecutado.")
    return df, X_scaled, X_pca, X_tsne, kmeans, pca


def conteo_clusters(df):
    """Muestra el conteo de clusters unicos de ambos algoritmos."""
    lineas = []

    lineas.append("--- CONTEO DE CLUSTERS UNICOS ---")

    kmeans_unicos = sorted(df['KMeans_Cluster'].unique())
    dbscan_unicos = sorted(df['DBSCAN_Cluster'].unique())

    lineas.append(f"\n  K-Means:")
    lineas.append(f"    Clusters unicos: {kmeans_unicos}")
    lineas.append(f"    Total clusters:  {len(kmeans_unicos)}")
    for c in kmeans_unicos:
        n = (df['KMeans_Cluster'] == c).sum()
        lineas.append(f"    Cluster {c}: {n} clientes ({n/len(df)*100:.1f}%)")

    lineas.append(f"\n  DBSCAN:")
    lineas.append(f"    Clusters unicos: {dbscan_unicos}")
    n_clusters_db = len([c for c in dbscan_unicos if c != -1])
    lineas.append(f"    Total clusters:  {n_clusters_db} (+ ruido)")
    for c in dbscan_unicos:
        n = (df['DBSCAN_Cluster'] == c).sum()
        etiqueta = "Ruido" if c == -1 else f"Cluster {c}"
        lineas.append(f"    {etiqueta}: {n} clientes ({n/len(df)*100:.1f}%)")

    texto = "\n".join(lineas)
    print(texto)
    return texto


def resumen_perfiles(df):
    """Genera el resumen de perfiles detectados."""
    lineas = []

    lineas.append("\n--- PERFILES DE CLIENTES (K-Means K=4) ---")
    medias = df.groupby('KMeans_Cluster')[COLS_NUMERICAS].mean()

    perfiles = {0: 'Conservadores', 1: 'Premium', 2: 'Jovenes activos', 3: 'Ahorradores'}
    descripciones = {
        0: 'Edad mayor, ingreso y gasto moderados',
        1: 'Jovenes con alto ingreso y alto gasto',
        2: 'Bajo ingreso pero gasto medio-alto',
        3: 'Alto ingreso pero bajo gasto',
    }

    for idx, row in medias.iterrows():
        lineas.append(f"\n  Cluster {idx} - {perfiles[idx]}:")
        lineas.append(f"    Edad promedio:    {row['Age']:.0f} anios")
        lineas.append(f"    Ingreso promedio: {row['Annual Income (k$)']:.0f} k$")
        lineas.append(f"    Gasto promedio:   {row['Spending Score (1-100)']:.0f}/100")
        lineas.append(f"    Descripcion:      {descripciones[idx]}")
        lineas.append(f"    Clientes:         {(df['KMeans_Cluster'] == idx).sum()}")

    texto = "\n".join(lineas)
    print(texto)
    return texto


def generar_resumen_visual(df, X_pca, X_tsne, pca):
    """Genera un grafico resumen con los 4 paneles principales."""
    print("\n  Generando resumen visual...")

    perfiles = {0: 'Conservadores', 1: 'Premium', 2: 'Jovenes activos', 3: 'Ahorradores'}
    colores = {0: 'steelblue', 1: 'gold', 2: 'limegreen', 3: 'tomato'}

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: K-Means Income vs Spending
    ax1 = fig.add_subplot(gs[0, 0])
    for c, nombre in perfiles.items():
        mask = df['KMeans_Cluster'] == c
        ax1.scatter(
            df.loc[mask, 'Annual Income (k$)'], df.loc[mask, 'Spending Score (1-100)'],
            c=colores[c], s=50, alpha=0.7, edgecolors='black', linewidth=0.5, label=nombre
        )
    ax1.set_xlabel('Annual Income (k$)', fontsize=11)
    ax1.set_ylabel('Spending Score (1-100)', fontsize=11)
    ax1.set_title('K-Means: Income vs Spending', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: DBSCAN Income vs Spending
    ax2 = fig.add_subplot(gs[0, 1])
    dbscan_unicos = sorted(df['DBSCAN_Cluster'].unique())
    cmap_db = plt.cm.tab10
    for i, c in enumerate(dbscan_unicos):
        mask = df['DBSCAN_Cluster'] == c
        color = 'gray' if c == -1 else cmap_db(i)
        label = 'Ruido' if c == -1 else f'Cluster {c}'
        ax2.scatter(
            df.loc[mask, 'Annual Income (k$)'], df.loc[mask, 'Spending Score (1-100)'],
            c=[color], s=50, alpha=0.7, edgecolors='black', linewidth=0.5, label=label
        )
    ax2.set_xlabel('Annual Income (k$)', fontsize=11)
    ax2.set_ylabel('Spending Score (1-100)', fontsize=11)
    ax2.set_title('DBSCAN: Income vs Spending', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: PCA
    ax3 = fig.add_subplot(gs[1, 0])
    for c, nombre in perfiles.items():
        mask = df['KMeans_Cluster'] == c
        ax3.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colores[c], s=50, alpha=0.7, edgecolors='black', linewidth=0.5, label=nombre
        )
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax3.set_title('PCA (Reduccion Lineal)', fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: t-SNE
    ax4 = fig.add_subplot(gs[1, 1])
    for c, nombre in perfiles.items():
        mask = df['KMeans_Cluster'] == c
        ax4.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=colores[c], s=50, alpha=0.7, edgecolors='black', linewidth=0.5, label=nombre
        )
    ax4.set_xlabel('t-SNE Dim 1', fontsize=11)
    ax4.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax4.set_title('t-SNE (Reduccion No Lineal)', fontsize=13)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Resumen Final: Segmentacion de Clientes', fontsize=16, fontweight='bold')

    ruta = os.path.join(RESULTS_DIR, '08_resumen_visual.png')
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")


def generar_reporte(df, txt_conteo, txt_perfiles):
    """Genera el reporte final."""
    lineas = []
    lineas.append("=" * 60)
    lineas.append("  ETAPA 8: ANALISIS BASICO Y RESUMEN FINAL")
    lineas.append("=" * 60)

    lineas.append(f"\n  Total de clientes analizados: {len(df)}")
    lineas.append(f"  Variables utilizadas: {COLS_NUMERICAS}")

    lineas.append("\n" + txt_conteo)
    lineas.append(txt_perfiles)

    lineas.append("\n--- RESUMEN COMPARATIVO ---")
    lineas.append("  K-Means (k=4):  4 clusters bien definidos, segmentacion granular")
    lineas.append("  DBSCAN:         2 clusters + ruido, detecta outliers automaticamente")
    lineas.append("  PCA:            Reduccion lineal, preserva varianza global")
    lineas.append("  t-SNE:          Reduccion no lineal, mejor separacion visual")

    lineas.append("\n--- CONCLUSIONES ---")
    lineas.append("  1. El numero optimo de clusters segun K-Means es 4 (metodo del codo)")
    lineas.append("  2. K-Means ofrece segmentacion mas detallada, DBSCAN detecta outliers")
    lineas.append("  3. PCA y t-SNE confirman la separacion de los clusters")
    lineas.append("  4. Se identificaron 4 perfiles: Conservadores, Premium,")
    lineas.append("     Jovenes activos y Ahorradores")

    lineas.append("\n" + "=" * 60)

    texto = "\n".join(lineas)
    print(texto)

    ruta_reporte = os.path.join(RESULTS_DIR, '08_analisis_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    df, X_scaled, X_pca, X_tsne, kmeans, pca = ejecutar_pipeline()
    txt_conteo = conteo_clusters(df)
    txt_perfiles = resumen_perfiles(df)
    generar_resumen_visual(df, X_pca, X_tsne, pca)
    generar_reporte(df, txt_conteo, txt_perfiles)
