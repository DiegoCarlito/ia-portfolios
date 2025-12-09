import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons, make_blobs

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100 
plt.rcParams['figure.figsize'] = (16, 7)

def gerar_dados_hibridos(n_samples=1000):
    np.random.seed(42)
    
    # 1. Estruturas não-lineares (Moons) - 600 pontos
    X_moons, _ = make_moons(n_samples=600, noise=0.05)
    
    # 2. Estruturas densas (Blobs) - 400 pontos
    X_blobs, _ = make_blobs(n_samples=400, centers=[(-1, -1), (2, 2)], cluster_std=0.4)
    
    # 3. Adicionando ruído proposital (Outliers) - 20 pontos
    outliers = np.random.uniform(low=-2.5, high=3, size=(20, 2))
    
    # Combinando tudo
    X = np.vstack([X_moons, X_blobs, outliers])
    return X

# =============================================================================
# 1. PREPARAÇÃO DOS DADOS
# =============================================================================
print("--- 1. Gerando e Processando Dados ---")
X = gerar_dados_hibridos()

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =============================================================================
# 2. MODELAGEM
# =============================================================================
print("--- 2. Treinando Modelos ---")

# K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
sil_kmeans = silhouette_score(X_scaled, labels_kmeans)

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

# Métricas DBSCAN (excluindo ruído para cálculo justo da silhueta)
mask_core = labels_dbscan != -1
if np.sum(mask_core) > 0:
    sil_dbscan = silhouette_score(X_scaled[mask_core], labels_dbscan[mask_core])
else:
    sil_dbscan = 0

print(f"K-Means Silhouette: {sil_kmeans:.4f}")
print(f"DBSCAN Silhouette: {sil_dbscan:.4f}")

# =============================================================================
# 3. VISUALIZAÇÃO
# =============================================================================
fig, axes = plt.subplots(1, 2)

# --- Gráfico 1: K-Means ---
# s=30 e alpha=0.7 para ver a densidade dos pontos
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, 
                           cmap='viridis', s=30, alpha=0.7, edgecolor='k', linewidth=0.3)

axes[0].set_title(f'K-Means (Silhouette: {sil_kmeans:.2f})\nFalha: Cortes lineares arbitrários', fontsize=14)
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')

# Legenda
legend1 = axes[0].legend(*scatter1.legend_elements(), title="Clusters", loc="upper right")
axes[0].add_artist(legend1)

# --- Gráfico 2: DBSCAN ---
unique_labels = set(labels_dbscan)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    class_member_mask = (labels_dbscan == k)
    xy = X_pca[class_member_mask]
    
    if k == -1:
        # Plota Ruído (Outliers)
        col = [0.2, 0.2, 0.2, 1]
        axes[1].scatter(xy[:, 0], xy[:, 1], c=[col], marker='x', s=40, 
                        alpha=0.6, linewidth=1.5, label='Ruído (Outlier)')
    else:
        # Plota Clusters Válidos
        axes[1].scatter(xy[:, 0], xy[:, 1], c=[col], marker='o', s=35, 
                        alpha=0.8, edgecolor='k', linewidth=0.3, label=f'Cluster {k}')

axes[1].set_title(f'DBSCAN (Silhouette: {sil_dbscan:.2f})\nSucesso: Topologia correta + Detecção de Ruído', fontsize=14)
axes[1].set_xlabel('Principal Component 1')
axes[1].legend(loc='upper right', frameon=True, framealpha=0.9)

plt.suptitle('Comparativo de Algoritmos: Partição vs Densidade', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()
