import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles, make_moons, load_iris
from src.dbscan_clustering import dbscan
import numpy as np


def test_two_circles():

    # Gera os dados
    X, y = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

    # Aplica DBSCAN
    modelo_dbscan = dbscan(epsilon=0.15, min_samples=8, distance='euclidean')
    rotulos, tipos = modelo_dbscan.fit(X)

    #Plota com identificação de tipos (Core, Borda, Ruído)
    plt.figure()
    cores = []
    for tipo in tipos:
        if tipo == -1:  # Ruído
            cores.append('red')
        elif tipo == 0:  # Borda
            cores.append('green')
        else:  # Core (tipo == 1)
            cores.append('blue')

    plt.scatter(X[:, 0], X[:, 1], c=cores, edgecolors='black', linewidth=0.5, s=50)
    plt.title("Dataset: Two Circles - (Ruído = R, Borda = G, Núcleo = B)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    #IMPRIME ESTATÍSTICAS
    numero_clusters = len(set(rotulos)) - (1 if 0 in rotulos else 0)
    numero_core = (tipos == 1).sum()
    numero_borda = (tipos == 0).sum()
    numero_ruido = (tipos == -1).sum()

    print("\n" + "=" * 50)
    print("Two Circles Dataset")
    print(f"Número de clusters encontrados: {numero_clusters}")
    print(f"Número de pontos CORE: {numero_core}")
    print(f"Número de pontos BORDA: {numero_borda}")
    print(f"Número de pontos RUÍDO: {numero_ruido}")
    print("=" * 50 + "\n")

def test_two_moons():

    X, y = make_moons(n_samples=300, noise=0.08, random_state=42)

    modelo_dbscan = dbscan(epsilon=0.15, min_samples=8, distance='euclidean')
    rotulos, tipos = modelo_dbscan.fit(X)

    plt.figure()
    cores = []
    for tipo in tipos:
        if tipo == -1:
            cores.append('red')
        elif tipo == 0:
            cores.append('green')
        else:
            cores.append('blue')

    plt.scatter(X[:, 0], X[:, 1], c=cores, edgecolors='black', linewidth=0.5, s=50)
    plt.title("Dataset: Two Moons - (Ruído = R, Borda = G, Núcleo = B)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    numero_clusters = len(set(rotulos)) - (1 if 0 in rotulos else 0)
    numero_core = (tipos == 1).sum()
    numero_borda = (tipos == 0).sum()
    numero_ruido = (tipos == -1).sum()

    print("Two Moons Dataset")
    print(f"Número de clusters encontrados: {numero_clusters}")
    print(f"Número de pontos CORE: {numero_core}")
    print(f"Número de pontos BORDA: {numero_borda}")
    print(f"Número de pontos RUÍDO: {numero_ruido}")
    print("=" * 50 + "\n")

def test_iris():
    
    iris = load_iris()
    X = iris.data[:, :3]  # Usa apenas as 3 primeiras features: sepal length, sepal width e petal length
    y_real = iris.target  # Classes reais: 0=setosa, 1=versicolor, 2=virginica

    modelo_dbscan = dbscan(epsilon=0.4, min_samples=5, distance='euclidean')
    rotulos, tipos = modelo_dbscan.fit(X)

    #PLOTA RESULTADO DBSCAN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cores = []
    for tipo in tipos:
        if tipo == -1:
            cores.append('red')
        elif tipo == 0:
            cores.append('green')
        else:
            cores.append('blue')

    ax1.scatter(X[:, 0], X[:, 1], c=cores, edgecolors='black', linewidth=0.5, s=50)
    ax1.set_title("Dataset: Iris - (Ruído = R, Borda = G, Núcleo = B)")
    ax1.set_xlabel("Sepal Length (cm)")
    ax1.set_ylabel("Sepal Width (cm)")

    #PLOTA CLASSES REAIS PARA COMPARAÇÃO
    ax2.scatter(X[y_real == 0, 0], X[y_real == 0, 1], label="Setosa")
    ax2.scatter(X[y_real == 1, 0], X[y_real == 1, 1], label="Versicolor")
    ax2.scatter(X[y_real == 2, 0], X[y_real == 2, 1], label="Virginica")
    ax2.legend()
    ax2.set_xlabel("Sepal length")
    ax2.set_ylabel("Sepal width")
    ax2.set_title("Iris - Classes Reais")
    plt.tight_layout()
    plt.show()

    #VISUALIZAÇÃO 3D DAS CLASSES REAIS
    X3 = iris.data[:, [0, 2, 3]]  # Sepal length, Petal length, Petal width

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=y_real, cmap='viridis')
    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Petal length")
    ax.set_zlabel("Petal width")
    ax.set_title("Iris - Visualização 3D (Classes Reais)")
    plt.show()

    #IMPRIME ESTATÍSTICAS
    numero_clusters = len(set(rotulos)) - (1 if 0 in rotulos else 0)
    numero_core = (tipos == 1).sum()
    numero_borda = (tipos == 0).sum()
    numero_ruido = (tipos == -1).sum()

    print("Iris Dataset")
    print(f"Número de clusters encontrados: {numero_clusters}")
    print(f"Número de pontos CORE: {numero_core}")
    print(f"Número de pontos BORDA: {numero_borda}")
    print(f"Número de pontos RUÍDO: {numero_ruido}")

if __name__ == "__main__":
    test_two_circles()
    test_two_moons()
    test_iris()
    print("=" * 50)
    print("Testes completados!")
    print("=" * 50)
