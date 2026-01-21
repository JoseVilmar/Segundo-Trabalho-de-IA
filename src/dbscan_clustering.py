import numpy as np
from scipy.spatial import distance

class dbscan:

  def __init__(self, epsilon = None, min_samples = None, distance = 'euclidean'):
    self.epsilon = epsilon
    self.min_samples = min_samples
    self.distance = distance

  def find_distance(self, dados, type = 'euclidean'):
    return distance.squareform(distance.pdist(dados, type))

  def find_neighbors(self, distancias_ponto):
    return np.where(distancias_ponto <= self.epsilon)[0]
  
  def expand_cluster(self, indices_vizinhos, matriz_distancias, cluster_atual, rotulos, tipos):

    for indice_vizinho in indices_vizinhos:
      
      if rotulos[indice_vizinho] == 0:

        vizinhos_do_vizinho = self.find_neighbors(matriz_distancias[indice_vizinho])
        
        if len(vizinhos_do_vizinho) >= self.min_samples:
            rotulos[indice_vizinho] = cluster_atual
            tipos[indice_vizinho] = 1  # Core

            rotulos, tipos = self.expand_cluster(vizinhos_do_vizinho, matriz_distancias, cluster_atual, rotulos, tipos)
        else:
            rotulos[indice_vizinho] = cluster_atual
            tipos[indice_vizinho] = 0  # Border
      
    return rotulos, tipos

  def fit(self, dados):

    matriz_distancias = self.find_distance(dados, self.distance)

    cluster_atual = 1
    numero_observacoes = dados.shape[0]
    rotulos = np.zeros(numero_observacoes)
    tipos = np.full(numero_observacoes, -1)  # -1 = ruído por padrão

    for indice_ponto in range(numero_observacoes):
    
      if rotulos[indice_ponto] == 0:
        
        indices_vizinhos = self.find_neighbors(matriz_distancias[indice_ponto])

        if len(indices_vizinhos) >= self.min_samples:
        
          rotulos[indice_ponto] = cluster_atual
          tipos[indice_ponto] = 1  # Core point

          rotulos, tipos = self.expand_cluster(indices_vizinhos, matriz_distancias, cluster_atual, rotulos, tipos)
         
          cluster_atual = cluster_atual + 1

    return rotulos, tipos


