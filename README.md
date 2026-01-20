# Segundo-Trabalho-de-IA
  DBSCAN Clustering - Atividade IA
  Este projeto implementa o algoritmo de agrupamento espacial baseado em densidade com ruído (DBSCAN) para a disciplina de Inteligência Artificial.

#Como Executar o Projeto
  Siga os passos abaixo para configurar o ambiente virtual e rodar os testes nos datasets (Two Moons, Two Circles e Iris).

#1. Criar o Ambiente Virtual (venv)
  No terminal, dentro da pasta do projeto, execute:
    Windows
      python -m venv venv

    Linux/Mac
      python3 -m venv venv

#2. Ativar o Ambiente Virtual
    Windows
      .\venv\Scripts\activate

    Linux/Mac
      source venv/bin/activate

#3. Instalar as Dependências
  Com o ambiente ativo, instale as bibliotecas necessárias (numpy, scipy, matplotlib e sklearn):
    pip install numpy scipy matplotlib scikit-learn

#4. Executar os Testes
  Para gerar os gráficos e as estatísticas de agrupamento, basta rodar o script principal:
    python main.py

#Estrutura do Projeto
  src/dbscan_clustering.py: Implementação da classe DBSCAN com suporte a diferentes métricas de distância e tipos de pontos (Núcleo, Borda e Ruído).
  main.py: Script de teste que gera os dados, executa a clusterização e plota os resultados comparativos.
