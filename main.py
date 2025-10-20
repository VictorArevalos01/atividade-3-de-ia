# Importação de bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Função responsável por carregar os dados
def carregar_dados(caminho_arquivo):
    """
    Faz a leitura de um arquivo CSV contendo o dataset.
    Considera-se que a última coluna representa os rótulos (saídas desejadas)
    e as demais colunas são as variáveis de entrada.
    """
    try:
        dados = pd.read_csv(caminho_arquivo)
        entradas = dados.iloc[:, :-1].values
        saidas = dados.iloc[:, -1].values
        print(f"Arquivo carregado com sucesso: {caminho_arquivo}")
        print(f"Dimensão das entradas: {entradas.shape}")
        print(f"Dimensão das saídas: {saidas.shape}")
        return entradas, saidas
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return None, None
    except Exception as erro:
        print(f"Falha ao carregar os dados: {erro}")
        return None, None

# Classe representando o Perceptron (modelo MCP)
class MCPPerceptron:
    def __init__(self, taxa_aprendizado=0.01, total_epocas=100):
        """
        Inicializa o modelo do Perceptron.

        Parâmetros:
        taxa_aprendizado (float): Passo de atualização dos pesos.
        total_epocas (int): Número máximo de ciclos de treino.
        """
        self.taxa_aprendizado = taxa_aprendizado
        self.total_epocas = total_epocas
        self.pesos = None
        self.viés = None
        self.erros_por_epoca = []

    def funcao_ativacao(self, entrada_liquida):
        """
        Função de ativação tipo degrau.
        Retorna 1 se a entrada líquida for >= 0, caso contrário, 0.
        """
        return 1 if entrada_liquida >= 0 else 0

    def prever(self, matriz_entradas):
        """
        Realiza a predição de classes com base nas entradas fornecidas.

        Parâmetros:
        matriz_entradas (array numpy): Conjunto de entradas para teste.

        Retorno:
        Array numpy com as previsões.
        """
        if self.pesos is None or self.viés is None:
            raise ValueError("Modelo não treinado. Execute o método 'ajustar' antes de prever.")

        matriz_entradas = np.asarray(matriz_entradas)

        if len(self.pesos) == matriz_entradas.shape[1] + 1:
            entradas_com_bias = np.hstack((np.ones((matriz_entradas.shape[0], 1)), matriz_entradas))
            entradas_liquidas = np.dot(entradas_com_bias, self.pesos)
        else:
            entradas_liquidas = np.dot(matriz_entradas, self.pesos) + self.viés

        saidas_previstas = np.array([self.funcao_ativacao(valor) for valor in entradas_liquidas])
        return saidas_previstas

    def ajustar(self, entradas_treino, saidas_treino):
        """
        Executa o treinamento do Perceptron.

        Parâmetros:
        entradas_treino (numpy array): Dados de entrada para treino.
        saidas_treino (numpy array): Rótulos correspondentes.
        """
        total_amostras, total_atributos = entradas_treino.shape

        self.pesos = np.random.rand(total_atributos + 1) * 0.01
        self.viés = 0

        entradas_treino_bias = np.hstack((np.ones((total_amostras, 1)), entradas_treino))

        print("\nIniciando o treinamento do Perceptron...")
        for epoca in range(self.total_epocas):
            erros = 0
            for idx in range(total_amostras):
                entrada_atual = entradas_treino_bias[idx]
                entrada_liquida = np.dot(entrada_atual, self.pesos)
                resposta = self.funcao_ativacao(entrada_liquida)
                erro = saidas_treino[idx] - resposta

                if erro != 0:
                    self.pesos += self.taxa_aprendizado * erro * entrada_atual
                    erros += 1

            self.erros_por_epoca.append(erros)

            if (epoca + 1) % 10 == 0 or epoca == 0 or epoca == self.total_epocas - 1:
                print(f"Época {epoca + 1}/{self.total_epocas}: Total de erros = {erros}")

            if erros == 0 and epoca > 0:
                print(f"Modelo convergiu na época {epoca + 1}.")
                break
        print("Treinamento finalizado.")

# --- Execução principal ---
if __name__ == "__main__":
    arquivo_dados = 'selected_digits.csv'

    # Etapa 1: Carregar o conjunto de dados
    entradas, saidas = carregar_dados(arquivo_dados)

    if entradas is None or saidas is None:
        exit()

    # Etapa 2: Divisão dos dados em treino e teste
    entradas_treino, entradas_teste, saidas_treino, saidas_teste = train_test_split(
        entradas, saidas, test_size=0.2, random_state=42, stratify=saidas
    )

    print(f"\nDimensão treino: entradas={entradas_treino.shape}, saídas={saidas_treino.shape}")
    print(f"Dimensão teste: entradas={entradas_teste.shape}, saídas={saidas_teste.shape}")

    # Etapa 3: Inicialização e treino do modelo
    perceptron_mcp = MCPPerceptron(taxa_aprendizado=0.1, total_epocas=100)
    perceptron_mcp.ajustar(entradas_treino, saidas_treino)

    # Etapa 4: Plotagem da curva de erros por época
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(perceptron_mcp.erros_por_epoca) + 1), perceptron_mcp.erros_por_epoca, marker='o')
    plt.title('Evolução dos Erros por Época')
    plt.xlabel('Época')
    plt.ylabel('Quantidade de Erros')
    plt.grid(True)
    plt.xticks(np.arange(1, len(perceptron_mcp.erros_por_epoca) + 1, max(1, len(perceptron_mcp.erros_por_epoca) // 10)))
    plt.show()

    # Etapa 5: Avaliação do modelo no conjunto de teste
    print("\nRealizando predições no conjunto de teste...")
    previsoes = perceptron_mcp.prever(entradas_teste)

    print("\nExemplo de algumas predições:")
    for i in range(min(10, len(saidas_teste))):
        print(f"Esperado: {saidas_teste[i]}, Previsto: {previsoes[i]}")

    precisao = accuracy_score(saidas_teste, previsoes)
    print(f"\nAcurácia obtida no teste: {precisao:.4f}")

    # Etapa 6: Visualização espacial (apenas para dados 2D)
    if entradas.shape[1] == 2:
        plt.figure(figsize=(10, 8))

        plt.scatter(entradas_teste[saidas_teste == 0, 0], entradas_teste[saidas_teste == 0, 1],
                    color='red', marker='o', label='Classe 0 (Real)')
        plt.scatter(entradas_teste[saidas_teste == 1, 0], entradas_teste[saidas_teste == 1, 1],
                    color='blue', marker='x', label='Classe 1 (Real)')

        plt.scatter(entradas_teste[previsoes == 0, 0], entradas_teste[previsoes == 0, 1],
                    facecolors='none', edgecolors='orange', marker='o', s=100,
                    label='Classe 0 (Previsto)')
        plt.scatter(entradas_teste[previsoes == 1, 0], entradas_teste[previsoes == 1, 1],
                    facecolors='none', edgecolors='green', marker='x', s=100,
                    label='Classe 1 (Previsto)')

        x_min, x_max = entradas[:, 0].min() - 1, entradas[:, 0].max() + 1
        y_min, y_max = entradas[:, 1].min() - 1, entradas[:, 1].max() + 1

        if perceptron_mcp.pesos is not None and perceptron_mcp.pesos[2] != 0:
            linha_x = np.linspace(x_min, x_max, 100)
            linha_y = (-perceptron_mcp.pesos[0] - perceptron_mcp.pesos[1] * linha_x) / perceptron_mcp.pesos[2]
            plt.plot(linha_x, linha_y, color='purple', linestyle='--', label='Fronteira de Decisão')

        plt.title('Resultado da Classificação no Conjunto de Teste')
        plt.xlabel('Atributo 1')
        plt.ylabel('Atributo 2')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.grid(True)
        plt.show()
