import random
import math
import matplotlib.pyplot as plt

class Cidade:
    """Representa uma cidade com coordenadas (x, y)."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distancia_para(self, outra_cidade):
        """Calcula a distância euclidiana entre esta cidade e outra."""
        dist_x = self.x - outra_cidade.x
        dist_y = self.y - outra_cidade.y
        return math.sqrt(dist_x**2 + dist_y**2)

class ResolvedorTSP_AG:
    """
    Classe que encapsula a lógica do Algoritmo Genético para o TSP.
    """
    def __init__(self, cidades, tam_populacao, taxa_mutacao, taxa_crossover, num_geracoes):
        self.cidades = cidades
        self.tam_populacao = tam_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.num_geracoes = num_geracoes
        self.populacao = self._criar_populacao_inicial()

    def _criar_populacao_inicial(self):
        """Cria uma população inicial de rotas aleatórias."""
        populacao = []
        for _ in range(self.tam_populacao):
            # Cria uma rota embaralhando uma cópia da lista de cidades
            rota = random.sample(self.cidades, len(self.cidades))
            populacao.append(rota)
        return populacao

    def _calcular_fitness(self, rota):
        """Calcula o fitness de uma rota (inverso da distância total)."""
        distancia_total = 0
        for i in range(len(rota)):
            cidade_origem = rota[i]
            # Se for a última cidade, a próxima é a primeira (ciclo)
            cidade_destino = rota[0] if i + 1 == len(rota) else rota[i + 1]
            distancia_total += cidade_origem.distancia_para(cidade_destino)
        
        # O fitness é o inverso da distância para que rotas menores tenham maior valor.
        return 1 / distancia_total

    def _selecao_torneio(self, fitness_populacao, tamanho_torneio=5):
        """Seleciona um indivíduo para ser pai usando o método de torneio."""
        
        # Converte o dicionário de fitness em uma lista de competidores (tuplas)
        lista_competidores = list(fitness_populacao.items())
        
        # Garante que o tamanho do torneio não seja maior que o número de
        # indivíduos únicos disponíveis na população.
        tamanho_real_torneio = min(tamanho_torneio, len(lista_competidores))
        
        # Se, por algum motivo, a lista estiver vazia, retorna o primeiro que encontrar
        # (embora isso seja raro devido ao elitismo)
        if tamanho_real_torneio == 0:
            return lista_competidores[0][0] 
        
        # Seleciona N competidores aleatórios da população
        competidores = random.sample(lista_competidores, tamanho_real_torneio)
        
        # O vencedor do torneio é aquele com o maior fitness
        vencedor = max(competidores, key=lambda item: item[1])
        return vencedor[0] # Retorna a rota (a chave do dicionário)

    def _crossover_ordenado(self, pai1, pai2):
        """
        Executa o Crossover Ordenado (OX1) para criar um filho.
        Este método preserva a ordem relativa das cidades dos pais.
        """
        filho = [None] * len(pai1)
        
        # Seleciona um subconjunto aleatório do primeiro pai
        inicio, fim = sorted(random.sample(range(len(pai1)), 2))
        
        # Copia o subconjunto do pai1 para o filho
        subconjunto_pai1 = pai1[inicio:fim]
        filho[inicio:fim] = subconjunto_pai1
        
        # Preenche o resto do filho com as cidades do pai2, na ordem em que aparecem,
        # sem duplicar as cidades que já vieram do pai1.
        ponteiro_pai2 = 0
        for i in range(len(filho)):
            if filho[i] is None:
                while pai2[ponteiro_pai2] in subconjunto_pai1:
                    ponteiro_pai2 += 1
                filho[i] = pai2[ponteiro_pai2]
                ponteiro_pai2 += 1
                
        return filho

    def _mutacao_troca(self, rota):
        """
        Executa a Mutação de Troca (Swap Mutation).
        Seleciona duas cidades aleatórias na rota e troca suas posições.
        """
        pos1, pos2 = random.sample(range(len(rota)), 2)
        rota[pos1], rota[pos2] = rota[pos2], rota[pos1]
        return rota

    def evoluir_populacao(self):
        """
        Executa um ciclo completo de evolução:
        Avaliação -> Seleção -> Crossover -> Mutação.
        """
        # 1. Avaliação: Calcula o fitness de cada indivíduo da população atual
        fitness_populacao = {tuple(rota): self._calcular_fitness(rota) for rota in self.populacao}
        
        nova_populacao = []
        
        # Mantém o melhor indivíduo da geração atual (elitismo)
        melhor_rota_atual = max(fitness_populacao, key=fitness_populacao.get)
        nova_populacao.append(list(melhor_rota_atual))
        
        # 2. Gera o resto da nova população
        while len(nova_populacao) < self.tam_populacao:
            # 3. Seleção: Seleciona dois pais
            pai1_tuple = self._selecao_torneio(fitness_populacao)
            pai2_tuple = self._selecao_torneio(fitness_populacao)
            pai1 = list(pai1_tuple)
            pai2 = list(pai2_tuple)

            filho = pai1 # Por padrão, o filho é uma cópia do pai1
            
            # 4. Crossover: Cruza os pais se a taxa de crossover for atingida
            if random.random() < self.taxa_crossover:
                filho = self._crossover_ordenado(pai1, pai2)
            
            # 5. Mutação: Aplica mutação no filho se a taxa de mutação for atingida
            if random.random() < self.taxa_mutacao:
                filho = self._mutacao_troca(filho)
            
            nova_populacao.append(filho)
        
        self.populacao = nova_populacao

    def encontrar_melhor_rota(self):
        """Executa o algoritmo genético por N gerações e retorna a melhor rota encontrada."""
        historico_distancias = []
        
        print(f"Executando o Algoritmo Genético por {self.num_geracoes} gerações...")
        
        for i in range(self.num_geracoes):
            self.evoluir_populacao()
            
            # Encontra a melhor rota da geração atual para registro
            fitness_atual = {tuple(rota): self._calcular_fitness(rota) for rota in self.populacao}
            melhor_rota_geracao = max(fitness_atual, key=fitness_atual.get)
            distancia_melhor = 1 / fitness_atual[melhor_rota_geracao]
            historico_distancias.append(distancia_melhor)
            
            if (i + 1) % 10 == 0:
                print(f"Geração {i+1:4d} | Melhor Distância: {distancia_melhor:.2f}")

        # Ao final, encontra a melhor rota da última população
        fitness_final = {tuple(rota): self._calcular_fitness(rota) for rota in self.populacao}
        melhor_rota_final = max(fitness_final, key=fitness_final.get)
        
        return list(melhor_rota_final), historico_distancias

def plotar_rota(cidades, rota, historico_distancias):
    """Plota a melhor rota e o gráfico de convergência."""
    plt.figure(figsize=(12, 6))

    # Gráfico 1: Rota
    plt.subplot(1, 2, 1)
    x_rota = [cidade.x for cidade in rota] + [rota[0].x]
    y_rota = [cidade.y for cidade in rota] + [rota[0].y]
    plt.plot(x_rota, y_rota, 'o-')
    
    for i, cidade in enumerate(cidades):
        plt.text(cidade.x, cidade.y, f' {i}')
        
    plt.title("Melhor Rota Encontrada")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")

    # Gráfico 2: Convergência
    plt.subplot(1, 2, 2)
    plt.plot(historico_distancias)
    plt.title("Convergência do Algoritmo")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Distância")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Parâmetros ---
    NUM_CIDADES = 20
    TAM_POPULACAO = 100
    TAXA_MUTACAO = 0.01
    TAXA_CROSSOVER = 0.9
    NUM_GERACOES = 100

    cidades = [Cidade(random.randint(0, 200), random.randint(0, 200)) for _ in range(NUM_CIDADES)]

    # Instancia e executa o resolvedor
    ag_tsp = ResolvedorTSP_AG(
        cidades=cidades,
        tam_populacao=TAM_POPULACAO,
        taxa_mutacao=TAXA_MUTACAO,
        taxa_crossover=TAXA_CROSSOVER,
        num_geracoes=NUM_GERACOES
    )
    
    melhor_rota, historico = ag_tsp.encontrar_melhor_rota()
    
    melhor_distancia = 1 / ag_tsp._calcular_fitness(melhor_rota)
    print("\n" + "="*50)
    print("      RESULTADO FINAL DO ALGORITMO GENÉTICO      ")
    print("="*50)
    print(f"Melhor distância encontrada: {melhor_distancia:.2f}")
    
    plotar_rota(cidades, melhor_rota, historico)
