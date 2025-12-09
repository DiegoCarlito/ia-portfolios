import heapq
import time

class No:
    """
    Representa um nó na árvore de busca para o 8-Puzzle.
    Além do estado, pai e ação, armazena g(n) - o custo do caminho.
    """
    def __init__(self, estado, pai, acao, custo_g):
        self.estado = estado
        self.pai = pai
        self.acao = acao
        self.custo_g = custo_g

    def __lt__(self, other):
        """Comparador para a fila de prioridade."""
        return False

class ResolvedorPuzzle:
    """
    Classe principal para resolver o 8-Puzzle.
    Contém a lógica para as buscas informadas e as funções heurísticas.
    """
    def __init__(self, estado_inicial, estado_objetivo):
        self.estado_inicial = estado_inicial
        self.estado_objetivo = estado_objetivo
        self._posicoes_objetivo = self._calcular_posicoes_objetivo()

    def _calcular_posicoes_objetivo(self):
        """Cria um dicionário mapeando cada peça à sua posição no estado objetivo."""
        posicoes = {}
        for i, linha in enumerate(self.estado_objetivo):
            for j, peca in enumerate(linha):
                if peca != 0:
                    posicoes[peca] = (i, j)
        return posicoes

    def h_pecas_fora_lugar(self, estado):
        """Heurística 1: Número de Peças Fora do Lugar."""
        erros = 0
        for i in range(3):
            for j in range(3):
                peca_atual = estado[i][j]
                if peca_atual != 0 and peca_atual != self.estado_objetivo[i][j]:
                    erros += 1
        return erros

    def h_distancia_manhattan(self, estado):
        """Heurística 2: Distância de Manhattan."""
        distancia = 0
        for i in range(3):
            for j in range(3):
                peca = estado[i][j]
                if peca != 0:
                    pos_objetivo = self._posicoes_objetivo[peca]
                    distancia += abs(i - pos_objetivo[0]) + abs(j - pos_objetivo[1])
        return distancia

    def _obter_sucessores(self, estado):
        """Encontra os estados sucessores trocando o vazio com seus vizinhos."""
        sucessores = []
        pos_vazio = None
        for i in range(3):
            for j in range(3):
                if estado[i][j] == 0:
                    pos_vazio = (i, j)
                    break
            if pos_vazio:
                break
        
        r, c = pos_vazio
        movimentos = [("CIMA", r - 1, c), ("BAIXO", r + 1, c),
                      ("ESQUERDA", r, c - 1), ("DIREITA", r, c + 1)]

        for acao, nr, nc in movimentos:
            if 0 <= nr < 3 and 0 <= nc < 3:
                novo_estado_lista = [list(linha) for linha in estado]
                novo_estado_lista[r][c], novo_estado_lista[nr][nc] = novo_estado_lista[nr][nc], novo_estado_lista[r][c]
                sucessores.append((acao, tuple(map(tuple, novo_estado_lista))))
        return sucessores

    def resolver(self, algoritmo, heuristica):
        """Executa o algoritmo de busca informado escolhido."""
        if heuristica == 'manhattan':
            h = self.h_distancia_manhattan
        else:
            h = self.h_pecas_fora_lugar

        no_inicial = No(self.estado_inicial, None, None, 0)
        fronteira = [(h(self.estado_inicial), no_inicial)]
        explorados = {self.estado_inicial: 0}
        self.nos_explorados = 0

        while fronteira:
            self.nos_explorados += 1
            prioridade, no_atual = heapq.heappop(fronteira)

            if no_atual.estado == self.estado_objetivo:
                return self._reconstruir_caminho(no_atual)

            for acao, estado_sucessor in self._obter_sucessores(no_atual.estado):
                novo_custo_g = no_atual.custo_g + 1
                
                if estado_sucessor in explorados and novo_custo_g >= explorados[estado_sucessor]:
                    continue
                
                explorados[estado_sucessor] = novo_custo_g
                novo_no = No(estado_sucessor, no_atual, acao, novo_custo_g)
                
                if algoritmo == 'a_estrela':
                    f_n = novo_custo_g + h(estado_sucessor)
                else: # Gulosa
                    f_n = h(estado_sucessor)
                
                heapq.heappush(fronteira, (f_n, novo_no))
                
        return None

    def _reconstruir_caminho(self, no_final):
        """Reconstrói a lista de ações do final para o início."""
        acoes = []
        no_atual = no_final
        while no_atual.pai is not None:
            acoes.append(no_atual.acao)
            no_atual = no_atual.pai
        acoes.reverse()
        return acoes

def imprimir_estado(estado):
    """Função auxiliar para imprimir o tabuleiro de forma legível."""
    print("-" * 7)
    for linha in estado:
        print("|", " ".join(map(str, linha)).replace('0', '_'), "|")
    print("-" * 7)

def exibir_solucao_passo_a_passo(resolvedor, estado_inicial, acoes):
    """
    Aplica a sequência de ações e exibe o estado do tabuleiro a cada passo.
    """
    print("\n" + "-"*25)
    print("VISUALIZAÇÃO DA SOLUÇÃO")
    print("-" * 25)
    
    estado_atual = estado_inicial
    print("Passo 0: Estado Inicial")
    imprimir_estado(estado_atual)

    for i, acao in enumerate(acoes):
        print(f"\nPasso {i + 1}: Ação -> {acao}")
        # Encontra o próximo estado aplicando a ação
        sucessores = resolvedor._obter_sucessores(estado_atual)
        for a, s in sucessores:
            if a == acao:
                estado_atual = s
                break
        imprimir_estado(estado_atual)

if __name__ == "__main__":
    ESTADO_OBJETIVO = ((1, 2, 3), (4, 5, 6), (7, 8, 0))
    ESTADO_INICIAL = ((1, 2, 3), (0, 4, 6), (7, 5, 8))

    print("="*50)
    print("RESOLVENDO O QUEBRA-CABEÇA DE 8 PEÇAS")
    print("="*50)
    print("Estado Inicial:")
    imprimir_estado(ESTADO_INICIAL)
    print("\nEstado Objetivo:")
    imprimir_estado(ESTADO_OBJETIVO)
    
    resolvedor = ResolvedorPuzzle(ESTADO_INICIAL, ESTADO_OBJETIVO)

    algoritmos_para_testar = ['gulosa', 'a_estrela']
    
    for alg in algoritmos_para_testar:
        print("\n" + "="*50)
        print(f"--- Executando: Busca {alg.replace('_', '*').title()} (Heurística: Manhattan) ---")
        print("="*50)
        
        inicio = time.time()
        solucao = resolvedor.resolver(algoritmo=alg, heuristica='manhattan')
        tempo = time.time() - inicio
        
        if solucao:
            print(f"✓ Solução encontrada!")
            print(f"  Profundidade (Passos): {len(solucao)}")
            print(f"  Nós explorados: {resolvedor.nos_explorados}")
            print(f"  Tempo: {tempo*1000:.2f}ms")
            print(f"  Sequência: {' → '.join(solucao)}")
            
            exibir_solucao_passo_a_passo(resolvedor, ESTADO_INICIAL, solucao)
        else:
            print("✗ Nenhuma solução encontrada.")
