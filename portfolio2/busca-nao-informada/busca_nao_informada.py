from collections import deque
import textwrap
import time

class No:
    """
    Representa um nó na árvore de busca. Um nó contém informações sobre
    um estado do problema, o nó que o precedeu (pai) e a ação que levou
    a este estado.
    """
    def __init__(self, estado, pai, acao):
        """
        Construtor da classe No.

        Args:
            estado (tuple): A coordenada (linha, coluna) no labirinto.
            pai (No): O nó que gerou este nó. None para o nó inicial.
            acao (str): A ação ('CIMA', 'BAIXO', etc.) que levou do pai a este nó.
        """
        self.estado = estado
        self.pai = pai
        self.acao = acao

class ResolvedorLabirinto:
    """
    Classe principal que encapsula a lógica para resolver um labirinto.
    Ela carrega o labirinto, encontra início e fim, e implementa
    os algoritmos de busca BFS e DFS.
    """
    def __init__(self, labirinto_str):
        """
        Inicializa o resolvedor de labirinto.

        Args:
            labirinto_str (str): Uma string multi-linhas representando o labirinto.
        
        Raises:
            ValueError: Se o labirinto não contiver um 'S' (início) e um 'E' (fim).
        """
        # Processa a string do labirinto, removendo espaços extras e quebrando em linhas.
        self.labirinto = labirinto_str.strip().splitlines()
        self.altura = len(self.labirinto)
        self.largura = len(self.labirinto[0])
        
        # Mapeia o labirinto para encontrar as posições de início e fim.
        self.inicio = self._encontrar_posicao('S')
        self.fim = self._encontrar_posicao('E')
        
        # Garante que o labirinto é válido.
        if not self.inicio or not self.fim:
            raise ValueError("Labirinto deve conter um 'S' (início) e um 'E' (fim).")

    def _encontrar_posicao(self, char):
        """
        Varre a matriz do labirinto para encontrar as coordenadas de um caractere.

        Args:
            char (str): O caractere a ser encontrado (ex: 'S' ou 'E').

        Returns:
            tuple: Uma tupla (linha, coluna) com as coordenadas, ou None se não for encontrado.
        """
        for i, linha in enumerate(self.labirinto):
            for j, coluna in enumerate(linha):
                if coluna == char:
                    return (i, j)
        return None

    def _obter_vizinhos(self, estado):
        """
        Calcula a função sucessora: encontra todos os vizinhos válidos de um estado.

        Args:
            estado (tuple): A coordenada (linha, coluna) atual.

        Returns:
            list: Uma lista de tuplas, onde cada tupla contém (acao, estado_vizinho).
        """
        linha, coluna = estado
        # Define as possíveis ações e os estados resultantes.
        candidatos = [
            ("CIMA", (linha - 1, coluna)),
            ("BAIXO", (linha + 1, coluna)),
            ("ESQUERDA", (linha, coluna - 1)),
            ("DIREITA", (linha, coluna + 1))
        ]
        
        vizinhos = []
        for acao, (r, c) in candidatos:
            # Verifica se o vizinho está dentro dos limites do labirinto.
            # E verifica se não é uma parede ('#').
            if 0 <= r < self.altura and 0 <= c < self.largura and self.labirinto[r][c] != '#':
                vizinhos.append((acao, (r, c)))
        return vizinhos

    def resolver(self, metodo='bfs'):
        """
        Método público que serve como interface para iniciar a busca.

        Args:
            metodo (str): O algoritmo a ser usado ('bfs' ou 'dfs').

        Returns:
            tuple: Uma tupla (acoes, celulas) representando o caminho, ou None se não houver solução.
        """
        if metodo == 'bfs':
            return self._buscar(use_bfs=True)
        elif metodo == 'dfs':
            return self._buscar(use_bfs=False)
        else:
            raise ValueError("Método de busca inválido. Use 'bfs' ou 'dfs'.")

    def _buscar(self, use_bfs):
        """
        Implementação do algoritmo de busca genérico.
        Ele utiliza uma Fila para BFS e uma Pilha para DFS.

        Args:
            use_bfs (bool): True para usar BFS (Fila), False para usar DFS (Pilha).

        Returns:
            tuple: O caminho da solução, ou None se não for encontrado.
        """
        # 1. Inicialização
        no_inicial = No(estado=self.inicio, pai=None, acao=None)
        
        # A fronteira armazena os nós a serem explorados.
        # A escolha da estrutura de dados aqui define o algoritmo.
        if use_bfs:
            fronteira = deque([no_inicial])  # Fila (First-In, First-Out)
        else:
            fronteira = [no_inicial]  # Pilha (Last-In, First-Out)

        # O conjunto de explorados armazena estados já visitados para evitar loops.
        explorados = {self.inicio}
        self.nos_explorados = 0

        # 2. Loop de Busca
        while fronteira:
            self.nos_explorados += 1
            
            # Remove um nó da fronteira de acordo com a estratégia (FIFO ou LIFO).
            if use_bfs:
                no_atual = fronteira.popleft()  # Pega o mais antigo (BFS)
            else:
                no_atual = fronteira.pop()  # Pega o mais recente (DFS)

            # 3. Teste de Objetivo
            if no_atual.estado == self.fim:
                # Se encontrou a solução, reconstrói e retorna o caminho.
                return self._reconstruir_caminho(no_atual)

            # 4. Expansão do Nó
            # Pega os vizinhos do nó atual.
            for acao, estado_vizinho in self._obter_vizinhos(no_atual.estado):
                # Se o vizinho ainda não foi explorado...
                if estado_vizinho not in explorados:
                    # ...adiciona ao conjunto de explorados...
                    explorados.add(estado_vizinho)
                    # ...cria um novo nó para ele...
                    novo_no = No(estado=estado_vizinho, pai=no_atual, acao=acao)
                    # ...e o adiciona à fronteira para ser explorado no futuro.
                    fronteira.append(novo_no)
        
        # Se o loop terminar e não houver retornado, não há solução.
        return None

    def _reconstruir_caminho(self, no_final):
        """
        Percorre o caminho de volta do nó final até o inicial usando os ponteiros 'pai'.

        Args:
            no_final (No): O nó que contém o estado objetivo.

        Returns:
            tuple: Uma tupla contendo (lista_de_acoes, lista_de_celulas).
        """
        acoes = []
        celulas = []
        no_atual = no_final
        # Enquanto não chegarmos ao nó inicial (que não tem pai)...
        while no_atual.pai is not None:
            # ...adiciona a ação e o estado à nossa lista.
            acoes.append(no_atual.acao)
            celulas.append(no_atual.estado)
            # ...e move para o nó pai.
            no_atual = no_atual.pai
        
        # As listas estão na ordem inversa (do fim para o começo), então as revertemos.
        acoes.reverse()
        celulas.reverse()
        return acoes, celulas

    def imprimir_solucao(self, celulas_caminho, metodo):
        """
        Exibe uma representação visual do labirinto com o caminho da solução.

        Args:
            celulas_caminho (list): A lista de coordenadas que formam o caminho.
            metodo (str): O nome do método ('BFS' ou 'DFS') para o título.
        """
        # Cria uma cópia do labirinto para não modificar o original.
        labirinto_solucao = [list(linha) for linha in self.labirinto]
        # Marca cada célula do caminho com um asterisco.
        for i, j in celulas_caminho:
            # Não sobrescreve 'S' e 'E'.
            if labirinto_solucao[i][j] not in ['S', 'E']:
                labirinto_solucao[i][j] = '*'
        
        print(f"\n--- Labirinto Resolvido ({metodo.upper()}) ---")
        for linha in labirinto_solucao:
            print("".join(linha))

# --- Bloco Principal de Execução ---
# Este bloco só é executado quando o script é rodado diretamente.
if __name__ == "__main__":
    
    # Define o labirinto a ser resolvido usando uma string multi-linhas.
    # textwrap.dedent é usado para remover a indentação comum, facilitando a formatação.
    labirinto_texto = textwrap.dedent("""\
        #########################################
        #S#   #       #           #           #
        # # # # ### ### ######### # ### ##### #
        #   #   #   # # #         #   # #   # #
        ##### ### # # # # ####### ### # # # # #
        #   # #   # #   # #     #     # # # # #
        # # # # ### ##### # ### # ##### ### # #
        # # #   #         # # # # #   # #   # #
        # # ##### ######### # # # # # # # ### #
        # #     # #         #   #   # #   #   #
        # ##### # # ######### ####### ####### #
        # #   # # #       # #   #     #       #
        # # ### # ##### # # # ### ### ####### #
        # #         #   #   #     #           E
        #########################################""")

    try:
        # Instancia a classe principal com o labirinto.
        resolvedor = ResolvedorLabirinto(labirinto_texto)
        
        # Imprime informações iniciais sobre o problema.
        print("="*50)
        print("RESOLVENDO LABIRINTO COM BFS E DFS")
        print("="*50)
        print(f"Início: {resolvedor.inicio}")
        print(f"Fim: {resolvedor.fim}")
        print(f"Dimensões: {resolvedor.altura}x{resolvedor.largura}")
        
        # --- Execução do BFS ---
        print("\n" + "="*50)
        print("--- Busca em Largura (BFS) ---")
        print("="*50)
        
        # Mede o tempo de início.
        inicio_bfs = time.time()
        solucao_bfs = resolvedor.resolver('bfs')
        # Calcula o tempo total.
        tempo_bfs = time.time() - inicio_bfs
        
        # Se uma solução foi encontrada, exibe as métricas.
        if solucao_bfs:
            acoes, celulas = solucao_bfs
            print(f"✓ Caminho encontrado!")
            print(f"  Passos: {len(acoes)}")
            print(f"  Nós explorados: {resolvedor.nos_explorados}")
            print(f"  Tempo: {tempo_bfs*1000:.2f}ms")
            # Mostra apenas as 5 primeiras ações para não poluir a saída.
            print(f"  Ações: {' → '.join(acoes[:5])}{'...' if len(acoes) > 5 else ''}")
            resolvedor.imprimir_solucao(celulas, 'BFS')
        else:
            print("✗ Nenhuma solução encontrada com BFS.")
        
        # --- Execução do DFS ---
        print("\n" + "="*50)
        print("--- Busca em Profundidade (DFS) ---")
        print("="*50)
        
        inicio_dfs = time.time()
        solucao_dfs = resolvedor.resolver('dfs')
        tempo_dfs = time.time() - inicio_dfs
        
        if solucao_dfs:
            acoes, celulas = solucao_dfs
            print(f"✓ Caminho encontrado!")
            print(f"  Passos: {len(acoes)}")
            print(f"  Nós explorados: {resolvedor.nos_explorados}")
            print(f"  Tempo: {tempo_dfs*1000:.2f}ms")
            print(f"  Ações: {' → '.join(acoes[:5])}{'...' if len(acoes) > 5 else ''}")
            resolvedor.imprimir_solucao(celulas, 'DFS')
        else:
            print("✗ Nenhuma solução encontrada com DFS.")
        
        # --- Seção de Comparação Final ---
        print("\n" + "="*50)
        print("COMPARAÇÃO BFS vs DFS")
        print("="*50)
        if solucao_bfs and solucao_dfs:
            passos_bfs = len(solucao_bfs[0])
            passos_dfs = len(solucao_dfs[0])
            print(f"BFS: {passos_bfs} passos (MENOR CAMINHO)")
            print(f"DFS: {passos_dfs} passos (pode ser maior)")
            print(f"Diferença: {abs(passos_dfs - passos_bfs)} passos")
        
    except ValueError as e:
        print(f"Erro: {e}")
