import time

class ResolvedorNRainhas:
    """
    Classe que encapsula a lógica do resolvedor de N-Rainhas usando backtracking.
    """
    def __init__(self, n):
        """
        Inicializa o resolvedor.

        Args:
            n (int): O tamanho do tabuleiro (N x N).
        """
        self.n = n
        # O 'tabuleiro' é a nossa estrutura de atribuição.
        # O índice da lista representa a COLUNA (Variável C_i).
        # O valor em self.posicoes[i] representa a LINHA (Valor do Domínio).
        # Inicializamos com -1 (não atribuído).
        self.posicoes = [-1] * n
        self.nos_visitados = 0

    def _eh_seguro(self, linha_proposta, col_proposta):
        """
        Verifica as restrições.
        Checa se uma rainha na (linha_proposta, col_proposta) conflita com
        qualquer rainha já colocada nas colunas anteriores (0 a col_proposta - 1).
        """
        # Itera por todas as colunas anteriores
        for col_existente in range(col_proposta):
            linha_existente = self.posicoes[col_existente]

            # 1. Restrição de Linha: Duas rainhas na mesma linha
            if linha_existente == linha_proposta:
                return False
            
            # 2. Restrição de Diagonal:
            # abs(linha1 - linha2) == abs(col1 - col2)
            if abs(linha_existente - linha_proposta) == abs(col_existente - col_proposta):
                return False
        
        # Se passou por todas as verificações, a posição é segura
        return True

    def _resolver_csp_util(self, col):
        """
        A função recursiva de backtracking.
        Tenta atribuir uma linha para a coluna 'col'.
        """
        self.nos_visitados += 1
        
        # --- CASO BASE ---
        # Se 'col' for igual a N, significa que conseguimos posicionar
        # rainhas em todas as colunas de 0 a N-1.
        if col == self.n:
            return True  # Sucesso!

        # --- PASSO RECURSIVO ---
        # Itera por todos os valores do domínio (linhas 0 a N-1)
        for linha in range(self.n):
            
            # 1. Verifica Restrições
            if self._eh_seguro(linha, col):
                
                # 2. Atribui o valor (Variável C_col = linha)
                self.posicoes[col] = linha
                
                # 3. Chama recursivamente para a próxima variável (próxima coluna)
                if self._resolver_csp_util(col + 1):
                    return True  # Sucesso encontrado!
                
                # 4. BACKTRACK
                # Se a chamada recursiva falhou, desfaz a atribuição.
                # (Em Python, podemos apenas deixar o loop continuar e
                # sobrescrever self.posicoes[col] na próxima iteração,
                # ou explicitamente resetar para -1)
                self.posicoes[col] = -1 
        
        # Se o loop terminou sem encontrar uma linha segura, retorna falha.
        return False

    def resolver(self):
        """
        Função pública para iniciar a busca.
        Começa a atribuição a partir da coluna 0.
        """
        if self._resolver_csp_util(col=0):
            return self.posicoes
        else:
            return None # Nenhuma solução encontrada

    def imprimir_solucao(self, solucao):
        """Imprime o tabuleiro de forma visual."""
        print(f"\n--- Solução para N = {self.n} ---")
        for linha in range(self.n):
            linha_str = ""
            for col in range(self.n):
                if solucao[col] == linha:
                    linha_str += " Q "
                else:
                    linha_str += " . "
            print(linha_str)
        print("-" * (self.n * 3))

if __name__ == "__main__":
    try:
        n_str = input("Digite o tamanho do tabuleiro (N): ")
        N = int(n_str)
        if N <= 3 and N != 1:
            print(f"Não existe solução para N={N}. Tente um valor maior.")
        elif N <= 0:
             print("O número deve ser positivo.")
        else:
            print(f"Resolvendo o problema das {N}-Rainhas...")
            
            resolvedor = ResolvedorNRainhas(N)
            
            inicio = time.time()
            solucao = resolvedor.resolver()
            tempo = time.time() - inicio
            
            if solucao:
                resolvedor.imprimir_solucao(solucao)
                print("\n" + "="*30)
                print("MÉTRICAS DA BUSCA")
                print("="*30)
                print(f"Solução encontrada (coluna: linha): {list(enumerate(solucao))}")
                print(f"Nós (estados) visitados: {resolvedor.nos_visitados}")
                print(f"Tempo de execução: {tempo*1000:.2f} ms")
            else:
                print(f"Nenhuma solução foi encontrada para N={N}.")

    except ValueError:
        print("Entrada inválida. Por favor, digite um número inteiro.")
