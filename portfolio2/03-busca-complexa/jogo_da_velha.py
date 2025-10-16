import math
import time

class JogoDaVelha:
    """
    Classe que gerencia o estado e as regras do Jogo da Velha.
    """
    def __init__(self):
        self.tabuleiro = [' ' for _ in range(9)]
        self.vencedor = None

    def imprimir_tabuleiro(self):
        """Imprime o tabuleiro atual no console."""
        print("")
        for i in range(3):
            print(" | ".join(self.tabuleiro[i*3:(i+1)*3]))
            if i < 2:
                print("---------")
        print("")

    def obter_acoes_possiveis(self):
        """Retorna uma lista de posições (0-8) que estão vazias."""
        return [i for i, spot in enumerate(self.tabuleiro) if spot == ' ']

    def fazer_jogada(self, posicao, jogador):
        """Faz uma jogada se a posição estiver livre."""
        if self.tabuleiro[posicao] == ' ':
            self.tabuleiro[posicao] = jogador
            if self.verificar_vitoria(posicao, jogador):
                self.vencedor = jogador
            return True
        return False

    def verificar_vitoria(self, posicao, jogador):
        """Verifica se a jogada na 'posicao' resultou em vitória para o 'jogador'."""
        linha_idx = posicao // 3
        linha = self.tabuleiro[linha_idx*3 : (linha_idx+1)*3]
        if all([spot == jogador for spot in linha]):
            return True
        
        col_idx = posicao % 3
        coluna = [self.tabuleiro[col_idx + i*3] for i in range(3)]
        if all([spot == jogador for spot in coluna]):
            return True
            
        if posicao % 2 == 0:
            diagonal1 = [self.tabuleiro[i] for i in [0, 4, 8]]
            if all([spot == jogador for spot in diagonal1]):
                return True
            diagonal2 = [self.tabuleiro[i] for i in [2, 4, 6]]
            if all([spot == jogador for spot in diagonal2]):
                return True
        
        return False

    def tabuleiro_cheio(self):
        """Verifica se não há mais espaços vazios."""
        return ' ' not in self.tabuleiro

# --- NOVA FUNÇÃO AUXILIAR ---
def imprimir_tabuleiro_referencia():
    """Imprime o tabuleiro com os números de 1 a 9 para referência do jogador."""
    print("Posições do Tabuleiro (1-9):")
    referencia = [[str(i) for i in range(j*3 + 1, (j+1)*3 + 1)] for j in range(3)]
    for linha in referencia:
        print(" | ".join(linha))
        if referencia.index(linha) < 2:
            print("---------")
    print("")

def minimax(estado_jogo, jogador_atual, jogador_ia):
    """
    Algoritmo Minimax para determinar a melhor jogada.
    """
    oponente = 'O' if jogador_ia == 'X' else 'X'
    
    if estado_jogo.vencedor == oponente:
        return {'posicao': None, 'pontuacao': 1 * (len(estado_jogo.obter_acoes_possiveis()) + 1) if oponente == jogador_ia else -1 * (len(estado_jogo.obter_acoes_possiveis()) + 1)}
    elif estado_jogo.vencedor == jogador_ia:
        return {'posicao': None, 'pontuacao': 1 * (len(estado_jogo.obter_acoes_possiveis()) + 1) if jogador_ia == jogador_ia else -1 * (len(estado_jogo.obter_acoes_possiveis()) + 1)}
    elif estado_jogo.tabuleiro_cheio():
        return {'posicao': None, 'pontuacao': 0}

    if jogador_atual == jogador_ia:
        melhor = {'posicao': None, 'pontuacao': -math.inf}
    else:
        melhor = {'posicao': None, 'pontuacao': math.inf}

    for jogada_possivel in estado_jogo.obter_acoes_possiveis():
        estado_jogo.fazer_jogada(posicao=jogada_possivel, jogador=jogador_atual)
        proximo_jogador = oponente if jogador_atual == jogador_ia else jogador_ia
        simulacao = minimax(estado_jogo, proximo_jogador, jogador_ia)
        
        estado_jogo.tabuleiro[jogada_possivel] = ' '
        estado_jogo.vencedor = None
        
        simulacao['posicao'] = jogada_possivel
        
        if jogador_atual == jogador_ia:
            if simulacao['pontuacao'] > melhor['pontuacao']:
                melhor = simulacao
        else:
            if simulacao['pontuacao'] < melhor['pontuacao']:
                melhor = simulacao
                
    return melhor

def jogar():
    """Função principal que gerencia o fluxo do jogo."""
    jogo = JogoDaVelha()
    humano = 'O'
    ia = 'X'
    
    print("="*30)
    print("JOGO DA VELHA com IA (Minimax)")
    print("="*30)
    print("Você joga como 'O'. A IA joga como 'X'.")
    
    # --- MUDANÇA: Imprime o tabuleiro de referência ---
    imprimir_tabuleiro_referencia()
    
    while True:
        # Turno do Humano
        if not jogo.tabuleiro_cheio() and jogo.vencedor is None:
            try:
                jogada_str = input(f"Sua jogada (1-9): ")
                if not jogada_str.isdigit():
                    print("Entrada inválida. Digite um número de 1 a 9.")
                    continue
                
                jogada_humano = int(jogada_str)
                
                # Converte a jogada 1-9 para o índice 0-8 ---
                pos = jogada_humano - 1

                # --- Validação da posição ---
                if not (0 <= pos <= 8 and pos in jogo.obter_acoes_possiveis()):
                    print("Posição inválida ou ocupada. Tente novamente.")
                    continue
                
                jogo.fazer_jogada(pos, humano)
                
            except ValueError:
                print("Entrada inválida. Digite um número de 1 a 9.")
                continue
        
        if jogo.vencedor or jogo.tabuleiro_cheio():
            break

        # Turno da IA
        print("\nTurno da IA ('X')...")
        time.sleep(0.5)
        melhor_jogada = minimax(jogo, ia, ia)
        jogo.fazer_jogada(melhor_jogada['posicao'], ia)
        
        jogo.imprimir_tabuleiro()

        if jogo.vencedor or jogo.tabuleiro_cheio():
            break

    # Fim de jogo
    print("\n--- FIM DE JOGO ---")
    jogo.imprimir_tabuleiro()
    if jogo.vencedor:
        print(f"O jogador '{jogo.vencedor}' venceu!")
    else:
        print("Deu empate!")

if __name__ == "__main__":
    jogar()
