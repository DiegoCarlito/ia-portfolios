import random

class MundoWumpus:
    """
    Representa o AMBIENTE (o mundo real) que está oculto do agente.
    """
    def __init__(self, tamanho=4):
        self.tamanho = tamanho
        self.wumpus = self._posicao_aleatoria()
        self.ouro = self._posicao_aleatoria(evitar=[self.wumpus])
        # Cria 3 poços
        self.pocos = [self._posicao_aleatoria(evitar=[self.wumpus, self.ouro]) for _ in range(3)]
        
        # O agente sempre começa em (0, 0)
        if (0, 0) in self.pocos:
            self.pocos.remove((0, 0))
        if self.wumpus == (0, 0):
            self.wumpus = self._posicao_aleatoria(evitar=[(0,0), self.ouro] + self.pocos)
        
    def _posicao_aleatoria(self, evitar=None):
        if evitar is None: evitar = []
        while True:
            pos = (random.randint(0, self.tamanho - 1), random.randint(0, self.tamanho - 1))
            if pos not in evitar:
                return pos

    def _obter_adjacentes(self, pos):
        """Retorna os vizinhos válidos de uma posição."""
        r, c = pos
        candidatos = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return [(nr, nc) for nr, nc in candidatos if 0 <= nr < self.tamanho and 0 <= nc < self.tamanho]

    def obter_percepcoes(self, pos):
        """Retorna as percepções do agente na posição (pos)."""
        fedor = False
        brisa = False
        brilho = False

        if pos == self.ouro:
            brilho = True

        for adj in self._obter_adjacentes(pos):
            if adj == self.wumpus:
                fedor = True
            if adj in self.pocos:
                brisa = True
                
        return (fedor, brisa, brilho)
        
    def verificar_morte(self, pos):
        """Verifica se o agente morreu ao entrar nesta casa."""
        return pos == self.wumpus or pos in self.pocos

class AgenteLogico:
    """
    Representa o AGENTE, com sua Base de Conhecimento (KB).
    """
    def __init__(self, tamanho):
        self.tamanho = tamanho
        # A Base de Conhecimento (KB) armazena fatos conhecidos.
        # Fatos são strings: "-P(1,2)" (Sem Poço), "W(2,3)" (Wumpus), "OK(1,1)" (Segura)
        self.kb = set()
        self.visitados = set()
        self.posicao_atual = (0, 0)
        self.tem_ouro = False
        # A fronteira é uma pilha (LIFO) de casas que o agente sabe que são seguras
        # e que ele pretende visitar.
        self.fronteira_segura = []
        
        # Conhecimento inicial: (0,0) é seguro.
        self._adicionar_facto_kb(f"OK(0,0)")

    def _adicionar_facto_kb(self, facto):
        """Adiciona um fato à Base de Conhecimento e imprime (TELL)."""
        if facto not in self.kb:
            # print(f"  [KB] Agente aprendeu: {facto}")
            self.kb.add(facto)

    def _obter_adjacentes(self, pos):
        """Retorna os vizinhos válidos de uma posição."""
        r, c = pos
        candidatos = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return [(nr, nc) for nr, nc in candidatos if 0 <= nr < self.tamanho and 0 <= nc < self.tamanho]

    def _inferir(self, pos, percepcoes):
        """
        O coração do agente. Usa percepções para inferir novos fatos
        sobre o mundo e os adiciona à KB.
        """
        fedor, brisa, brilho = percepcoes
        adjacentes = self._obter_adjacentes(pos)
        
        # --- Regra 1: Inferência de Segurança ---
        # Se não há fedor e não há brisa, todas as casas adjacentes
        # são seguras (sem Wumpus e sem Poços).
        if not fedor and not brisa:
            print(f"  [Inferência] Posição {pos} é limpa. Inferindo que vizinhos são seguros.")
            for (r, c) in adjacentes:
                self._adicionar_facto_kb(f"-P({r},{c})") # Não é Poço
                self._adicionar_facto_kb(f"-W({r},{c})") # Não é Wumpus
                
        # --- Regra 2: Inferência sobre Poços (Simples) ---
        if brisa:
            print(f"  [Inferência] Sentiu BRISA em {pos}. Um vizinho tem um poço.")
        else: # Se não sentiu brisa, todos os vizinhos NÃO têm poços.
            for (r, c) in adjacentes:
                self._adicionar_facto_kb(f"-P({r},{c})")

        # --- Regra 3: Inferência sobre Wumpus (Simples) ---
        if fedor:
            print(f"  [Inferência] Sentiu FEDOR em {pos}. Um vizinho tem o Wumpus.")
        else: # Se não sentiu fedor, todos os vizinhos NÃO têm o Wumpus.
            for (r, c) in adjacentes:
                self._adicionar_facto_kb(f"-W({r},{c})")
        
        # --- Regra 4: Definição de "OK" ---
        # Uma casa (r,c) é "OK" (segura) se sabemos que não há poço E não há wumpus.
        for (r, c) in adjacentes:
            if f"-P({r},{c})" in self.kb and f"-W({r},{c})" in self.kb:
                self._adicionar_facto_kb(f"OK({r},{c})")

    def escolher_proxima_acao(self):
        """
        Processo de decisão (ASK da KB):
        1. Se tem o ouro, sai do jogo (simplificado).
        2. Adiciona vizinhos seguros e não visitados à fronteira.
        3. Pega uma casa da fronteira para ser o próximo movimento.
        """
        if self.tem_ouro:
            return "SAIR"
            
        # 1. Adiciona vizinhos recém-descobertos e seguros à fronteira
        for (r, c) in self._obter_adjacentes(self.posicao_atual):
            if f"OK({r},{c})" in self.kb and (r, c) not in self.visitados:
                if (r, c) not in self.fronteira_segura:
                    print(f"  [Decisão] Casa ({r},{c}) é segura e será explorada.")
                    self.fronteira_segura.append((r, c))

        # 2. Pega a próxima casa segura da fronteira (estilo DFS/Pilha)
        if self.fronteira_segura:
            proxima_pos = self.fronteira_segura.pop()
            return ("MOVER", proxima_pos)
        else:
            # Se a fronteira está vazia e não achamos o ouro, o agente está preso.
            return "DESISTIR"

    def executar_passo(self, mundo):
        """Executa um ciclo completo de Percepção-Inferência-Ação."""
        
        print(f"\n--- Turno do Agente ---")
        print(f"Agente está em {self.posicao_atual}")
        self.visitados.add(self.posicao_atual)

        # 1. PERCEBER o ambiente
        percepcoes = mundo.obter_percepcoes(self.posicao_atual)
        fedor, brisa, brilho = percepcoes
        print(f"Agente percebe: Fedor={fedor}, Brisa={brisa}, Brilho={brilho}")
        
        # 2. ATUALIZAR KB E INFERIR (TELL)
        self._inferir(self.posicao_atual, percepcoes)

        # 3. VERIFICAR BRILHO (objetivo)
        if brilho:
            self.tem_ouro = True
            print(f"  [Ação] Pegou o Ouro em {self.posicao_atual}!")
            # Em um agente real, ele traçaria o caminho de volta para (0,0).
            # Vamos simplificar e apenas sair.
            return "VITORIA"

        # 4. ESCOLHER AÇÃO (ASK)
        acao = self.escolher_proxima_acao()
        
        if acao == "DESISTIR":
            print(f"  [Ação] Agente está preso e não há mais casas seguras. Desistindo.")
            return "DERROTA"
        elif acao == "VITORIA":
            return "VITORIA"
        
        elif acao[0] == "MOVER":
            self.posicao_atual = acao[1]
            print(f"  [Ação] Agente move-se para {self.posicao_atual}")
            # Verifica se a inferência estava errada (o que não deve acontecer)
            # ou se ele foi para uma casa que *pensava* ser segura.
            if mundo.verificar_morte(self.posicao_atual):
                print(f"  [MORTE] Agente entrou em {self.posicao_atual} e morreu.")
                if self.posicao_atual == mundo.wumpus: print("Foi pego pelo Wumpus!")
                if self.posicao_atual in mundo.pocos: print("Caiu em um poço!")
                return "DERROTA"
        
        return "CONTINUAR"

if __name__ == "__main__":
    TAMANHO_MUNDO = 4
    mundo = MundoWumpus(TAMANHO_MUNDO)
    agente = AgenteLogico(TAMANHO_MUNDO)
    
    print("="*40)
    print("      INICIANDO O MUNDO DE WUMPUS      ")
    print("="*40)
    print("O MUNDO REAL (OCULTO PARA O AGENTE):")
    print(f"Wumpus: {mundo.wumpus}")
    print(f"Ouro: {mundo.ouro}")
    print(f"Poços: {mundo.pocos}")
    
    status = "CONTINUAR"
    while status == "CONTINUAR":
        status = agente.executar_passo(mundo)

    print("\n" + "="*40)
    print("      FIM DE JOGO      ")
    print(f"Resultado: {status}")
    print(f"Casas visitadas: {agente.visitados}")
    print("="*40)
