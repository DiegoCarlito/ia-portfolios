import numpy as np
from hmmlearn import hmm

class AnalisadorMercadoHMM:
    def __init__(self, n_estados=3):
        """
        Inicializa o analisador HMM com parâmetros de Especialista.
        """
        print("=" * 70)
        print("    ANALISADOR DE REGIMES DE MERCADO COM HMM (Viterbi)")
        print("=" * 70)
        
        self.n_estados = n_estados
        self.mapa_estados = {0: "Bear (Baixa)", 1: "Lateral", 2: "Bull (Alta)"}
        self.mapa_obs = {
            0: "Queda Forte", 1: "Queda Leve", 2: "Neutro", 
            3: "Alta Leve", 4: "Alta Forte"
        }
        
        # Construção do modelo manual (Sistema Especialista)
        self.model = self._construir_modelo_especialista()
        print("[OK] Modelo HMM inicializado.\n")
        self._imprimir_parametros_modelo()

    def _construir_modelo_especialista(self):
        """
        Define manualmente as probabilidades de um HMM Categórico.
        Estados: 0=Bear, 1=Lateral, 2=Bull
        Observações: 0=QuedaForte ... 4=AltaForte
        """
        model = hmm.CategoricalHMM(n_components=3, n_iter=100, init_params="")

        # 1. Probabilidades Iniciais (Viés para mercado estável)
        model.startprob_ = np.array([0.15, 0.70, 0.15])

        # 2. Matriz de Transição (Inércia de Mercado)
        # Probabilidade de P(Estado_t | Estado_t-1)
        model.transmat_ = np.array([
            [0.85, 0.12, 0.03],  # Bear tende a continuar Bear
            [0.08, 0.84, 0.08],  # Lateral tende a continuar Lateral
            [0.03, 0.12, 0.85]   # Bull tende a continuar Bull
        ])

        # 3. Matriz de Emissão (O que vemos dado o regime)
        # Probabilidade de P(Observação | Estado)
        model.emissionprob_ = np.array([
            # Q.Forte  Q.Leve  Neutro  A.Leve  A.Forte
            [0.40,    0.30,   0.20,   0.08,   0.02],  # Bear
            [0.05,    0.20,   0.50,   0.20,   0.05],  # Lateral
            [0.02,    0.08,   0.20,   0.30,   0.40]   # Bull
        ])

        return model

    def _imprimir_parametros_modelo(self):
        """Exibe as matrizes de probabilidade no terminal."""
        print("--- Matriz de Transição (Dinâmica do Mercado) ---")
        print(f"{'':<10} {'Para Bear':<10} {'Para Lat.':<10} {'Para Bull':<10}")
        for i in range(3):
            print(f"De {self.mapa_estados[i][:4]:<7} | {self.model.transmat_[i,0]:.2f}       {self.model.transmat_[i,1]:.2f}       {self.model.transmat_[i,2]:.2f}")
        print()

    def _discretizar_retornos(self, retornos):
        """Converte retornos contínuos (%) em 5 categorias discretas."""
        obs = []
        for r in retornos:
            if r < -1.5:   obs.append(0) # Queda Forte
            elif r < -0.2: obs.append(1) # Queda Leve
            elif r < 0.2:  obs.append(2) # Neutro
            elif r < 1.5:  obs.append(3) # Alta Leve
            else:          obs.append(4) # Alta Forte
        return obs

    def analisar_sequencia(self, sequencia_obs):
        """Infere os estados ocultos usando algoritmo de Viterbi."""
        obs_array = np.array(sequencia_obs).reshape(-1, 1)
        logprob, estados = self.model.decode(obs_array, algorithm="viterbi")
        return estados

    def calcular_metricas(self, precos, retornos, estados):
        """Calcula estatísticas descritivas por regime."""
        metricas = {}
        for estado in range(self.n_estados):
            mask = estados == estado
            if np.sum(mask) > 0:
                # Filtra retornos correspondentes a este estado
                # Ajuste de índice: retornos tem len = len(precos)-1
                ret_regime = np.array(retornos)[mask]
                metricas[estado] = {
                    'media': np.mean(ret_regime),
                    'volatilidade': np.std(ret_regime),
                    'dias': np.sum(mask),
                    'proporcao': np.sum(mask) / len(estados) * 100
                }
        return metricas

    def gerar_relatorio(self, precos, retornos, observacoes, estados):
        """Gera relatório textual completo no terminal."""
        print("\n" + "=" * 70)
        print("                    RELATÓRIO DE INFERÊNCIA (VITERBI)")
        print("=" * 70)
        
        # 1. Estatísticas por Regime
        metricas = self.calcular_metricas(precos, retornos, estados)
        
        print("📊 ESTATÍSTICAS DOS REGIMES DETECTADOS:\n")
        for estado in range(self.n_estados):
            if estado in metricas:
                m = metricas[estado]
                nome = self.mapa_estados[estado]
                # Barra de progresso ASCII para proporção
                barra = "█" * int(m['proporcao'] / 5)
                print(f"Regime: {nome.upper()}")
                print(f"  • Frequência:    {m['proporcao']:5.1f}%  [{barra:<20}]")
                print(f"  • Retorno Médio: {m['media']:+5.2f}%")
                print(f"  • Volatilidade:  {m['volatilidade']:5.2f}%")
                print("-" * 40)

        # 2. Sequência Temporal
        print("\n📅 LOG DE TRANSAÇÕES E DETECÇÃO DE ESTADO:\n")
        print(f"{'DIA':<4} | {'RETORNO':<9} | {'OBSERVAÇÃO':<12} | {'ESTADO OCULTO (HMM)':<20} | {'PREÇO'}")
        print("-" * 70)
        
        for i, (obs, est, ret) in enumerate(zip(observacoes, estados, retornos)):
            preco = precos[i+1]
            
            # Formatação condicional para o estado
            estado_str = self.mapa_estados[est]
            if est == 0: flag = "🐻"
            elif est == 2: flag = "🐂"
            else: flag = "  "
            
            # Marca transições de regime
            transicao = ""
            if i > 0 and estados[i] != estados[i-1]:
                transicao = f"<< MUDANÇA DE REGIME"

            print(f"{i+1:<4} | {ret:>+6.2f}%  | {self.mapa_obs[obs]:<12} | {flag} {estado_str:<15} | ${preco:>6.2f} {transicao}")
        
        print("=" * 70)

def simular_mercado(dias=50):
    """Gera dados sintéticos com regimes pré-definidos para teste."""
    np.random.seed(42)
    # Regimes reais (Ground Truth): Lateral -> Bull -> Lateral -> Bear -> Lateral
    regimes = [1]*10 + [2]*15 + [1]*5 + [0]*10 + [1]*10
    
    preco = 100.0
    precos = [preco]
    retornos = []
    
    for r in regimes:
        if r == 0:   # Bear
            ret = np.random.normal(-1.0, 2.0)
        elif r == 1: # Lateral
            ret = np.random.normal(0.0, 0.5)
        else:        # Bull
            ret = np.random.normal(0.8, 1.5)
            
        retornos.append(ret)
        preco = preco * (1 + ret/100)
        precos.append(preco)
        
    return precos, retornos, regimes

if __name__ == "__main__":
    # 1. Configuração
    analisador = AnalisadorMercadoHMM()
    
    # 2. Obtenção de Dados (Simulados)
    precos, retornos, regimes_reais = simular_mercado(dias=50)
    
    # 3. Processamento (Discretização)
    observacoes = analisador._discretizar_retornos(retornos)
    
    # 4. Inferência (Viterbi)
    estados_inferidos = analisador.analisar_sequencia(observacoes)
    
    # 5. Relatório
    analisador.gerar_relatorio(precos, retornos, observacoes, estados_inferidos)
    
    # 6. Validação (Apenas para controle)
    acuracia = np.mean(estados_inferidos == np.array(regimes_reais))
    print(f"\n[VALIDAÇÃO] Acurácia do modelo em relação ao cenário simulado: {acuracia:.1%}")
