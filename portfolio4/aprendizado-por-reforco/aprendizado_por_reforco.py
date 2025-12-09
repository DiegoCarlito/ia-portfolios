import gymnasium as gym
import numpy as np
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hiperparâmetros de RL
        self.memory = deque(maxlen=2000) # Buffer de Experience Replay
        self.gamma = 0.95    # Fator de desconto (visão de futuro)
        self.epsilon = 1.0   # Taxa de exploração inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.update_target_freq = 10 # Frequência de atualização da rede alvo
        
        # Construção das Redes Neurais (Main e Target)
        # Main: Treinada a cada passo (Online Network)
        self.model = self._build_model()
        # Target: Usada para calcular o erro estável, atualizada lentamente
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Constrói a Rede Neural que aproxima a função Q(s,a)
        model = models.Sequential()
        # Entrada: Estado do ambiente (4 variáveis contínuas)
        model.add(layers.Input(shape=(self.state_size,)))
        # Camadas Ocultas
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        # Saída: Q-value para cada ação possível (Esquerda/Direita)
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copia os pesos da rede principal para a rede alvo
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Armazena a experiência no buffer (Experience Replay)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Política Epsilon-Greedy: Escolhe entre explorar ou explotar
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Ação Aleatória (Exploração)
        
        # Ação baseada no conhecimento atual (Explotação)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        # Treina a rede neural usando amostras aleatórias da memória
        if len(self.memory) < self.batch_size:
            return

        # Amostragem aleatória
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepara os batches para processamento vetorizado
        states = np.array([i[0] for i in minibatch]).reshape(self.batch_size, self.state_size)
        next_states = np.array([i[3] for i in minibatch]).reshape(self.batch_size, self.state_size)
        
        # Predições atuais
        targets = self.model.predict(states, verbose=0)
        # Predições do estado futuro usando a TARGET NETWORK (estabilidade)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i, (_, action, reward, _, done) in enumerate(minibatch):
            target = reward
            if not done:
                # Equação de Bellman: R + gamma * max(Q_target(s', a'))
                target = reward + self.gamma * np.amax(next_q_values[i])
            
            # Atualiza apenas o Q-value da ação tomada
            targets[i][action] = target
            
        # Backpropagation na rede principal
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decaimento da exploração
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    # Inicializa o ambiente CartPole
    # Objetivo: Equilibrar a vara. Recompensa: +1 por frame
    env = gym.make('CartPole-v1')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    EPISODES = 50  # Número de episódios de treino
    scores = []
    
    print(f"Iniciando treinamento DQN em {EPISODES} episódios...")
    print("Estado: [Posição Carro, Velocidade Carro, Ângulo Vara, Velocidade Ponta]")
    
    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # 1. Agente escolhe ação
            action = agent.act(state)
            
            # 2. Ambiente reage
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Penalidade personalizada se deixar cair (aprender mais rápido)
            reward = reward if not done else -10
            
            next_state = np.reshape(next_state, [1, state_size])
            
            # 3. Armazena experiência
            agent.remember(state, action, reward, next_state, done)
            
            # 4. Avança estado
            state = next_state
            time += 1
            
            # 5. Treina a rede (Replay)
            # Treinamos apenas quando terminamos o episódio ou a cada X passos para otimizar
            if done or truncated:
                print(f"Episódio: {e+1}/{EPISODES}, Pontuação: {time}, Epsilon: {agent.epsilon:.2f}")
                scores.append(time)
                # Sincroniza a rede alvo a cada N episódios
                if e % agent.update_target_freq == 0:
                    agent.update_target_model()
                    
        # Treina ao final do episódio
        agent.replay()

    # =============================================================================
    # Visualização
    # =============================================================================
    window_size = 5
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Pontuação por Episódio', alpha=0.5)
    plt.plot(range(window_size-1, len(scores)), moving_avg, label='Média Móvel (Tendência)', color='red', linewidth=2)
    plt.title('Performance do Agente DQN (CartPole-v1)')
    plt.ylabel('Tempo de Equilíbrio (Recompensa Total)')
    plt.xlabel('Episódio')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    env.close()
