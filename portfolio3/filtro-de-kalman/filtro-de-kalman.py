import numpy as np
import matplotlib.pyplot as plt

class KalmanFilterLinear:
    """Implementação genérica de um Filtro de Kalman Linear."""
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # Matriz de Transição de Estado
        self.H = H  # Matriz de Observação
        self.Q = Q  # Covariância do Ruído do Processo
        self.R = R  # Covariância do Ruído da Medição
        self.x = x0 # Estado Inicial
        self.P = P0 # Covariância Inicial

    def predict(self, u=None):
        """Passo de Predição (A Priori)"""
        # x = Fx + Bu (Se houver controle u)
        if u is None:
            self.x = self.F @ self.x
        else:
            # Assume-se que o modelo externo lida com B*u na chamada ou aqui
            # Para este exemplo simples, passamos Bu já calculado como u
            self.x = self.F @ self.x + u
            
        # P = FPF' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """Passo de Atualização (A Posteriori)"""
        # Inovação (Resíduo): y = z - Hx
        y = z - self.H @ self.x
        
        # Covariância da Inovação: S = HPH' + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Ganho de Kalman: K = PH'S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Atualiza Estado: x = x + Ky
        self.x = self.x + K @ y
        
        # Atualiza Covariância: P = (I - KH)P
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x

def simular_balistica(dt=0.1, steps=50):
    """Gera a trajetória real de um projétil sob gravidade."""
    g = 9.81
    
    # Estado Real Inicial: x=0, y=0, vx=20, vy=25
    x_real = np.array([[0], [0], [20], [25]], dtype=float)
    
    trajetoria_real = []
    medicoes = []
    
    # Matriz de Evolução Real (Física Discreta)
    # x = x + vx*dt
    # y = y + vy*dt - 0.5*g*dt^2 (aproximado na velocidade)
    # vx = vx
    # vy = vy - g*dt
    
    for _ in range(steps):
        # Atualiza Física
        # Posição
        x_real[0] += x_real[2] * dt
        x_real[1] += x_real[3] * dt
        # Velocidade (Gravidade atua em Y)
        x_real[3] -= g * dt
        
        # Para se bater no chão
        if x_real[1] < 0:
            break
            
        trajetoria_real.append(x_real.copy())
        
        # Gera Medição Ruidosa (Radar)
        # Ruído alto em X e Y
        noise = np.random.normal(0, 3.0, (2, 1)) 
        z = np.array([[x_real[0,0]], [x_real[1,0]]]) + noise
        medicoes.append(z)
        
    return trajetoria_real, medicoes

def main():
    print("Iniciando Rastreamento Balístico com Filtro de Kalman...")
    
    dt = 0.1
    
    # 1. Modelo do Filtro (Cinemática CV - Constant Velocity)
    # Estado: [x, y, vx, vy]
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Matriz de Observação (Medimos apenas X e Y)
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # Incertezas
    Q = np.eye(4) * 0.1   # Ruído do processo (pequeno, a física é bem comportada)
    R = np.eye(2) * 9.0   # Ruído da medição (alto, sigma=3 -> var=9)
    
    # Estado Inicial (Chute)
    x0 = np.array([[0], [0], [0], [0]]) # Começamos sem saber nada
    P0 = np.eye(4) * 500 # Incerteza inicial gigantesca
    
    kf = KalmanFilterLinear(F, H, Q, R, x0, P0)
    
    # 2. Simulação
    traj_real, medicoes = simular_balistica(dt=dt)
    
    estimativas = []
    
    # Vetor de Controle (Gravidade)
    # A gravidade é uma força externa conhecida que atua na velocidade Y
    u = np.array([[0], [0], [0], [-9.81 * dt]])
    
    # 3. Loop de Filtragem
    for z in medicoes:
        # Predição (com controle da gravidade)
        kf.predict(u=u)
        
        # Atualização (com a medição do radar)
        est = kf.update(z)
        estimativas.append(est)
        
    # 4. Visualização
    trx_real = [p[0,0] for p in traj_real]
    try_real = [p[1,0] for p in traj_real]
    
    meas_x = [z[0,0] for z in medicoes]
    meas_y = [z[1,0] for z in medicoes]
    
    est_x = [e[0,0] for e in estimativas]
    est_y = [e[1,0] for e in estimativas]
    
    plt.figure(figsize=(10, 6))
    plt.plot(trx_real, try_real, 'g-', linewidth=2, label='Trajetória Real (Projétil)')
    plt.scatter(meas_x, meas_y, c='r', marker='x', s=30, alpha=0.6, label='Medições Ruidosas (Radar)')
    plt.plot(est_x, est_y, 'b--', linewidth=2, label='Filtro de Kalman')
    
    plt.title("Rastreamento de Projétil 2D com Kalman Linear")
    plt.xlabel("Distância (m)")
    plt.ylabel("Altitude (m)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig("kalman_balistica.png", dpi=300)
    print("Gráfico salvo em 'kalman_balistica.png'")

if __name__ == "__main__":
    main()