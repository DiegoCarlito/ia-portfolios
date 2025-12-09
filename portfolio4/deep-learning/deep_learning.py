from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuração visual
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('ggplot')

def carregar_e_processar_dados():
    print("--- 1. Carregando Dataset CIFAR-10 ---")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalização dos pixels [0, 255] -> [0, 1]
    # Redes Neurais convergem mais rápido com dados normalizados
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-Hot Encoding dos rótulos (ex: 2 -> [0,0,1,0...])
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"Treino: {x_train.shape}, Teste: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

def construir_modelo_transfer_learning():
    print("\n--- 2. Construindo Arquitetura com VGG16 (Transfer Learning) ---")
    
    # Carrega a VGG16 sem as camadas densas do topo (include_top=False)
    # weights='imagenet': Usa os pesos aprendidos em milhões de imagens
    # input_shape=(32, 32, 3): Tamanho das imagens do CIFAR-10
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # Impedimos que os pesos da base sejam atualizados
    # Isso permite treinar na CPU rapidamente, focando apenas no classificador
    base_model.trainable = False
    
    # Criação do modelo sequencial
    model = models.Sequential([
        base_model,                 # Extrator de Features (Convoluções)
        layers.Flatten(),           # Vetorização
        layers.Dense(256, activation='relu'), # Camada Densa Intermediária
        layers.Dropout(0.5),        # Regularização (evita overfitting)
        layers.Dense(10, activation='softmax') # Camada de Saída (10 classes)
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def visualizar_predicoes(model, x_test, y_test_indices):
    class_names = ['Avião', 'Carro', 'Pássaro', 'Gato', 'Cervo', 
                   'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']
    
    # Pega 5 imagens aleatórias
    indices = np.random.choice(range(len(x_test)), 5, replace=False)
    images = x_test[indices]
    labels = y_test_indices[indices]
    
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i])
        col = 'green' if pred_labels[i] == labels[i][0] else 'red'
        plt.title(f"Pred: {class_names[pred_labels[i]]}\nReal: {class_names[labels[i][0]]}", 
                  color=col, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Dados
    (x_train, y_train), (x_test, y_test) = carregar_e_processar_dados()
    
    # 2. Modelo
    model = construir_modelo_transfer_learning()
    
    # 3. Treinamento
    print("\n--- 3. Iniciando Treinamento (Pode levar alguns minutos na CPU) ---")
    history = model.fit(x_train, y_train, 
                        epochs=5, 
                        batch_size=128,
                        validation_data=(x_test, y_test),
                        verbose=1)
    
    # 4. Métricas e Gráficos
    print("\n--- 4. Gerando Gráficos de Performance ---")
    
    # Gráfico de Acurácia e Perda
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Acurácia
    ax1.plot(history.history['accuracy'], label='Treino', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validação', marker='o')
    ax1.set_title('Acurácia do Modelo (VGG16 Transfer)')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    
    # Perda (Loss)
    ax2.plot(history.history['loss'], label='Treino', marker='o')
    ax2.plot(history.history['val_loss'], label='Validação', marker='o')
    ax2.set_title('Função de Perda (Loss)')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.suptitle('Performance de Deep Learning: Transfer Learning no CIFAR-10', fontsize=16)
    plt.show()
    
    # 5. Teste Visual
    # Reverte o one-hot encoding apenas para pegar o índice da classe real para exibição
    (_, _), (_, y_test_indices) = cifar10.load_data()
    visualizar_predicoes(model, x_test, y_test_indices)
