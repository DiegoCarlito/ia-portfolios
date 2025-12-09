import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# 1. ENGENHARIA DE DADOS
# =============================================================================
def gerar_dados_churn(n_samples=2000):
    np.random.seed(42)
    
    # Features Numéricas
    tenure = np.random.randint(1, 72, n_samples)  # meses de contrato
    monthly_charges = np.random.normal(70, 30, n_samples) # valor da fatura
    total_charges = tenure * monthly_charges + np.random.normal(0, 10, n_samples)
    
    # Features Categóricas
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples)
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
    
    # Target (Churn): Criando uma relação complexa não-linear para o modelo aprender
    churn_indices = []
    
    for i in range(n_samples):
        score = 0
        if contracts[i] == 'Month-to-month': score += 2
        if internet_service[i] == 'Fiber optic': score += 1.5
        if monthly_charges[i] > 90: score += 1
        if tenure[i] < 12: score += 1.5
        
        # Sigmoid para probabilidade
        prob = 1 / (1 + np.exp(-(score - 3)))
        if np.random.random() < prob:
            churn_indices.append(1)
        else:
            churn_indices.append(0)
            
    df = pd.DataFrame({
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contracts,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'Churn': churn_indices
    })
    
    return df

print("--- 1. Gerando Dataset Sintético ---")
df = gerar_dados_churn()
print(f"Shape dos dados: {df.shape}")
print(f"Distribuição de Churn:\n{df['Churn'].value_counts(normalize=True)}")

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# =============================================================================
# 2. CONSTRUÇÃO DO PIPELINE DE PRÉ-PROCESSAMENTO
# =============================================================================
# Seleciona colunas por tipo
numeric_features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Contract', 'InternetService', 'PaymentMethod']

# Pipeline Numérico: Inputação de média (caso haja nulos) -> Padronização
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline Categórico: Inputação de moda -> OneHotEncoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Processador de Colunas: Aplica as transformações específicas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# =============================================================================
# 3. MODELAGEM E OTIMIZAÇÃO DE HIPERPARÂMETROS
# =============================================================================
# Pipeline Final: Pré-processador -> Classificador
# Usamos Gradient Boosting, que constrói árvores sequencialmente para corrigir erros anteriores
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Grid de Hiperparâmetros para Otimização
param_grid = {
    'classifier__n_estimators': [100, 200],      # Número de árvores
    'classifier__learning_rate': [0.01, 0.1, 0.2], # Taxa de aprendizado
    'classifier__max_depth': [3, 5, 7],          # Profundidade máxima da árvore
    'classifier__subsample': [0.8, 1.0]          # Fração de amostras para treinar cada base learner
}

print("\n--- 2. Iniciando Otimização de Hiperparâmetros (RandomizedSearchCV) ---")
# Validação Cruzada Estratificada (5 dobras)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_grid,
    n_iter=10, # Número de combinações aleatórias a testar
    scoring='f1', # Otimizar para F1-Score (bom para desbalanceados)
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

print(f"Melhores parâmetros encontrados: {search.best_params_}")
print(f"Melhor F1-Score na validação: {search.best_score_:.4f}")

# =============================================================================
# 4. AVALIAÇÃO E VISUALIZAÇÃO
# =============================================================================
print("\n--- 3. Avaliação no Conjunto de Teste ---")
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Relatório de Classificação
print(classification_report(y_test, y_pred))

# Plotagem: Matriz de Confusão e Curva ROC
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Matriz de Confusão')
ax[0].set_ylabel('Real')
ax[0].set_xlabel('Predito')

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('Taxa de Falsos Positivos')
ax[1].set_ylabel('Taxa de Verdadeiros Positivos')
ax[1].set_title('Receiver Operating Characteristic (ROC)')
ax[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

# =============================================================================
# 5. INTERPRETABILIDADE (Feature Importance)
# =============================================================================
# Extraindo nomes das colunas após OneHotEncoding
ohe_cols = best_model.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_cols = numeric_features + list(ohe_cols)

# Extraindo importâncias do modelo dentro do pipeline
importances = best_model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[-10:] # Top 10

plt.figure(figsize=(10, 6))
plt.title('Top 10 Variáveis Mais Importantes (Feature Importance)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [all_cols[i] for i in indices])
plt.xlabel('Importância Relativa')
plt.show()
