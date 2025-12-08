from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ----------------------------------------------------------
# CONSTRUÇÃO DA REDE BAYESIANA
# ----------------------------------------------------------

# Cada tupla representa uma relação de causa → efeito
# Estrutura baseada no conhecimento médico sobre doenças cardíacas
model = DiscreteBayesianNetwork([
    # Fatores demográficos afetam colesterol
    ('age', 'chol'),
    ('sex', 'chol'),
    
    # Fatores de risco afetam o diagnóstico
    ('chol', 'num'),       # Colesterol alto causa Doença
    ('fbs', 'num'),        # Diabetes (glicemia em jejum) influencia Doença
    
    # Doença causa sintomas observáveis
    ('num', 'cp'),         # Doença causa Dor no Peito (chest pain)
    ('num', 'thalach')     # Doença afeta Frequência Cardíaca máxima
])

# ----------------------------------------------------------
# DEFINIÇÃO DAS CPDs (Tabelas de Probabilidade Condicional)
# ----------------------------------------------------------

# --- VARIÁVEIS RAIZ (sem pais) ---

# Age: 0=Jovem, 1=Idoso
# Distribuição assumida: 50/50 na população alvo
cpd_age = TabularCPD(
    variable='age',
    variable_card=2,
    values=[[0.5],   # Jovem
            [0.5]],  # Idoso
    state_names={'age': ['Jovem', 'Idoso']}
)

# Sex: 0=Feminino, 1=Masculino
# Dataset Cleveland tem aproximadamente 70% de homens
cpd_sex = TabularCPD(
    variable='sex',
    variable_card=2,
    values=[[0.3],   # Feminino
            [0.7]],  # Masculino
    state_names={'sex': ['Feminino', 'Masculino']}
)

# FBS (Fasting Blood Sugar): 0=Normal, 1=Diabetes
# Glicemia em jejum > 120 mg/dl indica diabetes
# Prevalência aproximada: 15% na população
cpd_fbs = TabularCPD(
    variable='fbs',
    variable_card=2,
    values=[[0.85],   # Normal
            [0.15]],  # Diabetes
    state_names={'fbs': ['Normal', 'Diabetes']}
)

# --- VARIÁVEIS INTERMEDIÁRIAS ---

# Chol (Colesterol): 0=Normal, 1=Alto
# Depende de Age e Sex
# Homens idosos têm maior probabilidade de colesterol alto
cpd_chol = TabularCPD(
    variable='chol',
    variable_card=2,
    evidence=['age', 'sex'],
    evidence_card=[2, 2],
    # Ordem das colunas: Age=[Jovem, Jovem, Idoso, Idoso], Sex=[F, M, F, M]
    values=[
        [0.90, 0.70, 0.60, 0.40],  # Normal
        [0.10, 0.30, 0.40, 0.60]   # Alto
    ],
    state_names={
        'chol': ['Normal', 'Alto'],
        'age': ['Jovem', 'Idoso'],
        'sex': ['Feminino', 'Masculino']
    }
)

# --- VARIÁVEL ALVO (Diagnóstico) ---

# num (Diagnóstico): 0=Saudável, 1=Doente
# Depende de Colesterol (chol) e Glicemia (fbs)
# Probabilidades baseadas em conhecimento médico:
# - Colesterol alto + diabetes = 80% chance de doença
# - Apenas diabetes = 20% chance
# - Apenas colesterol alto = 60% chance
# - Nenhum fator = 5% chance (risco base da população)
cpd_num = TabularCPD(
    variable='num',
    variable_card=2,
    evidence=['chol', 'fbs'],
    evidence_card=[2, 2],
    # Ordem: Chol=[Normal, Normal, Alto, Alto], FBS=[Normal, Diabetes, Normal, Diabetes]
    values=[
        [0.95, 0.80, 0.40, 0.20],  # Saudável
        [0.05, 0.20, 0.60, 0.80]   # Doente
    ],
    state_names={
        'num': ['Saudavel', 'Doente'],
        'chol': ['Normal', 'Alto'],
        'fbs': ['Normal', 'Diabetes']
    }
)

# --- SINTOMAS (Variáveis observáveis) ---

# CP (Chest Pain - Dor no Peito): 0=Assintomático, 1=Angina Típica
# Se doente, chance de ter angina sobe drasticamente
cpd_cp = TabularCPD(
    variable='cp',
    variable_card=2,
    evidence=['num'],
    evidence_card=[2],
    # Ordem: num=[Saudável, Doente]
    values=[
        [0.90, 0.20],  # Assintomático
        [0.10, 0.80]   # Angina
    ],
    state_names={
        'cp': ['Assintomatico', 'Angina'],
        'num': ['Saudavel', 'Doente']
    }
)

# Thalach (Frequência Cardíaca Máxima): 0=Normal, 1=Anormal/Baixa
# Doença cardíaca tende a reduzir a capacidade cardíaca
cpd_thalach = TabularCPD(
    variable='thalach',
    variable_card=2,
    evidence=['num'],
    evidence_card=[2],
    # Ordem: num=[Saudável, Doente]
    values=[
        [0.90, 0.40],  # Normal
        [0.10, 0.60]   # Anormal
    ],
    state_names={
        'thalach': ['Normal', 'Anormal'],
        'num': ['Saudavel', 'Doente']
    }
)

# Adiciona todas as CPDs ao modelo
model.add_cpds(cpd_age, cpd_sex, cpd_fbs, cpd_chol, cpd_num, cpd_cp, cpd_thalach)

# Verifica se o modelo é válido (CPDs somam 1, estrutura consistente)
assert model.check_model()

if __name__ == "__main__":
    
    # Método de inferência: eliminação de variáveis
    infer = VariableElimination(model)
    
    # Lista de cenários de teste com evidências observadas
    # IMPORTANTE: Quando usamos state_names, devemos passar os nomes dos estados (strings)
    cenarios = [
        (
            "Cenário 1: Paciente de baixo risco (jovem, feminino, sem comorbidades)",
            {
                "age": "Jovem",
                "sex": "Feminino",
                "fbs": "Normal"
            }
        ),
        
        (
            "Cenário 2: Paciente de alto risco demográfico (idoso, masculino, diabético)",
            {
                "age": "Idoso",
                "sex": "Masculino",
                "fbs": "Diabetes"
            }
        ),
        
        (
            "Cenário 3: Inferência reversa - paciente sintomático",
            {
                "cp": "Angina",
                "thalach": "Anormal"
            }
        ),
        
        (
            "Cenário 4: Evidência mista - jovem com sintomas preocupantes",
            {
                "age": "Jovem",
                "sex": "Masculino",
                "cp": "Angina"
            }
        ),
        
        (
            "Cenário 5: Colesterol alto isolado",
            {
                "chol": "Alto"
            }
        ),
        
        (
            "Cenário 6: Pior caso - todos os fatores de risco presentes",
            {
                "age": "Idoso",
                "sex": "Masculino",
                "fbs": "Diabetes",
                "chol": "Alto",
                "cp": "Angina",
                "thalach": "Anormal"
            }
        )
    ]
    
    # Loop para executar todos os cenários e calcular a probabilidade de doença
    for nome, evidencias in cenarios:
        print(f"--- {nome} ---")
        print(f"Evidências: {evidencias}")
        
        # Inferência: calcula P(num | evidências)
        resultado = infer.query(variables=["num"], evidence=evidencias)
        
        # Extrai probabilidade de doença (estado 1)
        prob = resultado.values[1] * 100
        
        print(resultado)
        print(f"-> Probabilidade de Doença Cardíaca: {prob:.2f}%\n")
