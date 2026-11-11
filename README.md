# Predictive Maintenance with XGBoost & SHAP

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/explainability-SHAP-red.svg)](https://shap.readthedocs.io/)

Este repositório contém um projeto completo de Ciência de Dados e Machine Learning voltado para **Manutenção Preditiva Industrial**. O objetivo é prever a ocorrência de falhas em máquinas com base em medições de sensores e interpretar o impacto de cada variável na predição utilizando **SHAP (SHapley Additive exPlanations)**.

---

## 📌 Contexto e Objetivo

Na indústria moderna, o monitoramento contínuo de variáveis de operação (temperatura, velocidade rotacional, torque e desgaste de componentes) permite a transição da manutenção reativa ou preventiva tradicional para a **manutenção preditiva**.

Principais benefícios:
- Redução de tempo de parada não planejada (*downtime*).
- Otimização do ciclo de vida útil de ferramentas.
- Transparência e explicabilidade nas decisões do modelo via SHAP para suporte à tomada de decisão de engenheiros e operadores.

---

## 🗂️ Estrutura do Projeto

A organização dos arquivos segue as melhores práticas para projetos de Machine Learning:

```text
predictive-maintenance-xgboost-shap/
├── data/
│   ├── raw/                       # Dados brutos originais (ai4i2020.csv)
│   └── processed/                 # Dados limpos e preparados (machine-failure.csv)
├── models/                        # Artefatos e modelos treinados serializados (.joblib)
├── notebooks/                     # Jupyter Notebooks de exploração e prototipagem
│   └── exploration_data.ipynb     # Análise exploratória, treino e interpretabilidade
├── reports/
│   └── figures/                   # Gráficos, matrizes de confusão e visualizações SHAP
├── src/                           # Código-fonte modular e reutilizável
│   ├── __init__.py                # Inicializador do pacote
│   ├── preprocessing.py          # Limpeza, divisão e transformação de features
│   ├── train.py                  # Treinamento, métricas e persistência do modelo
│   ├── explain.py                # Explicabilidade com SHAP e geração de gráficos
│   └── pipeline.py               # Pipeline ponta a ponta (entrypoint)
├── .gitignore                     # Arquivos e pastas ignorados pelo Git
├── requirements.txt               # Dependências do projeto
└── README.md                      # Documentação completa do projeto
```

---

## 📊 Dataset

O projeto utiliza o **AI4I 2020 Predictive Maintenance Dataset**, composto por 10.000 instâncias com dados sintéticos que refletem com precisão ambientes industriais reais.

### Variáveis utilizadas:
| Feature | Tipo | Descrição |
| :--- | :--- | :--- |
| `Type` | Categórica | Variante de qualidade do produto (L, M, H) |
| `Air temperature [K]` | Numérica | Temperatura do ar ambiente em Kelvin |
| `Process temperature [K]` | Numérica | Temperatura do processo em Kelvin |
| `Rotational speed [rpm]` | Numérica | Velocidade de rotação calculada a partir da potência |
| `Torque [Nm]` | Numérica | Torque aplicado ao equipamento |
| `Tool wear [min]` | Numérica | Tempo de desgaste acumulado da ferramenta |
| **`Machine failure`** | **Alvo (Target)** | Indicador binário de falha (0 = sem falha, 1 = com falha) |

---

## ⚙️ Metodologia e Pipeline

1. **Limpeza e Seleção de Features:**
   - Remoção de identificadores não preditivos (`UDI`, `Product ID`) e subtipos de falha (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), focando na predição global de falha da máquina.
2. **Divisão Estratificada:**
   - Divisão treino/teste (80/20) com estratificação para manter a proporção da classe minoritária (falha).
3. **Pré-processamento:**
   - `OneHotEncoder` na variável categórica `Type`.
   - `ColumnTransformer` para tratamento integrado no fluxo de modelagem.
4. **Modelagem:**
   - Classificador `XGBClassifier` (Extreme Gradient Boosting), adequado para dados tabulares desbalanceados.
5. **Avaliação:**
   - Avaliação com foco em métricas de classificação: Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão.
6. **Interpretabilidade (XAI):**
   - Utilização do `shap.TreeExplainer` para cálculo dos valores de Shapley e geração do gráfico *beeswarm* revelando a importância e a direção do impacto de cada feature.

---

## 🚀 Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/luciovagner/predictive-maintenance-xgboost-shap.git
cd predictive-maintenance-xgboost-shap
```

### 2. Criar e ativar o ambiente virtual
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Executar o pipeline completo
Para executar todo o fluxo (processamento dos dados, treino do modelo, avaliação e geração dos gráficos SHAP):
```bash
python -m src.pipeline
```

### 5. Explorar via Jupyter Notebook
Para visualizar os experimentos e gráficos interativos:
```bash
jupyter lab notebooks/exploration_data.ipynb
```

---

## 📈 Interpretabilidade com SHAP

A análise com SHAP permite entender:
- Quais características mais contribuem para o risco de falha (ex.: alto desgaste da ferramenta `Tool wear [min]` e combinações extremas de `Torque [Nm]` e `Rotational speed [rpm]`).
- O comportamento individual e global das predições, viabilizando intervenções preventivas direcionadas.

---

## 📝 Licença

Este projeto está sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
