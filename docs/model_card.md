# Model Card — ChurnMLP

## 1. Informações do Modelo

| Campo | Detalhe |
|-------|---------|
| Nome | ChurnMLP v2 |
| Tipo | Classificação binária — Multi-Layer Perceptron |
| Framework | PyTorch 2.2+ |
| Versão | 0.1.0 |
| Data | Abril 2026 |

## 2. Uso Pretendido

### Casos de uso primário
Identificar clientes de uma operadora de telecomunicações com alto risco
de cancelamento (churn), permitindo ações proativas de retenção.

### Usuários pretendidos
- Time de CRM — acionamento de campanhas de retenção
- Time de ML — monitoramento e retraining do modelo
- Diretoria — acompanhamento de métricas de negócio

### Casos de uso fora do escopo
- Predição de churn em outros setores (bancário, varejo, etc.)
- Segmentação de clientes para outras finalidades
- Decisões automatizadas sem revisão humana

## 3. Dataset

| Atributo | Detalhe |
|----------|---------|
| Nome | Telco Customer Churn — IBM |
| Fonte | Kaggle (blastchar/telco-customer-churn) |
| Volume | 7.032 registros (após limpeza) |
| Features | 44 (após pré-processamento) |
| Target | Churn (Yes=1 / No=0) |
| Balanceamento | ~26% churn — desbalanceado |
| Período | Estático — sem dimensão temporal |

### Pré-processamento aplicado
- Conversão de `TotalCharges` para numérico
- Remoção de 11 registros com `TotalCharges` nulo (tenure=0)
- Remoção de 22 registros duplicados
- StandardScaler nas features numéricas
- OneHotEncoder nas features categóricas

## 4. Arquitetura do Modelo

    Input (44 features)
        ↓
    Linear(44→128) → BatchNorm → ReLU → Dropout(0.4)
        ↓
    Linear(128→64) → BatchNorm → ReLU → Dropout(0.4)
        ↓
    Linear(64→32)  → BatchNorm → ReLU → Dropout(0.4)
        ↓
    Linear(32→1)   → Sigmoid
        ↓
    Output — probabilidade de churn [0, 1]

| Hiperparâmetro | Valor |
|----------------|-------|
| hidden_dims | [128, 64, 32] |
| dropout_rate | 0.4 |
| learning_rate | 5e-4 |
| batch_size | 64 |
| optimizer | Adam |
| loss | BCELoss |
| early_stopping patience | 15 |
| threshold de decisão | 0.35 |
| seed | 42 |

## 5. Métricas de Avaliação

### Comparativo v1 vs v2

| Métrica | v1 (threshold=0.50) | v2 (threshold=0.35) | Δ |
|---------|---------------------|---------------------|---|
| ROC-AUC | 0.8317 | 0.8509 | +0.0192 |
| PR-AUC | 0.6380 | 0.6644 | +0.0264 |
| F1 | 0.5723 | 0.6483 | +0.0760 |
| Recall | 0.5027 | 0.7628 | +0.2601 |
| Precision | — | 0.5637 | — |

### SLOs definidos no ML Canvas

| SLO | Alvo | Resultado | Status |
|-----|------|-----------|--------|
| ROC-AUC | >= 0.80 | 0.8509 | ✅ |
| Recall | >= 0.75 | 0.7628 | ✅ |
| Latência p99 | <= 200ms | ~15ms | ✅ |

### Análise de custo de negócio

| Cenário | Economia estimada |
|---------|------------------|
| Sem modelo (baseline) | R$ 0 |
| MLP v1 (threshold=0.50) | R$ 89.250 |
| MLP v2 (threshold=0.35) | R$ 149.250 |

## 6. Limitações

- **Dataset estático** — não captura mudanças de comportamento ao longo do tempo
- **Volume pequeno** — 7.032 registros podem limitar a generalização
- **Sem dimensão temporal** — não modela sazonalidade ou tendências
- **Distribuição fixa** — modelo pode degradar com mudanças no perfil dos clientes
- **Dados de uma empresa** — pode não generalizar para outras operadoras

## 7. Vieses Conhecidos

- **Variável `SeniorCitizen`** — clientes idosos podem ser tratados de forma diferente
- **Variável `gender`** — incluída no modelo mas sem evidência de impacto causal no churn
- **Clientes novos** — registros com `tenure=0` foram removidos — modelo não cobre esse perfil
- **Desbalanceamento** — classe minoritária (churn=1) corrigida via class weights

## 8. Plano de Monitoramento

### Métricas a monitorar
| Métrica | Frequência | Alerta |
|---------|------------|--------|
| ROC-AUC | Semanal | < 0.78 |
| Recall | Semanal | < 0.70 |
| Taxa de churn real | Mensal | Desvio > 5% |
| Latência p99 | Diária | > 200ms |
| Volume de requisições | Diária | Queda > 20% |

### Gatilhos para retraining
- ROC-AUC abaixo de 0.78 por 2 semanas consecutivas
- Distribuição das features com desvio > 2 desvios padrão (data drift)
- Mudança significativa na taxa real de churn (> 5%)

### Playbook de resposta
1. **Alerta de performance** → verificar data drift → retrainar com dados recentes
2. **Alerta de latência** → verificar carga do servidor → escalar horizontalmente
3. **Alerta de volume** → verificar integração com CRM → acionar time de engenharia

## 9. Cenários de Falha

| Cenário | Impacto | Mitigação |
|---------|---------|-----------|
| Feature faltante na requisição | API retorna 422 | Validação Pydantic |
| Modelo não encontrado | API não sobe | Verificar config.json |
| Data drift severo | Predições incorretas | Monitoramento + retraining |
| Sobrecarga de requisições | Latência alta | Rate limiting |

## 10. Rastreabilidade

- **Experimentos:** MLflow — experimento `churn-mlp`
- **Código:** GitHub — Projeto-Churn-TC01
- **Dados:** Kaggle — blastchar/telco-customer-churn
- **Configuração ativa:** `models/config.json