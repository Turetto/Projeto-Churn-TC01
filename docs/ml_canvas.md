# ML Canvas — Previsão de Churn

## 1. Stakeholders
| Papel | Interesse |
|-------|-----------|
| Diretoria | Reduzir perda de receita por cancelamento |
| Time de CRM | Lista de clientes em risco para ações de retenção |
| Time de ML | Modelo com AUC-ROC >= 0.80 em produção |

## 2. Problema de Negócio
Uma operadora de telecomunicações perde clientes em ritmo acelerado. Cada cliente cancelado representa perda de receita recorrente mensal (MRR). A diretoria precisa identificar clientes com alto risco de cancelamento **antes** que o cancelamento ocorra, para acionar campanhas de retenção.

## 3. Formulação como ML
- **Tipo:** Classificação binária supervisionada
- **Target:** `Churn` (1 = cancelou, 0 = permaneceu)
- **Granularidade:** 1 linha = 1 cliente
- **Horizonte:** predição no momento atual (não temporal)

## 4. Dados
| Atributo | Detalhe |
|----------|---------|
| Fonte | Telco Customer Churn — IBM (Kaggle) |
| Volume | ~7.043 registros |
| Features | 20 variáveis (demográficas, contratuais, de uso e cobrança) |
| Target | Coluna `Churn` (Yes/No) |
| Balanceamento | ~26% churn — desbalanceado, requer atenção |

## 5. Métricas Técnicas
| Métrica | Justificativa |
|---------|---------------|
| AUC-ROC | Mede separabilidade independente do threshold |
| PR-AUC | Mais informativa para classes desbalanceadas |
| F1-Score | Equilíbrio entre precisão e recall |
| Recall | Prioridade — minimizar falsos negativos (churns não detectados) |

## 6. Métrica de Negócio
**Custo de churn evitado:**
- Custo de um falso negativo (churn não detectado): perda do MRR do cliente
- Custo de um falso positivo (ação de retenção desnecessária): custo da campanha
- Assumindo MRR médio >> custo de campanha → maximizar Recall

## 7. SLOs (Service Level Objectives)
| SLO | Valor alvo |
|-----|-----------|
| AUC-ROC mínimo | >= 0.80 |
| Recall mínimo | >= 0.75 |
| Latência de inferência (p99) | <= 200ms |
| Disponibilidade da API | >= 99% |

## 8. Riscos e Limitações
- Dataset estático — não captura deriva temporal do comportamento
- Viés possível em variáveis demográficas (gênero, idoso)
- Volume pequeno (~7k) pode limitar generalização da MLP
- Features de cobrança podem ter missing values

## 9. Arquitetura de Deploy
- **Modo:** Real-time (request/response via FastAPI)
- **Justificativa:** Ações de retenção precisam ser acionadas individualmente