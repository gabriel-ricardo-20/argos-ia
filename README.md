# Argos · IA (Artificial Intelligence & Chatbot) - Global Solution 2026/2

Copiloto para o pequeno produtor rural (tema **CitizenTech**, frentes "Campo" e "Clima e emergências").
Integrantes, RM e turma: ver `integrantes.txt`.

## 1. O problema e como a IA ajuda o Argos
O pequeno produtor de milho decide o plantio sem saber se o ano tende a ser bom ou ruim. Uma quebra de safra (colher bem abaixo do normal) compromete a renda da família, e os alertas de clima costumam chegar em linguagem técnica.
O Argos usa **dois modelos de Machine Learning clássico**, treinados com dados públicos de 30 municípios (15 do Nordeste e 15 do Sul, 2010–2024):

| Modelo | Tipo | Pergunta que responde | Rota da API |
|---|---|---|---|
| `linreg_v1` — Regressão Linear | Regressão | "Quanto devo colher por hectare num ano com este clima?" (kg/ha) | `POST /api/v1/produtividade` |
| `logreg_v1` — Regressão Logística | Classificação | "Este ano tem risco de quebra de safra?" (BAIXO / MEDIO / ALTO) | `POST /api/v1/risco-quebra` |

Na tela **Hoje** do front, os dois aparecem juntos num card: o risco (cor + ícone + palavra) e o rendimento esperado. A decisão final é sempre do produtor e do técnico.

## 2. Estrutura
```
argos-ia/
  dados/        sidra_milho.csv · municipios.csv · ibge_mesorregioes.csv · clima_bruto/ (30 JSON)
                oni_bruto.txt · oni.csv · dataset_argos.csv · log_downloads.txt
  notebooks/    01_coleta_e_preparo · 02_eda · 03_regressao · 04_classificacao (todos com saídas)
  modelos/      regressao_v1.joblib · risco_quebra_v1.joblib
  api/          api.py · requirements.txt · insomnia_argos.json
  README.md · integrantes.txt
```
Pipeline: **coleta (01) → EDA (02) → pré-processamento, treino e avaliação (03 e 04) → exportação .joblib (fim do 03 e do 04) → API Flask (api/)**.

## 3. Dados e fontes (todos baixados em 18/11/2026; endereço completo de cada chamada em `dados/log_downloads.txt`)
| Fonte | O que foi usado | Endereço |
|---|---|---|
| IBGE/SIDRA — Produção Agrícola Municipal, **tabela 1612** | Variável 112 (rendimento médio, kg/ha) e 216 (área colhida, ha), produto 2711 (milho em grão), municípios de PE, BA, RN, PI, RS, SC e PR, 2005–2024 | https://apisidra.ibge.gov.br/values/t/1612/… · https://sidra.ibge.gov.br/tabela/1612 |
| IBGE — API de Localidades | Mesorregião de cada município (filtro do MATOPIBA) | https://servicodados.ibge.gov.br/api/v1/localidades/estados/{UF}/municipios |
| Open-Meteo — Historical Weather (archive-api) | Chuva diária, temperatura média e máxima, 2005–2024, um ponto por município (reanálise) | https://archive-api.open-meteo.com/v1/archive |
| Open-Meteo — Geocoding | Latitude e longitude da sede dos municípios | https://geocoding-api.open-meteo.com/v1/search |
| NOAA CPC — Oceanic Niño Index (ONI) | Média anual dos 12 trimestres móveis | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt |

**Recorte e critério (notebook 01):** milho, 2005–2024; só municípios com rendimento nos 20 anos (os símbolos do IBGE "-", "..." e "X" contam como ausente; "-" é zero, e sem área colhida o kg/ha não existe); fora das mesorregiões Extremo Oeste Baiano e Sudoeste Piauiense (cerrado do MATOPIBA); municípios do banco de dados primeiro; vagas restantes pela maior área colhida média. Petrolina, Juazeiro, Mossoró e Picos ficaram fora por terem anos sem colheita.

**Dataset (`dados/dataset_argos.csv`):** 450 linhas (30 municípios × 2010–2024), uma linha = um município em um ano.
- Entradas (nomes do contrato com o front): `chuva_total_mm`, `temp_media`, `temp_max_media`, `dias_acima_35`, `maior_seq_seca`, `oni`, `regiao_sul`, `media_5a`.
- Alvos: `rend_kg_ha` (regressão) e `quebra` = 1 quando `rend_kg_ha < 0,8 × media_5a` (regra do grupo; `media_5a` usa só os 5 anos **anteriores**, com `shift(1)`).
- Identificação: `cd_municipio`, `nm_municipio`, `sg_uf`, `ano`.
- Quebra em 27,3% das linhas. Separação **temporal**: treino 2010–2019 (300 linhas), teste 2020–2024 (150 linhas).

## 4. Métricas finais (teste 2020–2024)
**Regressão — `linreg_v1`** (7 entradas; `temp_media` saiu por colinearidade r = 0,99 com `temp_max_media`)
| | MAE (kg/ha) | RMSE (kg/ha) | R² |
|---|---|---|---|
| Baseline "rendimento = media_5a" | 1.152 | 1.577 | 0,544 |
| Regressão Linear | 1.199 | 1.605 | 0,527 |

O modelo ganha do baseline no treino (MAE 806 × 898), em 4 de 5 blocos de validação por anos (848 × 898) e nos anos **sem** quebra do teste (797 × 900), mas perde no teste como um todo por causa dos anos de quebra (1.956 × 1.626): ele prevê o ano típico.

**Classificação — `logreg_v1`** (escolhida contra KNN, K = 13, pelo **recall da quebra**)
| | Recall quebra (validação treino) | Recall quebra (teste) | Precisão quebra (teste) | AUC (teste) |
|---|---|---|---|---|
| Regressão Logística (limiar 0,4) | 0,848 (limiar 0,5) | **0,404** (21 de 52) | 0,356 | 0,565 |
| KNN (K = 13, limiar 0,5) | 0,380 | 0,154 (8 de 52) | 0,571 | 0,552 |

Risco: **BAIXO** p < 0,20 · **MEDIO** 0,20 ≤ p < 0,40 · **ALTO** p ≥ 0,40 (0,40 = limiar escolhido na validação do treino; 0,20 = metade do limiar, faixa onde ficaram 94% das quebras do treino).
No teste, o modelo avisa bem no Nordeste (18 de 26 quebras) e mal no Sul (3 de 26): as secas de La Niña de 2020–2022 quase não existiam no treino (Sul: 5,3% de quebra no treino × 34,7% no teste).

## 5. Como rodar (Windows, cmd)
Requisito: Python 3.12.
```cmd
cd argos-ia
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install -r api\requirements.txt
cd api
python api.py
```
A API sobe em `http://localhost:5000`. Os notebooks rodam no mesmo ambiente (VS Code ou `jupyter lab`, kernel `.venv`), de dentro da pasta `notebooks/`. Os downloads ficam salvos em `dados/`: rodar os notebooks de novo não depende de internet.
**Testes:** importar `api/insomnia_argos.json` no Insomnia (6 requisições, com linhas reais do dataset).

## 6. Contrato das rotas (prefixo `/api/v1`, CORS liberado, porta 5000)
| Método | Rota | Resposta 200 |
|---|---|---|
| GET | `/api/v1/saude` | `{"status": "ok"}` |
| POST | `/api/v1/produtividade` | `{"kg_ha_previsto": 959.2, "modelo": "linreg_v1"}` |
| POST | `/api/v1/risco-quebra` | `{"risco": "ALTO", "probabilidade": 0.62, "modelo": "logreg_v1"}` |

Corpo dos POST (8 campos numéricos; a API usa só os que o modelo precisa, na ordem salva no `.joblib`):
```json
{"chuva_total_mm": 482.2, "temp_media": 25.67, "temp_max_media": 31.42, "dias_acima_35": 48,
 "maior_seq_seca": 26, "oni": 0.81, "regiao_sul": 0, "media_5a": 669.6}
```
(Linha real: Araci/BA, 2023 — ano seco, quebra real. Ano normal de exemplo: Chapecó/SC, 2020 → risco BAIXO, 0,12; 9.125,1 kg/ha.)
- `probabilidade` de 0 a 1 com 2 casas; `kg_ha_previsto` com 1 casa; `risco` em maiúsculas e sem acento.
- Erros em JSON, sem stack trace: **400** `{"erro": "Falta o campo oni"}` / `{"erro": "O campo X precisa ser número"}` · **404** rota inexistente · **405** método errado · **500** erro interno.
- Os modelos são carregados uma vez, quando a API sobe (`joblib.load`); modelo, scaler, colunas, limiar e cortes vêm do `.joblib`.

## 7. Limitações e ética
- **Dados agregados:** o modelo aprende com a média do município, não com a roça de cada produtor. Pode não representar o pequeno produtor de uma comunidade específica (viés).
- **Viés de sobrevivência:** municípios com ano de perda total (área colhida zero) ficaram fora; os piores anos do semiárido estão sub-representados.
- **Clima grosseiro:** ano civil e chuva em mm absolutos; a época do plantio e a chuva relativa à normal de cada lugar não entram nesta versão. A amostra é concentrada na BA e no PR.
- **Mudança no clima:** o teste (2020–2024) teve secas de La Niña no Sul quase ausentes no treino; os resultados no teste são modestos e estão relatados assim.
- **Responsabilidade:** o risco é um alerta, não uma ordem; a decisão é do produtor e do técnico, com a fonte oficial ao lado.
- **Validade:** o modelo envelhece e precisa ser retreinado a cada safra (nova PAM do IBGE).
- **LGPD:** os modelos usam só dados públicos agregados por município; nenhum dado pessoal entra no treino. Dados do produtor no app seguem consentimento (banco de dados do Argos).

## 8. Versões
Python 3.12.10 · scikit-learn 1.9.1 · pandas 3.1.1 · numpy 2.5.5 · Flask 3.1.3 · flask-cors 6.0.5 · joblib 1.6.0 (fixadas em `api/requirements.txt`; a versão do scikit-learn do treino vai gravada em cada `.joblib`, e a API avisa se for diferente).
