# Turnover RH — do .pkl para produção

Você tem um modelo treinado. Alguém precisa usá-lo amanhã de manhã, todo dia,
sem você por perto. As quatro aulas de MLOps são sobre a distância entre essas
duas frases.

**Abra o `ROTEIRO_AULA1.html` no navegador e siga por ele.** Este README é só o
mapa da pasta.

---

## A pasta chega pela metade, de propósito

| Já está aqui | Para quê |
|---|---|
| `sql/01_criar_tabelas.sql` | o DDL das três tabelas — você roda no seu banco |
| `.gitignore` | o `.env` fica de fora do git |
| `app/banco.py` | as consultas ao Postgres |
| `app/static/` | o front: colaboradores, novo colaborador, lote |
| `treinar_modelo_v1.py` | rede de segurança se o seu `.pkl` não servir |
| `dados/features_padronizadas.csv` | a tabela que saiu da Aula 3 |
| `.env.exemplo` | modelo da conexão |

| Você escreve na Aula 1 | Para quê |
|---|---|
| `config.py` | constantes e a conexão lida do `.env` |
| `preparar_dados.py` | popula `treino_v1` e `validacao_congelada` (split 80/20) |
| `onboard_v1.py` | registra o `.pkl` como versão 1. **Não retreina nada** |
| `app/main.py` | a API. Carrega o modelo **do registry, pelo alias** |
| `monitorar.py` | mede um lote já pontuado. Não cria modelo |

O código dos cinco está no roteiro, com a explicação de cada pedaço.
Copiar sem ler funciona e não ensina nada.

---

## Subir

**1. O banco é seu.** Docker local, instalado na máquina ou na nuvem — onde
ele mora é escolha sua. Depois de ter um Postgres acessível:

```bash
createdb turnover -U postgres
createdb mlflow   -U postgres
psql turnover -U postgres -f sql/01_criar_tabelas.sql
psql turnover -U postgres -c "\dt"      # tem que listar 3 tabelas
```

O `-U postgres` não é opcional: é o superusuário padrão da instalação. Sem ele,
o cliente tenta entrar com o seu usuário do sistema, que quase nunca existe
dentro do banco. O script é `IF NOT EXISTS` — rodar duas vezes não quebra nada.

**2. A conexão.** Copie e edite com os seus dados:

```bash
cp .env.exemplo .env
```

**3. O resto:**

```bash
pip install -r requirements.txt
python preparar_dados.py           # 19.200 treino / 4.800 validacao (80/20)
```

Coloque o seu `modelo_v1.pkl` da Aula 4 em `artefatos/`.

Os lotes mensais **não estão na sua pasta**: chega um arquivo novo por aula, com
pessoas que não estão no seu banco — como o RH mandaria de verdade.

**4. Dois terminais, os dois abertos o tempo todo:**

```bash
# terminal 1 — MLflow
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri postgresql://turnover:turnover@127.0.0.1:5432/mlflow --default-artifact-root ./mlruns
```

**O servidor do MLflow não lê o seu `.env`** — a conexão vai na linha de comando.
Se o seu usuário não for `turnover`, gere o comando já com os seus dados:

```bash
python config.py
```

```bash
# terminal 2 — API de produção (só funciona depois do Passo 6)
uvicorn app.main:app --reload --port 8000
```

MLflow em <http://127.0.0.1:5000> · produção em <http://127.0.0.1:8000>

---

## Dois bancos, duas coisas diferentes

- **`mlflow`** — runs, métricas, parâmetros e o catálogo de versões. O MLflow
  cria as 19 tabelas sozinho. Abra e olhe `registered_model_aliases`: o
  `@champion` é uma linha lá. **Não guarda nenhuma linha do seu dado de
  treino**, só uma referência com digest.
- **`turnover`** — as três tabelas que *você* criou: `treino_v1`,
  `validacao_congelada` e `predicoes`. Uma linha de `predicoes` por pergunta
  feita ao modelo: a entrada, a probabilidade, a classe e qual versão respondeu.

Confundir os dois é o erro mais comum da semana.

---

## As rotas da API

| Rota | Faz o quê |
|---|---|
| `GET /` | devolve o front |
| `GET /saude` | versão em produção, alias, faixas de risco, lotes |
| `POST /prever` | um colaborador → probabilidade, classe e faixa |
| `POST /lote` | CSV → pontua tudo e grava |
| `GET /colaboradores` | todos os já avaliados, com a faixa calculada |
| `POST /recarregar` | relê o registry sem reiniciar a API |

As faixas de risco (**ok** abaixo de 50%, **atenção** de 50 a 60%, **alerta**
acima de 60%) são calculadas no backend, não no javascript: onde começa o
alerta é regra de negócio.

---

## O que vem depois

Nas próximas três aulas, lotes vão chegar e às vezes a métrica vai cair.
**"A métrica caiu" não diz qual é a causa.** Seu trabalho vai ser descobrir a
causa e escolher a resposta certa — e retreinar é só uma das respostas.

Hoje não tem incidente. Hoje o objetivo é ter um baseline: você não consegue
dizer que algo mudou se nunca mediu como era antes.
