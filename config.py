"""
Constantes compartilhadas por toda a aplicacao.

Regra da disciplina: nenhum destes valores aparece escrito no meio do codigo.
Se voce vir "champion" digitado dentro do main.py, alguem quebrou a regra.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent
load_dotenv(RAIZ / ".env")

# -------------------------------------------------------------- Postgres ----
# Um servidor Postgres, dois bancos: o seu (turnover) e o do MLflow.
# Onde o servidor vive e escolha sua -- Docker, local ou nuvem.
# Senha em variavel de ambiente, nao no codigo -- ver .env.exemplo.
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_SENHA = os.getenv("PG_SENHA", "123")
PG_DB_MLFLOW = os.getenv("PG_DB_MLFLOW", "mlflow")
PG_DB_APP = os.getenv("PG_DB_APP", "turnover")


def uri_postgres(banco):
    return f"postgresql://{PG_USER}:{PG_SENHA}@{PG_HOST}:{PG_PORT}/{banco}"


# Onde o MLflow guarda runs, metricas e o catalogo de versoes.
# E este endereco que voce passa no --backend-store-uri ao subir o servidor.
URI_BANCO_MLFLOW = uri_postgres(PG_DB_MLFLOW)

# Onde a aplicacao guarda o log de predicoes. Banco separado, de proposito.
URI_BANCO_APP = uri_postgres(PG_DB_APP)

# ---------------------------------------------------------------- MLflow ----
# Onde o MLflow guarda runs, metricas e versoes de modelo.
# E o endereco do servidor que voce sobe no terminal 1.
URI_TRACKING = os.getenv("MLFLOW_URI", "http://127.0.0.1:5000")

# Um experimento agrupa runs que respondem a mesma pergunta.
# Todos os treinos e monitoramentos do turnover caem aqui dentro.
NOME_EXPERIMENTO = "turnover_rh"

# Nome do modelo dentro do Model Registry (o catalogo de versoes).
# E o "turnover_rh" que aparece em models:/turnover_rh@champion
NOME_MODELO = "turnover_rh"

# ALIAS = um apelido movel que aponta para UMA versao do modelo.
# Promover = mover o apelido. A aplicacao nunca sabe o numero da versao.
ALIAS_PROD = "champion"       # o que esta atendendo producao agora
ALIAS_CANDIDATO = "challenger"  # candidato esperando aprovacao (usado da Aula 2 em diante)

# ------------------------------------------------------------ Aplicacao ----
PASTA_ARTEFATOS = RAIZ / "artefatos"
PASTA_DADOS = RAIZ / "dados"
PASTA_RELATORIOS = RAIZ / "relatorios"

# ---------------------------------------------------------------- Modelo ----
COLUNA_ID = "id_pessoa"
COLUNA_ALVO = "pediu_para_sair"

# O contrato de entrada da v1: 15 features comportamentais, nesta ordem.
# A ordem importa: e ela que vira a signature registrada no MLflow.
FEATURES = [
    "abs_eventos",
    "abs_qtd_total",
    "horas_previstas_total",
    "acidentes_eventos",
    "acidentes_com_afastamento",
    "acidentes_dias_perdidos",
    "he_eventos",
    "he_referencia_total",
    "he_valor_total",
    "hi_eventos",
    "hi_minutos_irregulares",
    "hi_minutos_extras",
    "mov_sal_eventos",
    "mov_sal_valor_total",
    "mov_sal_perc_medio",
]

# Acima disso, o colaborador entra na lista de risco.
# 0.5 e o default de qualquer classificador -- mas e uma decisao DE NEGOCIO,
# nao uma constante da matematica. Marina pode querer 0.7.
LIMIAR_DECISAO = 0.5

# As cores do card no front. Repare que a primeira faixa termina exatamente
# no LIMIAR_DECISAO: verde e, literalmente, "classe 0".
#   ate 50%   ok        verde
#   50 a 60%  atencao   amarelo
#   acima 60% alerta    vermelho
FAIXA_ATENCAO = 0.50
FAIXA_ALERTA = 0.60

# Nomes das tabelas (o DDL esta em sql/01_criar_tabelas.sql)
TABELA_TREINO = "treino_v1"
TABELA_VALIDACAO = "validacao_congelada"
TABELA_PREDICOES = "predicoes"

# Semente unica de toda a disciplina. E o que faz a turma inteira ver os
# mesmos numeros nas quatro aulas.
SEED = 42


# ------------------------------------------------------------- utilidade ----
# O servidor do MLflow e um processo separado: ele NAO le o seu .env.
# A conexao com o banco vai na linha de comando. Para nao digitar a senha
# duas vezes (e errar), rode `python config.py` e copie o comando pronto.
if __name__ == "__main__":
    print("python -m mlflow server"
          " --host 127.0.0.1"
          " --port 5000"
          f" --backend-store-uri {URI_BANCO_MLFLOW}"
          " --default-artifact-root ./mlruns")

# DOIS conjuntos de teste, com papeis diferentes -- um nao substitui o outro:
#
#   validacao_congelada  GUARDA DE REGRESSAO. Nunca muda, nunca e apagada.
#                        Responde "eu quebrei o que ja funcionava?"
#   validacao_atual      CONJUNTO DE ACEITACAO. Representa a populacao que o
#                        modelo atende HOJE. Cresce a cada lote rotulado.
#                        Responde "serve para o mundo de agora?"
TABELA_VALIDACAO_ATUAL = "validacao_atual"

# Ganho minimo de F1 para o candidato substituir o campeao.
# Abaixo disso a troca nao paga o risco de mexer em producao.
GANHO_MINIMO = 0.005