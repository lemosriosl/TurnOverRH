"""
Popula o banco com os dois conjuntos do modelo.

Le a tabela que voces montaram na Aula 3 e grava, dentro do SEU Postgres,
um split 80/20:

    treino_v1            19.200  o que o modelo estuda
    validacao_congelada   4.800  o juiz. O modelo nunca ve estas linhas

As tabelas ja tem que existir. Se nao existirem, rode antes:
    psql <seu_banco> -U postgres -f sql/01_criar_tabelas.sql

Os lotes mensais NAO saem daqui. Cada aula recebe um arquivo novo, com
pessoas novas -- como o RH mandaria de verdade.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from app import banco

ORIGEM = config.PASTA_DADOS / "features_padronizadas.csv"
PROPORCAO_TESTE = 0.20


def carregar_origem():
    return pd.read_csv(ORIGEM, sep=";").rename(
        columns={"nIdPessoa": config.COLUNA_ID})


def dividir(df=None):
    """Split 80/20 estratificado pelo alvo, com semente fixa.

    Estratificado significa que as duas metades tem a mesma proporcao de
    quem pediu demissao. Sem isso, o acaso poderia deixar a validacao com
    mais desligamentos que o treino e o F1 sairia distorcido.
    """
    if df is None:
        df = carregar_origem()
    treino, validacao = train_test_split(
        df, test_size=PROPORCAO_TESTE, random_state=config.SEED,
        stratify=df[config.COLUNA_ALVO])
    return treino, validacao


def main():
    banco.conferir()

    df = carregar_origem()
    print(f"origem: {len(df)} linhas, alvo em {df[config.COLUNA_ALVO].mean():.1%}")

    treino, validacao = dividir(df)
    n_t = banco.gravar_conjunto(config.TABELA_TREINO, treino)
    n_v = banco.gravar_conjunto(config.TABELA_VALIDACAO, validacao)

    print(f"treino_v1            {n_t:>6} linhas  ({1-PROPORCAO_TESTE:.0%})")
    print(f"validacao_congelada  {n_v:>6} linhas  ({PROPORCAO_TESTE:.0%})")
    print(f"  alvo no treino    {treino[config.COLUNA_ALVO].mean():.1%}")
    print(f"  alvo na validacao {validacao[config.COLUNA_ALVO].mean():.1%}")
    print("\nconfira:  SELECT count(*) FROM treino_v1;")


if __name__ == "__main__":
    main()