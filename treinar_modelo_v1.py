"""
Rede de seguranca do Passo 0.

Use SO SE o seu .pkl da Aula 4 nao aceitar exatamente as 15 features de
config.FEATURES. E o mesmo tipo de modelo que voces treinaram, agora sobre a
tabela padronizada -- exatamente o que o notebook da Aula 4 avisou que ia
acontecer ("o instrutor vai fornecer um csv com as features ja padronizadas").

Repare que este e o UNICO arquivo da disciplina que chama .fit().
Nenhum outro treina nada. Guarde essa observacao para a Aula 1.
"""
import pickle

from sklearn.ensemble import RandomForestClassifier

import config
from app import banco

DESTINO = config.PASTA_ARTEFATOS / "modelo_v1.pkl"


def main():
    treino = banco.carregar_conjunto(config.TABELA_TREINO)
    X = treino[config.FEATURES]
    y = treino[config.COLUNA_ALVO]

    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        random_state=config.SEED, n_jobs=-1)
    modelo.fit(X, y)

    config.PASTA_ARTEFATOS.mkdir(exist_ok=True)
    with open(DESTINO, "wb") as f:
        pickle.dump(modelo, f)

    print(f"modelo salvo em {DESTINO}")
    print(f"treinado em {len(treino)} linhas, {len(config.FEATURES)} features")


if __name__ == "__main__":
    main()
