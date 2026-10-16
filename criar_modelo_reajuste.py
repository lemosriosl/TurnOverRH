"""
Cria um MODELO NOVO no MLflow -- nao uma versao nova do turnover_rh.

    python criar_modelo_reajuste.py

Versao nova x modelo novo
-------------------------
Ate a Aula 4, toda mudanca virou uma versao do mesmo modelo, `turnover_rh`:
v1, v2, v3, v4. Todas respondem a mesma pergunta -- "esta pessoa vai sair?" --
com a mesma saida: uma probabilidade. Por isso elas competem no mesmo portao e
uma pode substituir a outra pelo alias @champion.

Este script cria `turnover_reajuste`. Outro nome no registry, porque e outra
pergunta: "o reajuste faz diferenca para esta pessoa?". A saida tambem e outra:
tres numeros por pessoa, nao um. Ele nao compete com o turnover_rh, nao passa
pelo promover.py e nao substitui ninguem. Os dois convivem.

Como o modelo responde
----------------------
Duas partes, treinadas separadamente:

    sem reajuste   chance de sair de quem NAO recebeu acao. Treinada com o
                   historico (treino_v1, fevereiro, marco) e com quem ficou
                   em atencao/ok em abril.
    com reajuste   chance de sair de quem RECEBEU o reajuste. Treinada com os
                   tratados de abril -- a faixa de alerta.

    efeito do reajuste = chance sem - chance com

O limite deste piloto
---------------------
Em abril, so a faixa de alerta recebeu reajuste. A parte "com reajuste" nunca
viu ninguem de fora dela, e para essas pessoas a resposta e extrapolacao. O
modelo diz isso: a coluna `confiavel` marca quem esta na regiao em que existe
gente parecida com e sem reajuste. Por isso esta e a versao 1 de um piloto
(@piloto), e a versao 2 depende do sorteio de julho.

Nao altera nenhum arquivo do projeto. Os nomes ficam aqui no topo, e nao no
config.py, so para nao mexer em arquivo existente -- num projeto que adota o
modelo, eles iriam para o config.
"""
import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config
from app import banco

# ---------------------------------------------------------- o modelo novo ----
NOME_MODELO_NOVO = "turnover_reajuste"        # outro nome no registry
EXPERIMENTO_NOVO = "turnover_reajuste"        # outra gaveta de runs
ALIAS_PILOTO = "piloto"                       # nao e @champion: nao atende a API
PERGUNTA = "o reajuste de retencao faz diferenca para esta pessoa?"

# As 15 colunas do arquivo, menos as que estao mortas (a hora extra desde abril,
# a hi_minutos_extras desde sempre). O modelo novo ja nasce sem elas.
MORTAS = ["he_eventos", "he_referencia_total", "he_valor_total", "hi_minutos_extras"]
FEATURES = [c for c in config.FEATURES if c not in MORTAS]
ALVO = config.COLUNA_ALVO
PARAMETROS = dict(n_estimators=300, max_depth=6, min_samples_leaf=20,
                  random_state=config.SEED, n_jobs=-1)


class ModeloReajuste(mlflow.pyfunc.PythonModel):
    """Duas florestas e uma regra de confianca, empacotadas como UM modelo.

    O MLflow sabe guardar um RandomForest sozinho (mlflow.sklearn). Um modelo
    feito de duas partes e uma regra precisa de um `pyfunc`: uma classe com
    `predict`, que o MLflow serializa inteira e devolve igual no load.
    """

    def __init__(self, sem, com, features, limite_confianca):
        self.sem = sem
        self.com = com
        self.features = features
        self.limite_confianca = limite_confianca

    def predict(self, context, model_input, params=None):
        X = model_input[self.features]
        p_sem = self.sem.predict_proba(X)[:, 1]
        p_com = self.com.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "prob_sair_sem_reajuste": p_sem,
            "prob_sair_com_reajuste": p_com,
            "efeito_reajuste": p_sem - p_com,
            # Confiavel = parecido com quem recebeu reajuste em abril. Fora
            # disso, a parte "com reajuste" esta extrapolando.
            "confiavel": p_sem >= self.limite_confianca,
        })


def com_gabarito(lote, arquivo):
    df = banco.carregar_lote(lote)
    gab = pd.read_csv(config.PASTA_DADOS / arquivo, sep=";")
    gab[config.COLUNA_ID] = gab[config.COLUNA_ID].astype(str)
    return df.merge(gab, on=config.COLUNA_ID)


def main():
    # ------------------------------------------------------------- dados ----
    historico = pd.concat([
        banco.carregar_conjunto(config.TABELA_TREINO),
        com_gabarito("lote_aula2_fev", "gabarito_aula2_fev.csv"),
        com_gabarito("lote_aula3_mar_corrigido", "gabarito_aula3_mar.csv"),
    ], ignore_index=True)

    abril = com_gabarito("lote_aula4_abr", "gabarito_aula4_abr.csv")
    reajuste = pd.read_csv(config.PASTA_DADOS / "intervencoes_aula4_abr.csv", sep=";")
    abril["reajuste"] = abril[config.COLUNA_ID].isin(
        reajuste[config.COLUNA_ID].astype(str))

    # 30% de abril fica fora do treino, para medir. O sorteio respeita quem
    # recebeu reajuste e quem saiu: as duas partes tem gente para ser medida.
    estrato = abril["reajuste"].astype(str) + abril[ALVO].astype(str)
    abril_treino, abril_teste = train_test_split(
        abril, test_size=0.30, random_state=config.SEED, stratify=estrato)

    treino_sem = pd.concat([historico, abril_treino[~abril_treino["reajuste"]]],
                           ignore_index=True)
    treino_com = abril_treino[abril_treino["reajuste"]]

    # ------------------------------------------------------------ treino ----
    sem = RandomForestClassifier(**PARAMETROS).fit(treino_sem[FEATURES], treino_sem[ALVO])
    com = RandomForestClassifier(**PARAMETROS).fit(treino_com[FEATURES], treino_com[ALVO])

    # A regiao confiavel: tao arriscado, sem acao, quanto os 5% menos
    # arriscados de quem recebeu reajuste.
    limite = float(np.quantile(sem.predict_proba(treino_com[FEATURES])[:, 1], 0.05))
    modelo = ModeloReajuste(sem, com, FEATURES, limite)

    # ------------------------------------------------------------ medida ----
    teste_sem = abril_teste[~abril_teste["reajuste"]]
    teste_com = abril_teste[abril_teste["reajuste"]]
    saida_teste_com = modelo.predict(None, teste_com)
    metricas = {
        "auc_sem_reajuste": roc_auc_score(
            teste_sem[ALVO], sem.predict_proba(teste_sem[FEATURES])[:, 1]),
        "saida_real_com_reajuste": teste_com[ALVO].mean(),
        "saida_prevista_com_reajuste": saida_teste_com["prob_sair_com_reajuste"].mean(),
        "saida_prevista_sem_reajuste_nos_tratados": saida_teste_com["prob_sair_sem_reajuste"].mean(),
        "efeito_medio_nos_tratados": saida_teste_com["efeito_reajuste"].mean(),
        "limite_confianca": limite,
    }

    # ----------------------------------------------------------- MLflow ----
    mlflow.set_tracking_uri(config.URI_TRACKING)
    mlflow.set_experiment(EXPERIMENTO_NOVO)
    exemplo = abril_teste[FEATURES].head(3)

    with mlflow.start_run(run_name="reajuste_v1_piloto") as run:
        mlflow.set_tags({
            "tipo": "modelo_novo",
            "pergunta": PERGUNTA,
            "origem": "aula5",
            "limitacao": "tratados so na faixa de alerta da v3 de abril",
        })
        mlflow.log_params({
            "n_features": len(FEATURES),
            "linhas_sem_reajuste": len(treino_sem),
            "linhas_com_reajuste": len(treino_com),
            **{k: v for k, v in PARAMETROS.items() if k != "n_jobs"},
        })
        mlflow.log_metrics(metricas)
        mlflow.pyfunc.log_model(
            artifact_path="modelo",
            python_model=modelo,
            signature=infer_signature(exemplo, modelo.predict(None, exemplo)),
            input_example=exemplo,
            registered_model_name=NOME_MODELO_NOVO,
        )
        run_id = run.info.run_id

    cliente = MlflowClient()
    versao = max(cliente.search_model_versions(f"name='{NOME_MODELO_NOVO}'"),
                 key=lambda v: int(v.version)).version
    cliente.update_registered_model(
        NOME_MODELO_NOVO,
        description=(f"Pergunta: {PERGUNTA} Saida: chance de sair sem e com reajuste, "
                     "a diferenca entre elas e se a pessoa esta na regiao confiavel. "
                     "Nao substitui o turnover_rh: responde outra pergunta."))
    cliente.set_registered_model_tag(NOME_MODELO_NOVO, "pergunta", PERGUNTA)
    cliente.set_registered_model_alias(NOME_MODELO_NOVO, ALIAS_PILOTO, versao)
    cliente.set_model_version_tag(NOME_MODELO_NOVO, versao, "fase", "piloto")

    print(f"modelo novo  {NOME_MODELO_NOVO}  (experimento {EXPERIMENTO_NOVO})")
    print(f"run_id       {run_id}")
    print(f"versao       {versao}  ->  @{ALIAS_PILOTO}")
    print(f"pergunta     {PERGUNTA}\n")
    print(f"treino sem reajuste  {len(treino_sem):>6} linhas")
    print(f"treino com reajuste  {len(treino_com):>6} linhas (so a faixa de alerta de abril)\n")
    print("medidas no teste de abril (30%, nunca vistos):")
    for k, v in metricas.items():
        print(f"  {k:<42} {v:.4f}")

    # --------------------------------- usar: pelo registry, pelo alias ----
    carregado = mlflow.pyfunc.load_model(f"models:/{NOME_MODELO_NOVO}@{ALIAS_PILOTO}")
    julho = pd.read_csv(config.PASTA_DADOS / "lote_aula5_jul.csv", sep=";")
    resposta = carregado.predict(julho)
    resposta.insert(0, config.COLUNA_ID, julho[config.COLUNA_ID])

    confiavel = resposta["confiavel"]
    print(f"\njulho pelo models:/{NOME_MODELO_NOVO}@{ALIAS_PILOTO}  ({len(julho)} pessoas)")
    print(f"  na regiao confiavel   {confiavel.sum():>5}   efeito medio "
          f"{resposta.loc[confiavel, 'efeito_reajuste'].mean():.3f}")
    print(f"  fora dela             {(~confiavel).sum():>5}   o modelo nao sabe: "
          f"ninguem parecido recebeu reajuste em abril")
    print("\nos 5 com maior efeito estimado do reajuste:")
    print(resposta.sort_values("efeito_reajuste", ascending=False).head(5)
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
