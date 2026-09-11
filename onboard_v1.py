"""
Adota um .pkl que ja existe como versao 1 do modelo no MLflow.

NAO TREINA NADA. Procure um .fit() aqui dentro: nao tem.
O que este script adiciona ao arquivo solto sao tres coisas que ele nao tinha:

    identidade  -> numero de versao + alias @champion
    contrato    -> signature (quais colunas, de que tipo, em que ordem)
    baseline    -> metricas recalculadas na validacao congelada

Uso:
    python onboard_v1.py --pkl artefatos/modelo_v1.pkl
"""
import argparse
import pickle

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from app import banco


def conferir_contrato(modelo):
    """O pkl da Aula 4 aceita as 15 features padronizadas?"""
    esperado = list(config.FEATURES)
    recebido = list(getattr(modelo, "feature_names_in_", esperado))
    if recebido != esperado:
        faltando = [c for c in esperado if c not in recebido]
        sobrando = [c for c in recebido if c not in esperado]
        raise SystemExit(
            "Seu modelo nao fala a mesma lingua que config.FEATURES.\n"
            f"  faltando: {faltando}\n"
            f"  sobrando: {sobrando}\n"
            "Rode `python treinar_modelo_v1.py` para gerar o pkl padronizado.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(config.PASTA_ARTEFATOS / "modelo_v1.pkl"))
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        modelo = pickle.load(f)
    conferir_contrato(modelo)

    # A validacao congelada e o juiz oficial da v1. O modelo nunca a viu.
    banco.conferir()
    validacao = banco.carregar_conjunto(config.TABELA_VALIDACAO)
    X = validacao[config.FEATURES]
    y = validacao[config.COLUNA_ALVO]

    # Isto aqui e `predict`, nao `fit`. Estamos medindo, nao aprendendo.
    y_pred = modelo.predict(X)
    metricas = {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "taxa_predicao_positiva": float(y_pred.mean()),
    }

    mlflow.set_tracking_uri(config.URI_TRACKING)
    mlflow.set_experiment(config.NOME_EXPERIMENTO)

    with mlflow.start_run(run_name="v1_onboarding_do_pkl") as run:
        # Tags = etiquetas para filtrar runs depois. Estas duas sao uma
        # confissao honesta: este modelo nasceu fora do MLflow.
        mlflow.set_tags({
            "tipo": "onboarding",
            "origem": "pre_mlflow",
            "lineage": "incompleto",
            "arquivo_origem": args.pkl,
        })

        # Lineage possivel: apontamos o dataset a mao. Ninguem garante que
        # foi ESTE arquivo que gerou o pkl -- e esse e o ponto da aula.
        treino = banco.carregar_conjunto(config.TABELA_TREINO)
        mlflow.log_input(
            mlflow.data.from_pandas(treino, name=config.TABELA_TREINO,
                                    targets=config.COLUNA_ALVO),
            context="training")

        # Params: entrada do treino. Sairam do objeto sklearn, que ainda
        # carrega os hiperparametros consigo.
        mlflow.log_params({
            "tipo_modelo": type(modelo).__name__,
            "n_features": len(config.FEATURES),
            **{k: v for k, v in modelo.get_params().items()
               if k in ("n_estimators", "max_depth", "random_state", "criterion")},
        })

        # Metrics: saida da medicao.
        mlflow.log_metrics(metricas)

        assinatura = infer_signature(X, y_pred)
        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="modelo",
            signature=assinatura,
            input_example=X.head(3),
            registered_model_name=config.NOME_MODELO,
        )
        run_id = run.info.run_id

    cliente = MlflowClient()
    versao = max(
        (v for v in cliente.search_model_versions(f"name='{config.NOME_MODELO}'")),
        key=lambda v: int(v.version))
    cliente.set_registered_model_alias(
        config.NOME_MODELO, config.ALIAS_PROD, versao.version)
    cliente.set_model_version_tag(
        config.NOME_MODELO, versao.version, "validado_em", "validacao_congelada")

    print(f"\nrun_id  {run_id}")
    print(f"versao  {versao.version}  ->  @{config.ALIAS_PROD}")
    print("\nbaseline na validacao congelada:")
    for k, v in metricas.items():
        print(f"  {k:<24} {v:.4f}")
    print(f"\nA producao agora resolve models:/{config.NOME_MODELO}"
          f"@{config.ALIAS_PROD} para a versao {versao.version}.")


if __name__ == "__main__":
    main()