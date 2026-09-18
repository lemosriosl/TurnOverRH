"""
Retreina o modelo incorporando um lote rotulado, e registra como @challenger.

Diferente do onboard_v1.py, aqui existe um .fit() -- e por isso o lineage
sai COMPLETO: o mesmo script que le o dado e o que treina e o que registra,
na mesma execucao. O digest que vai para o MLflow e do dado que o .fit()
realmente recebeu, nao do que alguem acha que recebeu.

    python treinar.py --incorporar lote_aula2_fev \\
                      --gabarito gabarito_aula2_fev.csv \\
                      --motivo data_drift

O lote rotulado e partido em dois:
    70%  entram no treino, junto com o treino_v1
    30%  viram a tabela validacao_atual -- o juiz do mundo novo

Treinar e validar nas mesmas linhas daria um numero bonito e mentiroso.
O candidato entra como @challenger. Quem decide se ele vira producao e o
promover.py -- treinar e promover sao duas decisoes diferentes.
"""
import argparse
import subprocess

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import config
from app import banco

FRACAO_VALIDACAO = 0.30


def commit_atual():
    """Parte do lineage que o onboard da v1 nao tinha: qual codigo treinou."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.RAIZ, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "sem_git"


def carregar_lote_rotulado(lote, gabarito):
    """Junta o que a API pontuou com o gabarito que o RH liberou depois."""
    pontuado = banco.carregar_lote(lote)
    if pontuado.empty:
        raise SystemExit(f"lote '{lote}' nao esta no banco. Envie pelo front primeiro.")

    gab = pd.read_csv(config.PASTA_DADOS / gabarito, sep=";")
    gab[config.COLUNA_ID] = gab[config.COLUNA_ID].astype(str)
    juntos = pontuado.merge(gab, on=config.COLUNA_ID, how="inner")
    if len(juntos) < len(pontuado):
        print(f"  aviso: {len(pontuado) - len(juntos)} linhas sem rotulo, descartadas")
    return juntos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incorporar", required=True, help="nome do lote no banco")
    ap.add_argument("--gabarito", required=True, help="csv em dados/ com o rotulo")
    ap.add_argument("--motivo", required=True,
                    help="por que este retreino existe. Vai como tag no MLflow")
    args = ap.parse_args()

    banco.conferir()
    colunas = [config.COLUNA_ID] + config.FEATURES + [config.COLUNA_ALVO]

    treino_antigo = banco.carregar_conjunto(config.TABELA_TREINO)
    lote = carregar_lote_rotulado(args.incorporar, args.gabarito)[colunas]

    # O lote e partido ANTES de qualquer treino. A parte de validacao nunca
    # e vista pelo .fit().
    lote_treino, lote_validacao = train_test_split(
        lote, test_size=FRACAO_VALIDACAO, random_state=config.SEED,
        stratify=lote[config.COLUNA_ALVO])

    treino = pd.concat([treino_antigo[colunas], lote_treino], ignore_index=True)
    X, y = treino[config.FEATURES], treino[config.COLUNA_ALVO]

    print(f"treino: {len(treino_antigo)} linhas antigas + {len(lote_treino)} do lote "
          f"= {len(treino)}")

    parametros = dict(n_estimators=300, max_depth=6,
                      random_state=config.SEED, n_jobs=-1)
    modelo = RandomForestClassifier(**parametros)
    modelo.fit(X, y)

    # Anexa ao conjunto de aceitacao. ANEXA, nao substitui -- ver banco.py.
    lote_validacao = lote_validacao.assign(origem=args.incorporar)
    novas, total = banco.acumular_validacao_atual(lote_validacao)
    print(f"validacao_atual: +{novas} linhas -> {total} no total")

    # Medida rapida nos dois juizes, so para o log deste run.
    medidas = {}
    for nome, tabela in (("congelada", config.TABELA_VALIDACAO),
                         ("atual", config.TABELA_VALIDACAO_ATUAL)):
        conjunto = banco.carregar_conjunto(tabela)
        pred = modelo.predict(conjunto[config.FEATURES])
        real = conjunto[config.COLUNA_ALVO]
        medidas[f"f1_validacao_{nome}"] = f1_score(real, pred)
        medidas[f"accuracy_validacao_{nome}"] = accuracy_score(real, pred)
        medidas[f"precision_validacao_{nome}"] = precision_score(real, pred)
        medidas[f"recall_validacao_{nome}"] = recall_score(real, pred)

    mlflow.set_tracking_uri(config.URI_TRACKING)
    mlflow.set_experiment(config.NOME_EXPERIMENTO)

    with mlflow.start_run(run_name=f"retreino_{args.incorporar}") as run:
        mlflow.set_tags({
            "tipo": "retreino",
            "motivo": args.motivo,
            "lote_incorporado": args.incorporar,
            "origem": "mlflow",
            "lineage": "completo",      # a diferenca em relacao a v1
            "commit": commit_atual(),
        })

        # Lineage de verdade: os dois datasets que o .fit() realmente viu.
        mlflow.log_input(
            mlflow.data.from_pandas(treino, name=f"treino+{args.incorporar}",
                                    targets=config.COLUNA_ALVO),
            context="training")
        mlflow.log_input(
            mlflow.data.from_pandas(lote_validacao, name=config.TABELA_VALIDACAO_ATUAL,
                                    targets=config.COLUNA_ALVO),
            context="validation")

        mlflow.log_params({
            "tipo_modelo": type(modelo).__name__,
            "n_features": len(config.FEATURES),
            "linhas_treino": len(treino),
            "linhas_do_lote": len(lote_treino),
            "fracao_validacao": FRACAO_VALIDACAO,
            **{k: v for k, v in parametros.items() if k != "n_jobs"},
        })
        mlflow.log_metrics(medidas)

        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="modelo",
            signature=infer_signature(X, modelo.predict(X)),
            input_example=X.head(3),
            registered_model_name=config.NOME_MODELO,
        )
        run_id = run.info.run_id

    cliente = MlflowClient()
    versao = max(cliente.search_model_versions(f"name='{config.NOME_MODELO}'"),
                 key=lambda v: int(v.version))
    cliente.set_registered_model_alias(
        config.NOME_MODELO, config.ALIAS_CANDIDATO, versao.version)

    print(f"\nrun_id  {run_id}")
    print(f"versao  {versao.version}  ->  @{config.ALIAS_CANDIDATO}")
    print("\nmedidas deste candidato:")
    for k, v in medidas.items():
        print(f"  {k:<32} {v:.4f}")
    print(f"\nA producao NAO mudou: @{config.ALIAS_PROD} continua onde estava.")
    print(f"Para decidir:  python promover.py --versao {versao.version}")


if __name__ == "__main__":
    main()