"""
O portao. Decide se o @challenger vira @champion.

    python promover.py --versao 2
    python promover.py --versao 2 --validacao validacao_atual
    python promover.py --versao 3 --validacao validacao_atual --zerar he_eventos he_referencia_total he_valor_total

Regra unica e nao negociavel: os dois modelos sao medidos NO MESMO CONJUNTO.
Comparar em conjuntos diferentes e a forma mais comum de promover um modelo
pior sem perceber.

Aula 4: cada modelo e medido com as PROPRIAS colunas (feature_names_in_), e o
--zerar mede os dois no mundo em que a coluna morreu: o conjunto rotulado com
essas colunas zeradas, como elas chegam hoje em producao.

Qual conjunto usar, isso sim e decisao sua -- e fica registrada na tag
`validado_em` da versao promovida. Um conjunto de teste representa uma
populacao; se a populacao mudou, ele envelheceu junto com ela.
"""
import argparse

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from app import banco

CONJUNTOS = {
    "validacao_congelada": config.TABELA_VALIDACAO,
    "validacao_atual": config.TABELA_VALIDACAO_ATUAL,
}


def medir(modelo, conjunto):
    # Cada modelo pega as colunas com que FOI treinado, na ordem em que foi
    # treinado. O sklearn guarda essa lista no proprio objeto desde o .fit().
    # Ate a Aula 3 isto era config.FEATURES -- e funcionava porque todas as
    # versoes comiam as mesmas 15 colunas. A v3 come 12.
    pred = modelo.predict(conjunto[list(modelo.feature_names_in_)])
    real = conjunto[config.COLUNA_ALVO]
    return {
        "f1": f1_score(real, pred),
        "accuracy": accuracy_score(real, pred),
        "precision": precision_score(real, pred),
        "recall": recall_score(real, pred),
        "taxa_predicao_positiva": float(pred.mean()),
    }


def simular(conjunto, zerar):
    """Devolve o conjunto como ele chegaria HOJE em producao.

    As colunas em `zerar` morreram na origem: vem no arquivo, mas 100% zeradas.
    O campeao continua lendo essas colunas -- e e com elas zeradas que ele vai
    trabalhar a partir de agora. Medir o campeao no conjunto intacto seria
    medir um modelo num mundo que nao existe mais.

    O rotulo do mes novo leva tres meses. O conjunto rotulado que ja existe,
    com as colunas zeradas, e o melhor retrato de abril que da para ter hoje.
    """
    if not zerar:
        return conjunto
    return conjunto.assign(**{coluna: 0.0 for coluna in zerar})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versao", required=True, help="versao candidata")
    ap.add_argument("--validacao", default="validacao_congelada",
                    choices=list(CONJUNTOS))
    ap.add_argument("--zerar", nargs="+", default=[], metavar="COLUNA",
                    help="colunas que chegam zeradas em producao: os dois "
                         "modelos sao medidos nesse mundo")
    ap.add_argument("--forcar", action="store_true",
                    help="promove mesmo reprovado. Fica registrado como tal")
    args = ap.parse_args()
    desconhecidas = [c for c in args.zerar if c not in config.FEATURES]
    if desconhecidas:
        raise SystemExit(f"--zerar com coluna que nao existe: {desconhecidas}")

    mlflow.set_tracking_uri(config.URI_TRACKING)
    # Sem set_experiment, o run cai no experimento "Default" -- fora da gaveta
    # onde vive o resto do historico do modelo. A decisao de promocao faz
    # parte desse historico tanto quanto o treino.
    mlflow.set_experiment(config.NOME_EXPERIMENTO)
    cliente = MlflowClient()

    campeao_versao = cliente.get_model_version_by_alias(
        config.NOME_MODELO, config.ALIAS_PROD).version
    if campeao_versao == args.versao:
        raise SystemExit(f"a versao {args.versao} JA e o @{config.ALIAS_PROD}.")

    # Carrega os dois modelos POR NUMERO DE VERSAO, nao por alias: precisamos
    # do campeao atual e do candidato, e o alias do candidato pode nem existir.
    campeao = mlflow.sklearn.load_model(
        f"models:/{config.NOME_MODELO}/{campeao_versao}")
    candidato = mlflow.sklearn.load_model(
        f"models:/{config.NOME_MODELO}/{args.versao}")

    # Mede nos DOIS conjuntos, sempre. Promover e uma troca: o numero que
    # voce ganha de um lado costuma custar algo do outro, e esconder o custo
    # e a forma mais elegante de tomar uma decisao ruim.
    resultado = {}
    for nome, tabela_sql in CONJUNTOS.items():
        conjunto = banco.carregar_conjunto(tabela_sql)
        if conjunto.empty:
            continue
        mundo = simular(conjunto, args.zerar)
        resultado[nome] = {
            "linhas": len(conjunto),
            "alvo": conjunto[config.COLUNA_ALVO].mean(),
            "campeao": medir(campeao, mundo),
            "candidato": medir(candidato, mundo),
        }
        if args.zerar:
            # So para o registro: quanto o campeao perdeu quando a coluna
            # morreu. NAO entra na decisao -- esse mundo acabou.
            resultado[nome]["campeao_antes"] = medir(campeao, conjunto)
    if args.validacao not in resultado:
        raise SystemExit(f"'{args.validacao}' esta vazio. "
                         f"Rode o treinar.py antes.")

    for nome, r in resultado.items():
        decide = " <- DECIDE" if nome == args.validacao else ""
        print(f"\n{nome}  ({r['linhas']} linhas, alvo em {r['alvo']:.1%})"
              f"{decide}")
        colunas = {}
        if "campeao_antes" in r:
            colunas[f"v{campeao_versao} antes"] = r["campeao_antes"]
            colunas[f"v{campeao_versao} zerada"] = r["campeao"]
        else:
            colunas[f"campeao v{campeao_versao}"] = r["campeao"]
        colunas[f"candidato v{args.versao}"] = r["candidato"]
        print(pd.DataFrame(colunas).round(4).to_string())
        d = r["candidato"]["f1"] - r["campeao"]["f1"]
        print(f"  ganho de f1   {d:+.4f}")

    ganho = (resultado[args.validacao]["candidato"]["f1"]
             - resultado[args.validacao]["campeao"]["f1"])
    aprovado = ganho >= config.GANHO_MINIMO
    veredito = "APROVADO" if aprovado else "REPROVADO"

    print(f"\n{'-' * 58}")
    print(f"decidido em      {args.validacao}")
    if args.zerar:
        print(f"com zeradas      {', '.join(args.zerar)}")
    print(f"ganho de f1      {ganho:+.4f}   (minimo exigido "
          f"{config.GANHO_MINIMO:+.4f})")
    print(f"veredito         {veredito}")
    outro = next((n for n in resultado if n != args.validacao), None)
    if outro:
        d = (resultado[outro]["candidato"]["f1"]
             - resultado[outro]["campeao"]["f1"])
        rotulo = "ganho" if d >= 0 else "custo"
        print(f"{rotulo} em {outro}   {d:+.4f}   <- a outra metade da decisao")

    with mlflow.start_run(run_name=f"promocao_v{args.versao}_{args.validacao}"):
        mlflow.set_tags({
            "tipo": "promocao",
            "versao_candidata": args.versao,
            "versao_campeao": campeao_versao,
            "validado_em": args.validacao,
            "veredito": veredito.lower(),
            "forcado": str(args.forcar).lower(),
            "simulado_com_zeradas": ",".join(args.zerar) or "nenhuma",
        })
        mlflow.log_params({
            "validacao": args.validacao,
            "linhas_validacao": resultado[args.validacao]["linhas"],
            "ganho_minimo": config.GANHO_MINIMO})
        # Metricas dos dois conjuntos no mesmo run: quem abrir isso em seis
        # meses ve o ganho E o custo, nao so o numero que justificou a decisao.
        for nome, r in resultado.items():
            mlflow.log_metrics({
                **{f"{nome}_campeao_{k}": v for k, v in r["campeao"].items()},
                **{f"{nome}_candidato_{k}": v for k, v in r["candidato"].items()},
                f"{nome}_ganho_f1": (r["candidato"]["f1"] - r["campeao"]["f1"]),
            })
            if "campeao_antes" in r:
                mlflow.log_metrics({f"{nome}_campeao_antes_{k}": v
                                    for k, v in r["campeao_antes"].items()})
        mlflow.log_metric("ganho_f1", ganho)

    if not aprovado and not args.forcar:
        print(f"\nA producao NAO mudou. @{config.ALIAS_PROD} segue na "
              f"v{campeao_versao}.")
        print("Antes de forcar, responda: este conjunto representa a "
              "populacao que o modelo vai atender?")
        return

    cliente.set_registered_model_alias(
        config.NOME_MODELO, config.ALIAS_PROD, args.versao)
    cliente.set_model_version_tag(
        config.NOME_MODELO, args.versao, "validado_em", args.validacao)
    cliente.delete_registered_model_alias(
        config.NOME_MODELO, config.ALIAS_CANDIDATO)

    print(f"\n@{config.ALIAS_PROD}  v{campeao_versao} -> v{args.versao}")
    print("Clique em 'Recarregar modelo' no front para a API pegar a troca.")


if __name__ == "__main__":
    main()