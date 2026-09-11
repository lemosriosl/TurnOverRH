"""
A API de producao.

A linha mais importante do arquivo inteiro:

    mlflow.sklearn.load_model(f"models:/{NOME_MODELO}@{ALIAS_PROD}")

Repare no que NAO esta ai: nenhum caminho de .pkl, nenhum numero de versao.
A aplicacao pede "o modelo que estiver como champion". Quem decide qual e ele
e o Model Registry -- por isso promover uma versao troca a producao sem que
ninguem edite uma linha de codigo.
"""
import io
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config              # noqa: E402
from app import banco      # noqa: E402

ESTATICOS = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def ciclo_de_vida(app):
    """Roda uma vez, no boot: confere o banco e busca o modelo no registry."""
    banco.conferir()
    carregar_modelo()
    yield


app = FastAPI(title="Turnover RH — producao", lifespan=ciclo_de_vida)

# app.mount serve arquivos de disco como arquivos de disco: o navegador pede
# /static/estilo.css e recebe o arquivo, sem o servidor montar nada.
app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")

# Estado do processo: o modelo vive na memoria, carregado uma vez.
MODELO = None
VERSAO = None
COLUNAS_CONTRATO = None


def carregar_modelo():
    """Le o registry pelo ALIAS e traz para a memoria. Chamado no boot e no
    botao 'Recarregar modelo'."""
    global MODELO, VERSAO, COLUNAS_CONTRATO
    mlflow.set_tracking_uri(config.URI_TRACKING)
    uri = f"models:/{config.NOME_MODELO}@{config.ALIAS_PROD}"
    try:
        # A signature registrada e o contrato de entrada. Guardamos as colunas
        # para recusar CSV fora do formato -- isso vale ouro na Aula 4.
        info = mlflow.models.get_model_info(uri)
        COLUNAS_CONTRATO = [c["name"] for c in info.signature.inputs.to_dict()]

        # sklearn (e nao pyfunc) porque precisamos de predict_proba: o front
        # mostra probabilidade, nao 0/1. pyfunc so expoe `predict`.
        MODELO = mlflow.sklearn.load_model(uri)

        versao = MlflowClient().get_model_version_by_alias(
            config.NOME_MODELO, config.ALIAS_PROD)
        VERSAO = versao.version
        print(f"[modelo] versao {VERSAO} carregada de {uri}")
    except Exception as erro:
        MODELO, VERSAO, COLUNAS_CONTRATO = None, None, None
        print(f"[modelo] nenhum modelo em producao ({type(erro).__name__}). "
              f"Rode onboard_v1.py.")


class Colaborador(BaseModel):
    """Uma linha de entrada. Os nomes das features batem com config.FEATURES."""
    id_pessoa: str | None = None
    abs_eventos: float = 0
    abs_qtd_total: float = 0
    horas_previstas_total: float = 0
    acidentes_eventos: float = 0
    acidentes_com_afastamento: float = 0
    acidentes_dias_perdidos: float = 0
    he_eventos: float = 0
    he_referencia_total: float = 0
    he_valor_total: float = 0
    hi_eventos: float = 0
    hi_minutos_irregulares: float = 0
    hi_minutos_extras: float = 0
    mov_sal_eventos: float = 0
    mov_sal_valor_total: float = 0
    mov_sal_perc_medio: float = 0


def faixa_de(prob):
    """ok / atencao / alerta. Regra de negocio, calculada no backend."""
    if prob >= config.FAIXA_ALERTA:
        return "alerta"
    if prob >= config.FAIXA_ATENCAO:
        return "atencao"
    return "ok"


def pontuar(df):
    """Aplica o modelo. Nunca chama fit -- treinar em producao e outra coisa."""
    X = df[config.FEATURES]
    prob = MODELO.predict_proba(X)[:, 1]
    classe = (prob >= config.LIMIAR_DECISAO).astype(int)
    return prob, classe


@app.get("/")
def pagina():
    return FileResponse(ESTATICOS / "index.html")


@app.get("/saude")
def saude():
    return {
        "modelo_carregado": MODELO is not None,
        "versao_modelo": VERSAO,
        "alias": config.ALIAS_PROD,
        "modelo": config.NOME_MODELO,
        "limiar": config.LIMIAR_DECISAO,
        "n_features": len(COLUNAS_CONTRATO) if COLUNAS_CONTRATO else None,
        "faixa_atencao": config.FAIXA_ATENCAO,
        "faixa_alerta": config.FAIXA_ALERTA,
        "lotes": banco.listar_lotes(),
    }


@app.get("/colaboradores")
def colaboradores():
    """Todos os colaboradores ja pontuados, com a faixa de risco calculada.

    A faixa sai do backend de proposito: onde comeca o 'alerta' e regra de
    negocio. Se ela morasse no javascript, mudar a politica de retencao viraria
    um deploy de front."""
    return {"colaboradores": banco.listar_colaboradores()}


@app.post("/recarregar")
def recarregar():
    carregar_modelo()
    return saude()


@app.post("/prever")
def prever(colaborador: Colaborador):
    if MODELO is None:
        return JSONResponse({"erro": "nenhum modelo em producao"}, status_code=503)

    dados = colaborador.model_dump()
    # O id nao e feature: sai do dicionario antes de virar linha do modelo.
    id_pessoa = dados.pop("id_pessoa") or f"avulso-{datetime.now():%H%M%S}"

    df = pd.DataFrame([dados])
    prob, classe = pontuar(df)

    banco.salvar_predicoes(
        lote="avulso", versao=VERSAO, ids=[id_pessoa],
        probabilidades=prob, classes=classe, entradas=[dados])

    return {
        "id_pessoa": id_pessoa,
        "probabilidade": round(float(prob[0]), 4),
        "classe": int(classe[0]),
        "faixa": faixa_de(float(prob[0])),
        "limiar": config.LIMIAR_DECISAO,
        "versao_modelo": VERSAO,
    }


@app.post("/lote")
async def lote(arquivo: UploadFile = File(...)):
    if MODELO is None:
        return JSONResponse({"erro": "nenhum modelo em producao"}, status_code=503)

    conteudo = await arquivo.read()
    df = pd.read_csv(io.BytesIO(conteudo), sep=";")

    # O contrato: se faltar coluna, o arquivo e recusado INTEIRO.
    faltando = [c for c in COLUNAS_CONTRATO if c not in df.columns]
    if faltando:
        return JSONResponse({
            "erro": "contrato violado -- este CSV nao serve para a versao em producao",
            "versao_modelo": VERSAO,
            "colunas_faltando": faltando,
        }, status_code=422)

    prob, classe = pontuar(df)
    nome = Path(arquivo.filename).stem
    ids = df[config.COLUNA_ID] if config.COLUNA_ID in df.columns else range(len(df))

    banco.salvar_predicoes(
        lote=nome, versao=VERSAO, ids=list(ids),
        probabilidades=prob, classes=classe,
        entradas=df[config.FEATURES].to_dict(orient="records"))

    return {
        "lote": nome,
        "linhas": len(df),
        "versao_modelo": VERSAO,
        "taxa_predicao_positiva": round(float(classe.mean()), 4),
        "probabilidade_media": round(float(prob.mean()), 4),
    }