"""
O banco DA APLICACAO. Encanamento -- vem pronto.

Guarda uma coisa so: o log de predicoes. Toda vez que alguem pergunta ao
modelo, fica registrado o que foi perguntado, o que ele respondeu e QUAL
VERSAO respondeu.

Nao guarda dado de treino. Nao guarda metrica de experimento. Isso e MLflow,
que vive no outro banco do mesmo servidor.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values

import config

def conectar():
    """Uma conexao nova por operacao. Simples e suficiente para a turma --
    em producao de verdade entraria um pool (psycopg_pool, SQLAlchemy)."""
    return psycopg2.connect(config.URI_BANCO_APP)


def conferir():
    """As tabelas existem? Quem cria e voce, rodando sql/01_criar_tabelas.sql.
    A aplicacao nao cria schema sozinha -- DDL escondido dentro do app e um
    otimo jeito de descobrir tarde demais que producao mudou de forma."""
    esperadas = {config.TABELA_TREINO, config.TABELA_VALIDACAO,
                 config.TABELA_PREDICOES}
    with conectar() as con, con.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'")
        existentes = {linha[0] for linha in cur.fetchall()}
    faltando = esperadas - existentes
    if faltando:
        raise SystemExit(
            f"faltam tabelas no banco: {sorted(faltando)}\n"
            f"rode: psql -d {config.PG_DB_APP} -f sql/01_criar_tabelas.sql")
    return sorted(esperadas)


def carregar_conjunto(tabela):
    """Le treino_v1 ou validacao_congelada do banco como DataFrame."""
    colunas = [config.COLUNA_ID] + config.FEATURES + [config.COLUNA_ALVO]
    with conectar() as con, con.cursor() as cur:
        cur.execute(f"SELECT {', '.join(colunas)} FROM {tabela}")
        return pd.DataFrame(cur.fetchall(), columns=colunas)


def gravar_conjunto(tabela, df):
    """Substitui o conteudo da tabela pelo DataFrame."""
    colunas = [config.COLUNA_ID] + config.FEATURES + [config.COLUNA_ALVO]
    linhas = [tuple(linha) for linha in df[colunas].itertuples(index=False)]
    with conectar() as con, con.cursor() as cur:
        cur.execute(f"TRUNCATE {tabela}")
        execute_values(cur,
            f"INSERT INTO {tabela} ({', '.join(colunas)}) VALUES %s", linhas)
    return len(linhas)


def salvar_predicoes(lote, versao, ids, probabilidades, classes, entradas):
    """Grava N predicoes numa unica ida ao banco. `entradas` e lista de dicts."""
    linhas = [
        (lote, str(i), str(versao), float(p), int(c), Json(e))
        for i, p, c, e in zip(ids, probabilidades, classes, entradas)
    ]
    with conectar() as con, con.cursor() as cur:
        execute_values(cur,
            "INSERT INTO predicoes "
            "(lote, id_pessoa, versao_modelo, probabilidade, classe, features) "
            "VALUES %s", linhas)
    return len(linhas)


def carregar_lote(nome):
    """DataFrame com as features + predicoes de um lote ja pontuado."""
    with conectar() as con, con.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT DISTINCT ON (id_pessoa) "
            "       id_pessoa, versao_modelo, probabilidade, classe, features "
            "FROM predicoes WHERE lote = %s "
            "ORDER BY id_pessoa, id DESC", (nome,))
        linhas = cur.fetchall()
    if not linhas:
        return pd.DataFrame()

    # O JSONB volta do Postgres ja como dict -- nao precisa de json.loads.
    features = pd.DataFrame([linha["features"] for linha in linhas])
    meta = pd.DataFrame([{
        config.COLUNA_ID: linha["id_pessoa"],
        "versao_modelo": linha["versao_modelo"],
        "probabilidade": linha["probabilidade"],
        "classe": linha["classe"],
    } for linha in linhas])
    return pd.concat([meta, features], axis=1)


def listar_colaboradores(limite=5000):
    """Todos os colaboradores ja pontuados, do mais recente para o mais antigo.
    A faixa de risco sai daqui pronta -- e regra de negocio, nao de front."""
    with conectar() as con, con.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT DISTINCT ON (id_pessoa) "
            "       id_pessoa, lote, versao_modelo, probabilidade, classe, criado_em "
            "FROM predicoes ORDER BY id_pessoa, id DESC")
        pessoas = cur.fetchall()

    def faixa(p):
        if p >= config.FAIXA_ALERTA:
            return "alerta"
        if p >= config.FAIXA_ATENCAO:
            return "atencao"
        return "ok"

    pessoas.sort(key=lambda x: x["probabilidade"], reverse=True)
    return [dict(p) | {"faixa": faixa(p["probabilidade"]),
                       "criado_em": str(p["criado_em"])}
            for p in pessoas[:limite]]


def resumo_por_lote():
    """Totais na tabela inteira, nao na lista truncada dos cards.

    DISTINCT ON (lote, id_pessoa) usa a pontuacao mais recente de cada
    pessoa no lote -- reenviar fevereiro nao duplica a conta.
    """
    alerta, atencao = config.FAIXA_ALERTA, config.FAIXA_ATENCAO
    with conectar() as con, con.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH ultimas AS (
                SELECT DISTINCT ON (lote, id_pessoa)
                       lote, versao_modelo, probabilidade, classe
                FROM predicoes
                ORDER BY lote, id_pessoa, id DESC
            )
            SELECT lote,
                   COUNT(*)::int AS n,
                   AVG(classe::double precision) AS positiva,
                   AVG(probabilidade) AS media,
                   STRING_AGG(DISTINCT versao_modelo, ',') AS versoes,
                   COUNT(*) FILTER (WHERE probabilidade >= %s)::int AS alerta,
                   COUNT(*) FILTER (
                       WHERE probabilidade >= %s AND probabilidade < %s)::int AS atencao,
                   COUNT(*) FILTER (WHERE probabilidade < %s)::int AS ok
            FROM ultimas
            GROUP BY lote
            """,
            (alerta, atencao, alerta, atencao))
        linhas = cur.fetchall()
    return [{
        "lote": r["lote"],
        "n": int(r["n"]),
        "positiva": float(r["positiva"] or 0),
        "media": float(r["media"] or 0),
        "versoes": r["versoes"] or "",
        "alerta": int(r["alerta"]),
        "atencao": int(r["atencao"]),
        "ok": int(r["ok"]),
    } for r in linhas]


def listar_lotes():
    with conectar() as con, con.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT lote, COUNT(*) AS n, MAX(criado_em) AS quando, "
            "       STRING_AGG(DISTINCT versao_modelo, ',') AS versoes "
            "FROM predicoes GROUP BY lote ORDER BY quando DESC")
        return [dict(linha) | {"quando": str(linha["quando"])}
                for linha in cur.fetchall()]

def existe_tabela(nome):
    """O monitorar.py usa para aceitar tabela OU nome de lote na referencia."""
    with conectar() as con, con.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (nome,))
        return cur.fetchone()[0]


def acumular_validacao_atual(df):
    """ANEXA um lote rotulado ao conjunto de aceitacao.

    Acumula de proposito, nao substitui. Se cada mes apagasse o anterior, o
    juiz seria sempre 100% da unidade mais recente -- e superestimaria o ganho
    de um modelo especializado nela.

    ON CONFLICT DO NOTHING: rodar o treino duas vezes nao duplica ninguem.
    """
    colunas = ([config.COLUNA_ID] + config.FEATURES
               + [config.COLUNA_ALVO, "origem"])
    linhas = [tuple(linha) for linha in df[colunas].itertuples(index=False)]
    with conectar() as con, con.cursor() as cur:
        execute_values(cur,
            f"INSERT INTO {config.TABELA_VALIDACAO_ATUAL} "
            f"({', '.join(colunas)}) VALUES %s "
            f"ON CONFLICT ({config.COLUNA_ID}) DO NOTHING", linhas)
        cur.execute(f"SELECT count(*) FROM {config.TABELA_VALIDACAO_ATUAL}")
        total = cur.fetchone()[0]
    return len(linhas), total