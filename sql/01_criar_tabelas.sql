-- =====================================================================
-- Turnover RH -- criacao das tabelas da aplicacao
--
-- Onde o banco vive e escolha sua: Docker, instalado na maquina ou nuvem.
-- Rode assim (o -U postgres e obrigatorio: e o superusuario padrao):
--
--     createdb turnover -U postgres
--     createdb mlflow   -U postgres
--     psql turnover -U postgres -f sql/01_criar_tabelas.sql
--     psql turnover -U postgres -c "\\dt"
--
-- Rodar duas vezes nao quebra nada: e IF NOT EXISTS.
--
-- Sao tres tabelas com tres papeis diferentes:
--   treino_v1            o que o modelo estudou
--   validacao_congelada  o conjunto de teste, congelado. O modelo nunca viu
--   predicoes            o que o modelo respondeu em producao
--
-- As duas primeiras sao preenchidas pelo preparar_dados.py.
-- A terceira e preenchida pela API, uma linha por pergunta feita ao modelo.
-- =====================================================================

-- ------------------------------------------------- o que o modelo viu ----
CREATE TABLE IF NOT EXISTS treino_v1 (
    id_pessoa                    TEXT PRIMARY KEY,
    abs_eventos                  DOUBLE PRECISION NOT NULL DEFAULT 0,
    abs_qtd_total                DOUBLE PRECISION NOT NULL DEFAULT 0,
    horas_previstas_total        DOUBLE PRECISION NOT NULL DEFAULT 0,
    acidentes_eventos            DOUBLE PRECISION NOT NULL DEFAULT 0,
    acidentes_com_afastamento    DOUBLE PRECISION NOT NULL DEFAULT 0,
    acidentes_dias_perdidos      DOUBLE PRECISION NOT NULL DEFAULT 0,
    he_eventos                   DOUBLE PRECISION NOT NULL DEFAULT 0,
    he_referencia_total          DOUBLE PRECISION NOT NULL DEFAULT 0,
    he_valor_total               DOUBLE PRECISION NOT NULL DEFAULT 0,
    hi_eventos                   DOUBLE PRECISION NOT NULL DEFAULT 0,
    hi_minutos_irregulares       DOUBLE PRECISION NOT NULL DEFAULT 0,
    hi_minutos_extras            DOUBLE PRECISION NOT NULL DEFAULT 0,
    mov_sal_eventos              DOUBLE PRECISION NOT NULL DEFAULT 0,
    mov_sal_valor_total          DOUBLE PRECISION NOT NULL DEFAULT 0,
    mov_sal_perc_medio           DOUBLE PRECISION NOT NULL DEFAULT 0,
    pediu_para_sair              SMALLINT NOT NULL
);

-- ----------------------------- o teste, congelado: o modelo nunca viu ----
CREATE TABLE IF NOT EXISTS validacao_congelada (
    id_pessoa                    TEXT PRIMARY KEY,
    abs_eventos                  DOUBLE PRECISION NOT NULL DEFAULT 0,
    abs_qtd_total                DOUBLE PRECISION NOT NULL DEFAULT 0,
    horas_previstas_total        DOUBLE PRECISION NOT NULL DEFAULT 0,
    acidentes_eventos            DOUBLE PRECISION NOT NULL DEFAULT 0,
    acidentes_com_afastamento    DOUBLE PRECISION NOT NULL DEFAULT 0,
    acidentes_dias_perdidos      DOUBLE PRECISION NOT NULL DEFAULT 0,
    he_eventos                   DOUBLE PRECISION NOT NULL DEFAULT 0,
    he_referencia_total          DOUBLE PRECISION NOT NULL DEFAULT 0,
    he_valor_total               DOUBLE PRECISION NOT NULL DEFAULT 0,
    hi_eventos                   DOUBLE PRECISION NOT NULL DEFAULT 0,
    hi_minutos_irregulares       DOUBLE PRECISION NOT NULL DEFAULT 0,
    hi_minutos_extras            DOUBLE PRECISION NOT NULL DEFAULT 0,
    mov_sal_eventos              DOUBLE PRECISION NOT NULL DEFAULT 0,
    mov_sal_valor_total          DOUBLE PRECISION NOT NULL DEFAULT 0,
    mov_sal_perc_medio           DOUBLE PRECISION NOT NULL DEFAULT 0,
    pediu_para_sair              SMALLINT NOT NULL
);

-- ------------------------------- o log de producao: quem respondeu o que ----
CREATE TABLE IF NOT EXISTS predicoes (
    id             BIGSERIAL PRIMARY KEY,
    criado_em      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    lote           TEXT             NOT NULL,   -- 'avulso' ou o nome do csv
    id_pessoa      TEXT,
    versao_modelo  TEXT             NOT NULL,   -- QUAL versao respondeu
    probabilidade  DOUBLE PRECISION NOT NULL,
    classe         SMALLINT         NOT NULL,
    features       JSONB            NOT NULL    -- a entrada exata recebida
);

CREATE INDEX IF NOT EXISTS idx_predicoes_lote      ON predicoes (lote);
CREATE INDEX IF NOT EXISTS idx_predicoes_criado_em ON predicoes (criado_em DESC);


-- =====================================================================
-- Para recomecar do zero, descomente e rode:
--
-- DROP TABLE IF EXISTS treino_v1, validacao_congelada, predicoes;
-- =====================================================================
