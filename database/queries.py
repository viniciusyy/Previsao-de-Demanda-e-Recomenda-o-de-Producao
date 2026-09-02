# ============================================================
# CONSULTA PRINCIPAL DOS DADOS HISTÓRICOS
# ============================================================

QUERY_HISTORICAL_DATA = """
SELECT
    o.id_operacao,
    f.codigo AS feira,

    o.data_producao,
    o.data_venda,

    p.id_produto,
    p.nome AS produto,

    c.nome AS categoria,

    rp.quantidade_produzida,
    rp.quantidade_sobra,
    rp.quantidade_vendida,

    o.clima,

    o.eh_feriado,
    o.nome_feriado,

    o.observacoes

FROM operacoes AS o

INNER JOIN feiras AS f
    ON f.id_feira = o.id_feira

INNER JOIN registros_producao AS rp
    ON rp.id_operacao = o.id_operacao

INNER JOIN produtos AS p
    ON p.id_produto = rp.id_produto

INNER JOIN categorias AS c
    ON c.id_categoria = p.id_categoria

ORDER BY
    o.data_venda ASC,
    f.codigo ASC,
    p.id_produto ASC;
"""


# ============================================================
# VERIFICAÇÃO DE INTEGRIDADE REFERENCIAL
# ============================================================


QUERY_REFERENTIAL_INTEGRITY = """
SELECT
    'FEIRA_INEXISTENTE' AS tipo,
    o.id_operacao AS id_operacao,
    o.id_feira AS id_feira,
    NULL AS id_produto,
    NULL AS id_categoria,
    'Operação referencia uma feira inexistente.' AS descricao

FROM operacoes AS o

LEFT JOIN feiras AS f
    ON f.id_feira = o.id_feira

WHERE f.id_feira IS NULL


UNION ALL


SELECT
    'OPERACAO_INEXISTENTE' AS tipo,
    rp.id_operacao AS id_operacao,
    NULL AS id_feira,
    rp.id_produto AS id_produto,
    NULL AS id_categoria,
    'Registro de produção referencia uma operação inexistente.'
        AS descricao

FROM registros_producao AS rp

LEFT JOIN operacoes AS o
    ON o.id_operacao = rp.id_operacao

WHERE o.id_operacao IS NULL


UNION ALL


SELECT
    'PRODUTO_INEXISTENTE' AS tipo,
    rp.id_operacao AS id_operacao,
    NULL AS id_feira,
    rp.id_produto AS id_produto,
    NULL AS id_categoria,
    'Registro de produção referencia um produto inexistente.'
        AS descricao

FROM registros_producao AS rp

LEFT JOIN produtos AS p
    ON p.id_produto = rp.id_produto

WHERE p.id_produto IS NULL


UNION ALL


SELECT
    'CATEGORIA_INEXISTENTE' AS tipo,
    NULL AS id_operacao,
    NULL AS id_feira,
    p.id_produto AS id_produto,
    p.id_categoria AS id_categoria,
    'Produto referencia uma categoria inexistente.'
        AS descricao

FROM produtos AS p

LEFT JOIN categorias AS c
    ON c.id_categoria = p.id_categoria

WHERE c.id_categoria IS NULL;
"""