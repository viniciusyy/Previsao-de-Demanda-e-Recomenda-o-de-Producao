"""
Consultas SQL utilizadas pelo sistema.

As consultas ao banco de dados devem permanecer centralizadas
neste módulo para evitar SQL espalhado por diferentes partes
do projeto.
"""


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