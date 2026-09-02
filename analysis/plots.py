"""
Geração dos gráficos da análise exploratória.

Todos os gráficos são produzidos utilizando matplotlib
e armazenados em reports/figures.

Os gráficos desta etapa possuem finalidade descritiva.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.exploratory import (
    prepare_analysis_data,
)


def _get_figures_directory() -> Path:
    """
    Retorna a pasta onde os gráficos serão armazenados.
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    figures_directory = (
        project_root
        / "reports"
        / "figures"
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return figures_directory


def _save_figure(
    filename: str,
) -> Path:
    """
    Salva a figura atual e fecha o gráfico.
    """

    path = (
        _get_figures_directory()
        / filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


def plot_demand_over_time(
    analyses: dict[str, pd.DataFrame],
) -> Path:
    """
    Gera a demanda total do negócio ao longo do tempo.

    Importante:

    Em datas com duas feiras simultâneas, como sábado
    e domingo, as vendas das duas operações são somadas.
    """

    data = analyses[
        "analise_por_data"
    ].copy()

    data["data_venda"] = pd.to_datetime(
        data["data_venda"]
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        data["data_venda"],
        data["total_vendido"],
        marker="o",
    )

    plt.title(
        "Demanda total observada ao longo do tempo"
    )

    plt.xlabel(
        "Data da venda"
    )

    plt.ylabel(
        "Quantidade vendida"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid(
        alpha=0.3
    )

    return _save_figure(
        "01_demanda_total_ao_longo_tempo.png"
    )


def plot_demand_over_time_by_fair(
    analyses: dict[str, pd.DataFrame],
) -> Path:
    """
    Gera séries temporais separadas por feira.

    Esse gráfico evita misturar duas feiras realizadas
    simultaneamente no mesmo dia.
    """

    data = analyses[
        "analise_por_data_feira"
    ].copy()

    data["data_venda"] = pd.to_datetime(
        data["data_venda"]
    )

    plt.figure(
        figsize=(13, 7)
    )

    fairs = sorted(
        data["feira"].dropna().unique()
    )

    for fair in fairs:
        fair_data = data[
            data["feira"] == fair
        ].sort_values(
            "data_venda"
        )

        plt.plot(
            fair_data["data_venda"],
            fair_data["total_vendido"],
            marker="o",
            label=fair,
        )

    plt.title(
        "Demanda observada ao longo do tempo por feira"
    )

    plt.xlabel(
        "Data da venda"
    )

    plt.ylabel(
        "Quantidade vendida"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend(
        title="Feira"
    )

    return _save_figure(
        "02_demanda_ao_longo_tempo_por_feira.png"
    )


def plot_sales_by_fair(
    analyses: dict[str, pd.DataFrame],
) -> Path:
    """
    Gera gráfico do total vendido por feira.
    """

    data = analyses[
        "analise_por_feira"
    ].copy()

    data = data.sort_values(
        "total_vendido",
        ascending=False,
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        data["feira"],
        data["total_vendido"],
    )

    plt.title(
        "Quantidade total vendida por feira"
    )

    plt.xlabel(
        "Feira"
    )

    plt.ylabel(
        "Quantidade vendida"
    )

    return _save_figure(
        "03_vendas_por_feira.png"
    )


def plot_sales_by_category(
    analyses: dict[str, pd.DataFrame],
) -> Path:
    """
    Gera gráfico de vendas por categoria.
    """

    data = analyses[
        "analise_por_categoria"
    ].copy()

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        data["categoria"],
        data["total_vendido"],
    )

    plt.title(
        "Quantidade total vendida por categoria"
    )

    plt.xlabel(
        "Categoria"
    )

    plt.ylabel(
        "Quantidade vendida"
    )

    return _save_figure(
        "04_vendas_por_categoria.png"
    )


def plot_sales_by_product(
    analyses: dict[str, pd.DataFrame],
) -> Path:
    """
    Gera gráfico das vendas totais por produto.
    """

    data = analyses[
        "analise_por_produto"
    ].copy()

    data = data.sort_values(
        "total_vendido",
        ascending=True,
    )

    plt.figure(
        figsize=(11, 11)
    )

    plt.barh(
        data["produto"],
        data["total_vendido"],
    )

    plt.title(
        "Quantidade total vendida por produto"
    )

    plt.xlabel(
        "Quantidade vendida"
    )

    plt.ylabel(
        "Produto"
    )

    return _save_figure(
        "05_vendas_por_produto.png"
    )


def plot_production_sales_leftover_by_fair(
    analyses: dict[str, pd.DataFrame],
) -> Path:
    """
    Compara produção, venda e sobra acumuladas por feira.
    """

    data = analyses[
        "analise_por_feira"
    ].copy()

    x = list(
        range(
            len(data)
        )
    )

    width = 0.25

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        [
            value - width
            for value in x
        ],
        data["total_produzido"],
        width=width,
        label="Produzido",
    )

    plt.bar(
        x,
        data["total_vendido"],
        width=width,
        label="Vendido",
    )

    plt.bar(
        [
            value + width
            for value in x
        ],
        data["total_sobra"],
        width=width,
        label="Sobra",
    )

    plt.xticks(
        x,
        data["feira"],
    )

    plt.title(
        "Produção, vendas e sobras por feira"
    )

    plt.xlabel(
        "Feira"
    )

    plt.ylabel(
        "Quantidade"
    )

    plt.legend()

    return _save_figure(
        "06_producao_venda_sobra_por_feira.png"
    )


def plot_leftover_rate_by_fair(
    analyses: dict[str, pd.DataFrame],
) -> Path:
    """
    Gera gráfico da taxa acumulada de sobra por feira.
    """

    data = analyses[
        "analise_por_feira"
    ].copy()

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        data["feira"],
        data["taxa_sobra_percentual"],
    )

    plt.title(
        "Taxa de sobra por feira"
    )

    plt.xlabel(
        "Feira"
    )

    plt.ylabel(
        "Taxa de sobra (%)"
    )

    return _save_figure(
        "07_taxa_sobra_por_feira.png"
    )


def plot_average_sales_by_climate(
    analyses: dict[str, pd.DataFrame],
) -> Path:
    """
    Compara a média vendida por operação em cada clima.

    Essa comparação é mais adequada que utilizar somente
    o volume total acumulado, pois cada clima pode possuir
    número diferente de operações registradas.
    """

    data = analyses[
        "analise_por_clima"
    ].copy()

    data = data.sort_values(
        "media_vendida_por_operacao",
        ascending=False,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        data["clima"],
        data["media_vendida_por_operacao"],
    )

    plt.title(
        "Média vendida por operação em cada clima"
    )

    plt.xlabel(
        "Clima"
    )

    plt.ylabel(
        "Quantidade média vendida por operação"
    )

    plt.xticks(
        rotation=20
    )

    return _save_figure(
        "08_media_vendida_por_operacao_por_clima.png"
    )


def plot_holiday_comparison(
    analyses: dict[str, pd.DataFrame],
) -> Path:
    """
    Compara a quantidade total média vendida por operação
    em feriados e dias normais.
    """

    data = analyses[
        "analise_feriados"
    ].copy()

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        data["tipo_dia"],
        data["media_vendida_por_operacao"],
    )

    plt.title(
        "Média vendida por operação: feriado × dia normal"
    )

    plt.xlabel(
        "Tipo de dia"
    )

    plt.ylabel(
        "Quantidade média vendida por operação"
    )

    return _save_figure(
        "09_feriado_vs_dia_normal.png"
    )


def plot_production_vs_sales(
    dataframe: pd.DataFrame,
) -> Path:
    """
    Compara produção total e venda total por operação.
    """

    df = prepare_analysis_data(
        dataframe
    )

    operations = (
        df.groupby(
            [
                "id_operacao",
                "data_venda",
                "feira",
            ]
        )
        .agg(
            total_produzido=(
                "quantidade_produzida",
                "sum",
            ),

            total_vendido=(
                "quantidade_vendida",
                "sum",
            ),
        )
        .reset_index()
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        operations[
            "total_produzido"
        ],
        operations[
            "total_vendido"
        ],
        alpha=0.7,
    )

    plt.title(
        "Produção × venda por operação"
    )

    plt.xlabel(
        "Quantidade produzida"
    )

    plt.ylabel(
        "Quantidade vendida"
    )

    plt.grid(
        alpha=0.3
    )

    return _save_figure(
        "10_producao_vs_venda.png"
    )


def plot_sales_vs_leftover(
    dataframe: pd.DataFrame,
) -> Path:
    """
    Compara venda total e sobra total por operação.
    """

    df = prepare_analysis_data(
        dataframe
    )

    operations = (
        df.groupby(
            [
                "id_operacao",
                "data_venda",
                "feira",
            ]
        )
        .agg(
            total_vendido=(
                "quantidade_vendida",
                "sum",
            ),

            total_sobra=(
                "quantidade_sobra",
                "sum",
            ),
        )
        .reset_index()
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        operations[
            "total_vendido"
        ],
        operations[
            "total_sobra"
        ],
        alpha=0.7,
    )

    plt.title(
        "Venda × sobra por operação"
    )

    plt.xlabel(
        "Quantidade vendida"
    )

    plt.ylabel(
        "Quantidade de sobra"
    )

    plt.grid(
        alpha=0.3
    )

    return _save_figure(
        "11_venda_vs_sobra.png"
    )


def generate_all_plots(
    dataframe: pd.DataFrame,
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:
    """
    Gera todos os gráficos da análise exploratória.
    """

    generated_files = [
        plot_demand_over_time(
            analyses
        ),

        plot_demand_over_time_by_fair(
            analyses
        ),

        plot_sales_by_fair(
            analyses
        ),

        plot_sales_by_category(
            analyses
        ),

        plot_sales_by_product(
            analyses
        ),

        plot_production_sales_leftover_by_fair(
            analyses
        ),

        plot_leftover_rate_by_fair(
            analyses
        ),

        plot_average_sales_by_climate(
            analyses
        ),

        plot_holiday_comparison(
            analyses
        ),

        plot_production_vs_sales(
            dataframe
        ),

        plot_sales_vs_leftover(
            dataframe
        ),
    ]

    return generated_files