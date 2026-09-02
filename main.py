"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

    Análise Exploratória dos Dados.

Fluxo:

MySQL
    ↓
Carregamento
    ↓
Validação
    ↓
Análise exploratória
    ↓
Tabelas
    ↓
Gráficos

Nenhum modelo de previsão é treinado nesta fase.
"""

from analysis.exploratory import (
    run_exploratory_analysis,
    save_exploratory_tables,
)
from analysis.plots import (
    generate_all_plots,
)
from database.connection import (
    test_connection,
)
from database.repository import (
    load_historical_data,
    load_referential_integrity_issues,
)
from preprocessing.validation import (
    validate_historical_data,
)


def main() -> None:
    """
    Executa a Fase 4 do projeto.
    """

    print("=" * 70)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("E RECOMENDAÇÃO DE PRODUÇÃO")
    print("=" * 70)

    print()
    print(
        "Análise Exploratória dos Dados"
    )
    print()

    try:
        # ====================================================
        # CONEXÃO
        # ====================================================

        print(
            "Conectando ao banco de dados..."
        )

        if test_connection():
            print(
                "Conexão com MySQL realizada com sucesso."
            )

        # ====================================================
        # CARREGAMENTO
        # ====================================================

        print()
        print(
            "Carregando dados históricos..."
        )

        dataframe = load_historical_data()

        print(
            "Dados carregados com sucesso."
        )

        print(
            f"Registros carregados: {len(dataframe)}"
        )

        # ====================================================
        # VALIDAÇÃO
        # ====================================================

        print()
        print(
            "Verificando qualidade dos dados..."
        )

        referential_issues = (
            load_referential_integrity_issues()
        )

        (
            validation_summary,
            validation_details,
            censored_data,
        ) = validate_historical_data(
            dataframe=dataframe,
            referential_issues=referential_issues,
        )

        errors = validation_summary.loc[
            validation_summary["nivel"] == "ERRO",
            "quantidade",
        ].sum()

        warnings = validation_summary.loc[
            validation_summary["nivel"] == "ALERTA",
            "quantidade",
        ].sum()

        if errors > 0:
            raise ValueError(
                f"A base possui {int(errors)} erro(s) "
                "de validação. "
                "A análise exploratória não será executada."
            )

        print(
            "Nenhum erro crítico encontrado."
        )

        if warnings > 0:
            print(
                f"Atenção: existem {int(warnings)} "
                "alerta(s) na base."
            )

        print(
            f"Possíveis casos de demanda censurada: "
            f"{len(censored_data)}"
        )

        # ====================================================
        # ANÁLISE EXPLORATÓRIA
        # ====================================================

        print()
        print(
            "Executando análise exploratória..."
        )

        analyses = (
            run_exploratory_analysis(
                dataframe
            )
        )

        print(
            "Análise exploratória concluída."
        )

        # ====================================================
        # RESUMO GERAL
        # ====================================================

        general_summary = analyses[
            "resumo_geral"
        ].iloc[0]

        print()
        print("=" * 70)
        print("RESUMO GERAL DA BASE")
        print("=" * 70)
        print()

        print(
            f"Quantidade de registros: "
            f"{general_summary['quantidade_registros']}"
        )

        print(
            f"Quantidade de operações: "
            f"{general_summary['quantidade_operacoes']}"
        )

        print(
            f"Quantidade de feiras: "
            f"{general_summary['quantidade_feiras']}"
        )

        print(
            f"Quantidade de produtos: "
            f"{general_summary['quantidade_produtos']}"
        )

        print(
            f"Quantidade de categorias: "
            f"{general_summary['quantidade_categorias']}"
        )

        print()

        print(
            f"Primeira data: "
            f"{general_summary['primeira_data_venda']}"
        )

        print(
            f"Última data: "
            f"{general_summary['ultima_data_venda']}"
        )

        print(
            f"Período calendário: "
            f"{general_summary['periodo_calendario_dias']} dias"
        )

        # ====================================================
        # PRODUÇÃO E VENDAS
        # ====================================================

        print()
        print("=" * 70)
        print("PRODUÇÃO E VENDAS")
        print("=" * 70)
        print()

        print(
            f"Total produzido: "
            f"{general_summary['total_produzido']}"
        )

        print(
            f"Total vendido: "
            f"{general_summary['total_vendido']}"
        )

        print(
            f"Total de sobra: "
            f"{general_summary['total_sobra']}"
        )

        print(
            f"Taxa geral de sobra: "
            f"{general_summary['taxa_sobra_percentual']:.2f}%"
        )

        # ====================================================
        # DEMANDA CENSURADA
        # ====================================================

        print()
        print("=" * 70)
        print("POSSÍVEL DEMANDA CENSURADA")
        print("=" * 70)
        print()

        print(
            "Registros identificados: "
            f"{general_summary['possivel_demanda_censurada']}"
        )

        print(
            "Percentual sobre os registros: "
            f"{general_summary[
                'percentual_possivel_demanda_censurada'
            ]:.2f}%"
        )

        print()
        print(
            "Esse indicador é calculado por produto dentro "
            "de cada operação."
        )

        print(
            "Ele não significa que a feira inteira ficou "
            "sem produtos."
        )

        print(
            "Também não permite calcular a quantidade "
            "de demanda perdida."
        )

        # ====================================================
        # FEIRAS
        # ====================================================

        print()
        print("=" * 70)
        print("ANÁLISE POR FEIRA")
        print("=" * 70)
        print()

        fair_columns = [
            "feira",
            "quantidade_operacoes",
            "total_produzido",
            "total_vendido",
            "total_sobra",
            "taxa_sobra_percentual",
            "media_vendida_por_operacao",
            "media_sobra_por_operacao",
        ]

        print(
            analyses[
                "analise_por_feira"
            ][fair_columns].to_string(
                index=False
            )
        )

        # ====================================================
        # CATEGORIAS
        # ====================================================

        print()
        print("=" * 70)
        print("ANÁLISE POR CATEGORIA")
        print("=" * 70)
        print()

        category_columns = [
            "categoria",
            "total_produzido",
            "total_vendido",
            "total_sobra",
            "taxa_sobra_percentual",
        ]

        print(
            analyses[
                "analise_por_categoria"
            ][category_columns].to_string(
                index=False
            )
        )

        # ====================================================
        # PRODUTOS
        # ====================================================

        print()
        print("=" * 70)
        print("10 PRODUTOS MAIS VENDIDOS")
        print("=" * 70)
        print()

        product_columns = [
            "id_produto",
            "produto",
            "categoria",
            "total_vendido",
            "total_sobra",
        ]

        print(
            analyses[
                "analise_por_produto"
            ][product_columns]
            .head(10)
            .to_string(
                index=False
            )
        )

        # ====================================================
        # CLIMA
        # ====================================================

        print()
        print("=" * 70)
        print("ANÁLISE DESCRITIVA POR CLIMA")
        print("=" * 70)
        print()

        climate_columns = [
            "clima",
            "quantidade_operacoes",
            "total_vendido",
            "media_vendida_por_operacao",
            "mediana_vendida_por_operacao",
            "media_sobra_por_operacao",
        ]

        print(
            analyses[
                "analise_por_clima"
            ][climate_columns].to_string(
                index=False
            )
        )

        print()
        print(
            "A comparação principal utiliza média por operação, "
            "pois cada clima possui quantidade diferente "
            "de observações."
        )

        print(
            "Diferenças observadas entre os climas "
            "não demonstram causalidade."
        )

        # ====================================================
        # FERIADOS
        # ====================================================

        print()
        print("=" * 70)
        print("FERIADOS × DIAS NORMAIS")
        print("=" * 70)
        print()

        holiday_columns = [
            "tipo_dia",
            "quantidade_operacoes",
            "total_vendido",
            "media_vendida_por_operacao",
            "mediana_vendida_por_operacao",
            "media_sobra_por_operacao",
        ]

        print(
            analyses[
                "analise_feriados"
            ][holiday_columns].to_string(
                index=False
            )
        )

        print()
        print(
            "A comparação considera o total vendido "
            "em cada operação."
        )

        print(
            "Essa análise também é descritiva "
            "e não demonstra causalidade."
        )

        # ====================================================
        # SALVAR TABELAS
        # ====================================================

        print()
        print(
            "Salvando tabelas..."
        )

        table_files = (
            save_exploratory_tables(
                analyses
            )
        )

        print(
            f"{len(table_files)} tabelas geradas."
        )

        # ====================================================
        # GRÁFICOS
        # ====================================================

        print()
        print(
            "Gerando gráficos..."
        )

        figure_files = (
            generate_all_plots(
                dataframe=dataframe,
                analyses=analyses,
            )
        )

        print(
            f"{len(figure_files)} gráficos gerados."
        )

        # ====================================================
        # ARQUIVOS GERADOS
        # ====================================================

        print()
        print("=" * 70)
        print("ARQUIVOS GERADOS")
        print("=" * 70)
        print()

        print("Tabelas:")

        for path in table_files:
            print(
                f"- {path.name}"
            )

        print()
        print("Gráficos:")

        for path in figure_files:
            print(
                f"- {path.name}"
            )



        print()
        print(
            "A análise desta fase é descritiva."
        )


    except Exception as exc:
        print()
        print("=" * 70)
        print("ERRO")
        print("=" * 70)
        print()

        print(exc)


if __name__ == "__main__":
    main()