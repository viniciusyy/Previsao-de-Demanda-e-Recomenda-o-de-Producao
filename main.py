"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

Definição da granularidade da previsão.

Fluxo:

MySQL
    ↓
Carregamento
    ↓
Validação
    ↓
Comparação das granularidades
    ↓
Escolha da granularidade
    ↓
Tabelas e gráficos

Nenhum modelo de previsão é treinado nesta fase.
"""

from analysis.granularity import (
    run_granularity_analysis,
    save_granularity_tables,
)
from analysis.plots import (
    generate_granularity_plots,
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
    Executa a Fase 5 do projeto.
    """

    print("=" * 70)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("E RECOMENDAÇÃO DE PRODUÇÃO")
    print("=" * 70)

    print()
    print(
        "DEFINIÇÃO DA GRANULARIDADE"
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
                "Conexão com MySQL realizada "
                "com sucesso."
            )

        # ====================================================
        # CARREGAMENTO
        # ====================================================

        print()
        print(
            "Carregando dados históricos..."
        )

        dataframe = (
            load_historical_data()
        )

        print(
            "Dados carregados com sucesso."
        )

        print(
            f"Registros carregados: "
            f"{len(dataframe)}"
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
            _validation_details,
            censored_data,
        ) = validate_historical_data(
            dataframe=dataframe,
            referential_issues=(
                referential_issues
            ),
        )

        errors = validation_summary.loc[
            validation_summary[
                "nivel"
            ] == "ERRO",
            "quantidade",
        ].sum()

        warnings = validation_summary.loc[
            validation_summary[
                "nivel"
            ] == "ALERTA",
            "quantidade",
        ].sum()

        if errors > 0:
            raise ValueError(
                f"A base possui "
                f"{int(errors)} erro(s) "
                "de validação. "
                "A análise de granularidade "
                "não será executada."
            )

        print(
            "Nenhum erro crítico encontrado."
        )

        if warnings > 0:
            print(
                f"Atenção: existem "
                f"{int(warnings)} "
                "alerta(s) na base."
            )

        print(
            "Possíveis casos de demanda "
            "censurada: "
            f"{len(censored_data)}"
        )

        # ====================================================
        # ANÁLISE DAS GRANULARIDADES
        # ====================================================

        print()
        print(
            "Analisando granularidades..."
        )

        analyses = (
            run_granularity_analysis(
                dataframe
            )
        )

        summary = analyses[
            "resumo_granularidades"
        ]

        columns = [
            "nome_granularidade",
            "quantidade_series",
            "minimo_observacoes_por_serie",
            "mediana_observacoes_por_serie",
            "media_observacoes_por_serie",
            "maximo_observacoes_por_serie",
            "percentual_series_com_12_ou_mais",
        ]

        print()
        print("=" * 70)
        print(
            "COMPARAÇÃO DAS GRANULARIDADES"
        )
        print("=" * 70)
        print()

        print(
            summary[
                columns
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "A referência de 12 observações "
            "é apenas um indicador exploratório."
        )

        print(
            "Ela não comprova que uma série "
            "seja suficiente para treinar ou "
            "validar qualquer modelo."
        )

        # ====================================================
        # DECISÃO DA GRANULARIDADE
        # ====================================================

        decision = analyses[
            "decisao_granularidade"
        ].iloc[0]

        print()
        print("=" * 70)
        print(
            "GRANULARIDADE ESCOLHIDA"
        )
        print("=" * 70)
        print()

        print(
            "Granularidade: "
            f"{decision['nome_granularidade']}"
        )

        print(
            "Unidade de cada série: "
            f"{decision['unidade_da_serie']}"
        )

        print(
            "Variável-alvo: "
            f"{decision['variavel_alvo']}"
        )

        print(
            "Quantidade de séries: "
            f"{decision['quantidade_series']}"
        )

        print(
            "Total de observações: "
            f"{decision['total_observacoes']}"
        )

        print(
            "Mediana de observações por série: "
            f"{decision[
                'observacoes_mediana_por_serie'
            ]}"
        )

        print(
            "Coeficiente de variação mediano: "
            f"{decision[
                'coeficiente_variacao_mediano'
            ]}"
        )

        print()
        print(
            "Justificativa:"
        )

        print(
            decision[
                "justificativa"
            ]
        )

        print()
        print(
            "Estratégia de modelagem:"
        )

        print(
            decision[
                "estrategia_de_modelagem"
            ]
        )

        print()
        print(
            "Estratégia para os produtos:"
        )

        print(
            decision[
                "estrategia_para_produtos"
            ]
        )

        # ====================================================
        # SALVAR TABELAS
        # ====================================================

        print()
        print(
            "Salvando tabelas..."
        )

        table_files = (
            save_granularity_tables(
                analyses
            )
        )

        print(
            f"{len(table_files)} "
            "tabelas geradas."
        )

        # ====================================================
        # GERAR GRÁFICOS
        # ====================================================

        print()
        print(
            "Gerando gráficos..."
        )

        figure_files = (
            generate_granularity_plots(
                analyses
            )
        )

        print(
            f"{len(figure_files)} "
            "gráficos gerados."
        )

        # ====================================================
        # ARQUIVOS GERADOS
        # ====================================================

        print()
        print("=" * 70)
        print(
            "ARQUIVOS GERADOS"
        )
        print("=" * 70)
        print()

        print(
            "Tabelas:"
        )

        for path in table_files:
            print(
                f"- {path.name}"
            )

        print()
        print(
            "Gráficos:"
        )

        for path in figure_files:
            print(
                f"- {path.name}"
            )

        # ====================================================
        # FINALIZAÇÃO
        # ====================================================

        print()

        print(
            "Granularidade definida: "
            f"{decision['nome_granularidade']}"
        )

        print(
            "A decisão foi registrada em "
            "decisao_granularidade.csv."
        )

        print(
            "Nenhum modelo de previsão "
            "foi treinado nesta fase."
        )

    except Exception as exc:
        print()
        print("=" * 70)
        print(
            "ERRO"
        )
        print("=" * 70)
        print()

        print(
            exc
        )


if __name__ == "__main__":
    main()