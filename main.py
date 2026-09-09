"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

Engenharia de atributos.

Fluxo:

MySQL
    ↓
Carregamento e validação
    ↓
Agregação por categoria + feira
    ↓
Atributos temporais
    ↓
Lags e médias móveis
    ↓
Validação contra vazamento temporal
    ↓
Dataset de modelagem

"""

from analysis.plots import (
    generate_feature_engineering_plots,
)
from database.connection import (
    test_connection,
)
from database.repository import (
    load_historical_data,
    load_referential_integrity_issues,
)
from preprocessing.features import (
    run_feature_engineering,
    save_feature_engineering_outputs,
)
from preprocessing.validation import (
    validate_historical_data,
)


def main() -> None:
 

    print("=" * 70)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("E RECOMENDAÇÃO DE PRODUÇÃO")
    print("=" * 70)


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
        # VALIDAÇÃO DA BASE
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
                "A engenharia de atributos "
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
        # ENGENHARIA DE ATRIBUTOS
        # ====================================================

        print()
        print(
            "Criando atributos para "
            "categoria + feira..."
        )

        analyses = (
            run_feature_engineering(
                dataframe
            )
        )

        summary = analyses[
            "resumo_engenharia_atributos"
        ].iloc[0]

        # ====================================================
        # RESUMO
        # ====================================================

        print()
        print("=" * 70)
        print(
            "RESUMO DA ENGENHARIA DE ATRIBUTOS"
        )
        print("=" * 70)
        print()

        print(
            "Registros originais do MySQL: "
            f"{summary[
                'registros_origem_mysql'
            ]}"
        )

        print(
            "Linhas agregadas em categoria + feira: "
            f"{summary[
                'linhas_categoria_feira'
            ]}"
        )

        print(
            "Quantidade de séries: "
            f"{summary[
                'quantidade_series'
            ]}"
        )

        print(
            "Janela histórica máxima: "
            f"{summary[
                'janela_historica_maxima'
            ]}"
        )

        print(
            "Linhas removidas pela criação "
            "dos atributos: "
            f"{summary[
                'linhas_removidas_por_historico'
            ]}"
        )

        print(
            "Linhas disponíveis para modelagem: "
            f"{summary[
                'linhas_dataset_modelagem'
            ]}"
        )

        print(
            "Percentual de dados mantidos: "
            f"{summary[
                'percentual_dados_mantidos'
            ]:.2f}%"
        )

        print(
            "Observações de modelagem por série: "
            f"{summary[
                'observacoes_minimas_modelagem_por_serie'
            ]}"
            " a "
            f"{summary[
                'observacoes_maximas_modelagem_por_serie'
            ]}"
        )

        print(
            "Itens possivelmente censurados: "
            f"{summary[
                'itens_possivelmente_censurados'
            ]}"
        )

        print(
            "Percentual de itens possivelmente "
            "censurados: "
            f"{summary[
                'percentual_itens_possivelmente_censurados'
            ]:.2f}%"
        )

        print(
            "Linhas categoria + feira com algum "
            "item censurado: "
            f"{summary[
                'linhas_com_algum_item_possivelmente_censurado'
            ]}"
        )

        # ====================================================
        # VALIDAÇÃO DOS ATRIBUTOS
        # ====================================================

        feature_validation = analyses[
            "validacao_engenharia_atributos"
        ]

        print()
        print("=" * 70)
        print(
            "VALIDAÇÃO DOS ATRIBUTOS"
        )
        print("=" * 70)
        print()

        print(
            feature_validation.to_string(
                index=False
            )
        )

        # ====================================================
        # SALVAR RESULTADOS
        # ====================================================

        print()
        print(
            "Salvando datasets e relatórios..."
        )

        table_files = (
            save_feature_engineering_outputs(
                analyses
            )
        )

        print(
            f"{len(table_files)} "
            "arquivos CSV gerados."
        )

        # ====================================================
        # GRÁFICO
        # ====================================================

        print()
        print(
            "Gerando gráfico..."
        )

        figure_files = (
            generate_feature_engineering_plots(
                analyses
            )
        )

        print(
            f"{len(figure_files)} "
            "gráfico(s) gerado(s)."
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
            "CSVs:"
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
        print("=" * 70)
        print(
            "EXECUTADA COM SUCESSO"
        )
        print("=" * 70)
        print()

        print(
            "Granularidade utilizada: "
            "Categoria + feira"
        )

        print(
            "Lags criados: "
            "1, 2 e 3 ocorrências anteriores."
        )

        print(
            "Médias móveis criadas: "
            "2, 3 e 4 ocorrências anteriores."
        )

        print(
            "Nenhum atributo histórico utiliza "
            "a demanda da própria linha."
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