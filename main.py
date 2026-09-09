"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

Validação temporal.

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
    generate_temporal_validation_plots,
)
from database.connection import test_connection
from database.repository import (
    load_historical_data,
    load_referential_integrity_issues,
)
from evaluation.temporal_validation import (
    run_temporal_validation,
    save_temporal_validation_outputs,
)
from preprocessing.features import (
    run_feature_engineering,
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
        print("Conectando ao banco de dados...")

        if test_connection():
            print(
                "Conexão com MySQL realizada com sucesso."
            )

        print()
        print("Carregando dados históricos...")

        dataframe = load_historical_data()

        print("Dados carregados com sucesso.")
        print(
            f"Registros carregados: {len(dataframe)}"
        )

        print()
        print("Verificando qualidade dos dados...")

        referential_issues = (
            load_referential_integrity_issues()
        )

        (
            validation_summary,
            _validation_details,
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
                f"A base possui {int(errors)} "
                "erro(s) de validação. "
                "A validação temporal não será executada."
            )

        print("Nenhum erro crítico encontrado.")

        if warnings > 0:
            print(
                "Atenção: existem "
                f"{int(warnings)} alerta(s) na base."
            )

        print(
            "Possíveis casos de demanda censurada: "
            f"{len(censored_data)}"
        )

        print()
        print(
            "Recriando o dataset de modelagem..."
        )

        feature_analyses = (
            run_feature_engineering(dataframe)
        )

        modeling_data = feature_analyses[
            "dataset_modelagem_categoria_feira"
        ]

        print(
            "Linhas disponíveis para a "
            "validação temporal: "
            f"{len(modeling_data)}"
        )

        print(
            "Séries preservadas: "
            f"{modeling_data['serie'].nunique()}"
        )

        print()
        print(
            "Criando folds temporais walk-forward..."
        )

        analyses = run_temporal_validation(
            modeling_data
        )

        fold_summary = analyses[
            "resumo_folds_temporais"
        ]

        display_columns = [
            "fold",
            "observacoes_treino_por_serie",
            "posicao_teste_por_serie",
            "linhas_treino",
            "linhas_teste",
            "series_teste",
        ]

        print()
        print("=" * 70)
        print("RESUMO DOS FOLDS TEMPORAIS")
        print("=" * 70)
        print()

        print(
            fold_summary[
                display_columns
            ].to_string(index=False)
        )

        temporal_validation = analyses[
            "validacao_folds_temporais"
        ]

        print()
        print("=" * 70)
        print("VALIDAÇÃO DOS FOLDS")
        print("=" * 70)
        print()

        print(
            temporal_validation.to_string(
                index=False
            )
        )

        print()
        print(
            "Salvando configurações e folds..."
        )

        table_files = (
            save_temporal_validation_outputs(
                analyses
            )
        )

        print(
            f"{len(table_files)} arquivos CSV gerados."
        )

        print()
        print("Gerando gráfico...")

        figure_files = (
            generate_temporal_validation_plots(
                analyses
            )
        )

        print(
            f"{len(figure_files)} gráfico(s) gerado(s)."
        )

        print()
        print("=" * 70)
        print("ARQUIVOS GERADOS")
        print("=" * 70)
        print()

        print("CSVs:")

        for path in table_files:
            print(f"- {path.name}")

        print()
        print("Gráficos:")

        for path in figure_files:
            print(f"- {path.name}")

        print()
        print("=" * 70)
        print("EXECUTADA COM SUCESSO")
        print("=" * 70)
        print()

        print(
            "Estratégia: walk-forward "
            "com janela expansiva."
        )
        print("Quantidade de folds: 5.")
        print(
            "Horizonte de teste: uma ocorrência "
            "por série e fold."
        )
        print("Embaralhamento dos dados: não.")

        total_test_predictions = int(
            fold_summary["linhas_teste"].sum()
        )

        print(
            "Total de previsões de teste: "
            f"{total_test_predictions}."
        )

        print(
            "O pré-processamento dos modelos deverá "
            "ser ajustado somente no treino de cada fold."
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