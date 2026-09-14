"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

Modelos de referência e avaliação

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
    generate_baseline_plots,
)
from database.connection import test_connection
from database.repository import (
    load_historical_data,
    load_referential_integrity_issues,
)
from evaluation.temporal_validation import (
    run_temporal_validation,
)
from forecasting.baselines import (
    run_baseline_evaluation,
    save_baseline_outputs,
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
                "Os baselines não serão executados."
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
            "Linhas disponíveis para modelagem: "
            f"{len(modeling_data)}"
        )

        print(
            "Séries preservadas: "
            f"{modeling_data['serie'].nunique()}"
        )

        print()
        print(
            "Recriando os folds temporais..."
        )

        temporal_analyses = (
            run_temporal_validation(
                modeling_data
            )
        )

        fold_details = temporal_analyses[
            "folds_validacao_temporal"
        ]

        print(
            "Casos de teste disponíveis: "
            f"{int((fold_details['conjunto'] == 'teste').sum())}"
        )

        print()
        print(
            "Executando modelos de referência..."
        )

        analyses = run_baseline_evaluation(
            fold_details
        )

        validation = analyses[
            "validacao_baselines"
        ]

        print()
        print("=" * 70)
        print("VALIDAÇÃO DOS BASELINES")
        print("=" * 70)
        print()

        print(
            validation.to_string(index=False)
        )

        metrics = analyses[
            "metricas_baselines_geral"
        ]

        metric_columns = [
            "nome_modelo",
            "quantidade_previsoes",
            "quantidade_alvos_zero",
            "mae",
            "mse",
            "rmse",
            "mape",
        ]

        print()
        print("=" * 70)
        print("MÉTRICAS GERAIS DOS BASELINES")
        print("=" * 70)
        print()

        print(
            metrics[
                metric_columns
            ].to_string(index=False)
        )

        ranking = analyses[
            "ranking_baselines"
        ]

        ranking_columns = [
            "posicao",
            "nome_modelo",
            "mae",
            "rmse",
            "mape",
        ]

        print()
        print("=" * 70)
        print("RANKING DOS BASELINES")
        print("=" * 70)
        print()

        print(
            ranking[
                ranking_columns
            ].to_string(index=False)
        )

        decision = analyses[
            "decisao_baseline_referencia"
        ].iloc[0]

        print()
        print("=" * 70)
        print("BASELINE DE REFERÊNCIA")
        print("=" * 70)
        print()

        print(
            "Modelo: "
            f"{decision['nome_modelo']}"
        )

        print(
            f"MAE: {decision['mae']:.4f}"
        )

        print(
            f"MSE: {decision['mse']:.4f}"
        )

        print(
            f"RMSE: {decision['rmse']:.4f}"
        )

        print(
            f"MAPE: {decision['mape']:.4f}%"
        )

        print()
        print(
            "Salvando previsões e relatórios..."
        )

        table_files = save_baseline_outputs(
            analyses
        )

        print(
            f"{len(table_files)} arquivos CSV gerados."
        )

        print()
        print("Gerando gráficos...")

        figure_files = generate_baseline_plots(
            analyses
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
        print("EXECUTADO COM SUCESSO")
        print("=" * 70)
        print()

        print(
            "Todos os baselines foram avaliados "
            "nos mesmos 90 casos de teste."
        )

        print(
            "Critério principal do ranking: MAE."
        )

        print(
            "Critérios de desempate: RMSE e MAPE."
        )

        print(
            "Baseline de referência: "
            f"{decision['nome_modelo']}."
        )

        print(
            "O baseline vencedor ainda não representa "
            "o modelo final do sistema."
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