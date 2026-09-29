"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

Previsão operacional por categoria + feira.

"""

from analysis.plots import generate_operational_forecast_plots
from database.connection import test_connection
from database.repository import (
    load_historical_data,
    load_referential_integrity_issues,
)
from evaluation.temporal_validation import run_temporal_validation
from forecasting.baselines import run_baseline_evaluation
from forecasting.mlp_model import run_mlp_evaluation
from forecasting.operational_forecast import (
    run_operational_forecast,
    save_operational_forecast_outputs,
)
from forecasting.statistical_models import run_statistical_model_evaluation
from preprocessing.features import run_feature_engineering
from preprocessing.validation import validate_historical_data


def main() -> None:
    

    print("=" * 70)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("E RECOMENDAÇÃO DE PRODUÇÃO")
    print("=" * 70)
    print()
    print("PREVISÃO OPERACIONAL")
    print()

    try:
        print("Conectando ao banco de dados...")

        if test_connection():
            print("Conexão com MySQL realizada com sucesso.")

        print()
        print("Carregando dados históricos...")

        dataframe = load_historical_data()

        print("Dados carregados com sucesso.")
        print(f"Registros carregados: {len(dataframe)}")

        print()
        print("Verificando qualidade dos dados...")

        referential_issues = load_referential_integrity_issues()
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
        warnings_count = validation_summary.loc[
            validation_summary["nivel"] == "ALERTA",
            "quantidade",
        ].sum()

        if errors > 0:
            raise ValueError(
                f"A base possui {int(errors)} erro(s) de validação. "
                "A Fase não será executada."
            )

        print("Nenhum erro crítico encontrado.")

        if warnings_count > 0:
            print(
                f"Atenção: existem {int(warnings_count)} alerta(s) na base."
            )

        print(
            "Possíveis casos de demanda censurada: "
            f"{len(censored_data)}"
        )

        print()
        print("Recriando o dataset de modelagem...")

        feature_analyses = run_feature_engineering(dataframe)
        modeling_data = feature_analyses[
            "dataset_modelagem_categoria_feira"
        ]
        category_fair_data = feature_analyses[
            "base_categoria_feira_com_atributos"
        ]

        print(
            "Linhas disponíveis para modelagem: "
            f"{len(modeling_data)}"
        )
        print(f"Séries preservadas: {modeling_data['serie'].nunique()}")

        print()
        print("Revalidando a seleção do modelo com a base atual...")

        temporal_analyses = run_temporal_validation(modeling_data)
        fold_details = temporal_analyses["folds_validacao_temporal"]
        baseline_analyses = run_baseline_evaluation(fold_details)
        phase9_analyses = run_statistical_model_evaluation(
            fold_details,
            baseline_analyses,
        )
        phase10_analyses = run_mlp_evaluation(
            fold_details,
            phase9_analyses,
        )

        current_decision = phase10_analyses[
            "decisao_modelo_fase10"
        ].iloc[0]

        print(
            "Modelo líder revalidado: "
            f"{current_decision['nome_modelo_lider']}."
        )
        print(f"MAE revalidado: {current_decision['mae_lider']:.4f}")

        print()
        print("Gerando previsões para as próximas feiras...")

        analyses = run_operational_forecast(
            category_fair_data=category_fair_data,
            phase10_analyses=phase10_analyses,
            source_record_count=len(dataframe),
        )

        validation = analyses["validacao_previsao_operacional"]

        print()
        print("=" * 70)
        print("VALIDAÇÃO DA PREVISÃO OPERACIONAL")
        print("=" * 70)
        print()
        print(validation.to_string(index=False))

        forecasts = analyses[
            "previsoes_operacionais_categoria_feira"
        ]
        forecast_columns = [
            "data_producao_prevista",
            "data_venda_prevista",
            "feira",
            "categoria",
            "demanda_historico_3",
            "demanda_historico_2",
            "demanda_historico_1",
            "previsao_bruta",
            "previsao_operacional",
        ]

        print()
        print("=" * 70)
        print("PREVISÕES POR CATEGORIA + FEIRA")
        print("=" * 70)
        print()

        forecast_display = forecasts[forecast_columns].copy()

        forecast_display["previsao_bruta"] = (
            forecast_display["previsao_bruta"].round(4)
        )

        print(forecast_display.to_string(index=False))

        summary = analyses["resumo_previsao_operacional"]

        print()
        print("=" * 70)
        print("RESUMO POR FEIRA")
        print("=" * 70)
        print()
        print(summary.to_string(index=False))

        decision = analyses["decisao_modelo_operacional"].iloc[0]

        print()
        print("=" * 70)
        print("DECISÃO OPERACIONAL")
        print("=" * 70)
        print()
        print(f"Modelo aplicado: {decision['nome_modelo_operacional']}")
        print(f"Data de corte: {decision['data_corte']:%Y-%m-%d}")
        print(
            "Séries previstas: "
            f"{int(decision['quantidade_series_previstas'])}"
        )
        print(
            "Registros do MySQL no snapshot: "
            f"{int(decision['registros_origem_mysql'])}"
        )
        print("Recomendação de produção gerada: não")

        print()
        print("Salvando previsões e relatórios...")

        table_files = save_operational_forecast_outputs(analyses)

        print(f"{len(table_files)} arquivos CSV gerados.")

        print()
        print("Gerando gráficos...")

        figure_files = generate_operational_forecast_plots(analyses)

        print(f"{len(figure_files)} gráfico(s) gerado(s).")

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
        print("O modelo foi revalidado com o snapshot atual do MySQL.")
        print("Foram geradas 18 previsões em categoria + feira.")
        print("As previsões utilizam somente as três ocorrências anteriores.")
        print("Nenhuma distribuição entre produtos foi realizada.")
        print("Nenhuma recomendação de produção foi gerada nesta fase.")

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERRO")
        print("=" * 70)
        print()
        print(f"Falha na execução: {exc}")


if __name__ == "__main__":
    main()

