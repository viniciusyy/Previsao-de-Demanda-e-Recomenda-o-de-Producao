"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

Suavização exponencial e regressão linear.

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
from analysis.plots import generate_statistical_model_plots
from database.connection import test_connection
from database.repository import (
    load_historical_data,
    load_referential_integrity_issues,
)
from evaluation.temporal_validation import run_temporal_validation
from forecasting.baselines import run_baseline_evaluation
from forecasting.statistical_models import (
    run_statistical_model_evaluation,
    save_statistical_model_outputs,
)
from preprocessing.features import run_feature_engineering
from preprocessing.validation import validate_historical_data


def main() -> None:
    

    print("=" * 70)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("E RECOMENDAÇÃO DE PRODUÇÃO")
    print("=" * 70)
    print()
    print("SUAVIZAÇÃO EXPONENCIAL E REGRESSÃO LINEAR")
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

        warnings = validation_summary.loc[
            validation_summary["nivel"] == "ALERTA",
            "quantidade",
        ].sum()

        if errors > 0:
            raise ValueError(
                f"A base possui {int(errors)} erro(s) de validação. "
                "A Fase 9 não será executada."
            )

        print("Nenhum erro crítico encontrado.")

        if warnings > 0:
            print(f"Atenção: existem {int(warnings)} alerta(s) na base.")

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

        print(
            "Linhas disponíveis para modelagem: "
            f"{len(modeling_data)}"
        )
        print(f"Séries preservadas: {modeling_data['serie'].nunique()}")

        print()
        print("Recriando os folds temporais...")

        temporal_analyses = run_temporal_validation(modeling_data)
        fold_details = temporal_analyses["folds_validacao_temporal"]
        test_cases = fold_details.loc[
            fold_details["conjunto"] == "teste"
        ]

        print(f"Casos de teste disponíveis: {len(test_cases)}")

        print()
        print("Recriando os modelos de referência...")

        baseline_analyses = run_baseline_evaluation(fold_details)
        baseline_decision = baseline_analyses[
            "decisao_baseline_referencia"
        ].iloc[0]

        print(
            "Baseline de referência recalculado: "
            f"{baseline_decision['nome_modelo']}."
        )

        print()
        print("Executando os modelos da Fase 9...")

        analyses = run_statistical_model_evaluation(
            fold_details,
            baseline_analyses,
        )

        validation = analyses["validacao_modelos_fase9"]

        print()
        print("=" * 70)
        print("VALIDAÇÃO DOS MODELOS DA FASE 9")
        print("=" * 70)
        print()
        print(validation.to_string(index=False))

        new_metrics = analyses["metricas_modelos_fase9_geral"]
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
        print("MÉTRICAS DOS NOVOS MODELOS")
        print("=" * 70)
        print()
        print(new_metrics[metric_columns].to_string(index=False))

        ranking = analyses["ranking_modelos_fase9"]
        ranking_columns = [
            "posicao",
            "nome_modelo",
            "mae",
            "rmse",
            "mape",
        ]

        print()
        print("=" * 70)
        print("RANKING GERAL APÓS A FASE 9")
        print("=" * 70)
        print()
        print(ranking[ranking_columns].to_string(index=False))

        decision = analyses["decisao_modelos_fase9"].iloc[0]

        print()
        print("=" * 70)
        print("DECISÃO PROVISÓRIA DA FASE 9")
        print("=" * 70)
        print()
        print(f"Modelo líder: {decision['nome_modelo']}")
        print(f"MAE: {decision['mae']:.4f}")
        print(f"MSE: {decision['mse']:.4f}")
        print(f"RMSE: {decision['rmse']:.4f}")
        print(f"MAPE: {decision['mape']:.4f}%")
        print(
            "Superou o baseline em MAE: "
            f"{'sim' if decision['superou_baseline_em_mae'] else 'não'}"
        )
        print(
            "Variação percentual do MAE em relação ao baseline: "
            f"{decision['melhoria_percentual_mae_sobre_baseline']:.4f}%"
        )

        print()
        print("Salvando previsões e relatórios...")

        table_files = save_statistical_model_outputs(analyses)

        print(f"{len(table_files)} arquivos CSV gerados.")

        print()
        print("Gerando gráficos...")

        figure_files = generate_statistical_model_plots(analyses)

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
        print("Modelos novos: suavização exponencial e regressão linear.")
        print("Todos os modelos foram avaliados nos mesmos 90 casos.")
        print("O pré-processamento foi ajustado somente no treino.")
        print("A decisão ainda é provisória: a MLP não foi avaliada.")

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERRO")
        print("=" * 70)
        print()
        print(f"Falha na execução: {exc}")


if __name__ == "__main__":
    main()
