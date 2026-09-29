"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

Rede neural MLP.

"""

from analysis.plots import generate_mlp_plots
from database.connection import test_connection
from database.repository import (
    load_historical_data,
    load_referential_integrity_issues,
)
from evaluation.temporal_validation import run_temporal_validation
from forecasting.baselines import run_baseline_evaluation
from forecasting.mlp_model import run_mlp_evaluation, save_mlp_outputs
from forecasting.statistical_models import run_statistical_model_evaluation
from preprocessing.features import run_feature_engineering
from preprocessing.validation import validate_historical_data


def main() -> None:
    """Executa a Fase 10 do projeto."""

    print("=" * 70)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("E RECOMENDAÇÃO DE PRODUÇÃO")
    print("=" * 70)
    print()
    print("REDE NEURAL MLP")
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
                "A Fase 10 não será executada."
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
        print("Recriando os seis modelos anteriores...")

        baseline_analyses = run_baseline_evaluation(fold_details)
        phase9_analyses = run_statistical_model_evaluation(
            fold_details,
            baseline_analyses,
        )

        print("Modelos anteriores recriados com sucesso.")

        print()
        print("Treinando e avaliando a MLP...")

        analyses = run_mlp_evaluation(
            fold_details,
            phase9_analyses,
        )

        validation = analyses["validacao_mlp"]

        print()
        print("=" * 70)
        print("VALIDAÇÃO DA MLP")
        print("=" * 70)
        print()
        print(validation.to_string(index=False))

        selection = analyses["selecao_configuracao_mlp_por_fold"]
        selected = selection.loc[selection["selecionada"]].copy()
        selection_columns = [
            "fold",
            "configuracao",
            "camadas_ocultas",
            "alpha",
            "mae_validacao_interna",
            "rmse_validacao_interna",
            "iteracoes",
            "alerta_convergencia",
        ]

        print()
        print("=" * 70)
        print("CONFIGURAÇÃO SELECIONADA POR FOLD")
        print("=" * 70)
        print()
        print(selected[selection_columns].to_string(index=False))

        mlp_metrics = analyses["metricas_mlp_geral"]
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
        print("MÉTRICAS DA MLP")
        print("=" * 70)
        print()
        print(mlp_metrics[metric_columns].to_string(index=False))

        ranking = analyses["ranking_modelos_fase10"]
        ranking_columns = [
            "posicao",
            "nome_modelo",
            "mae",
            "rmse",
            "mape",
        ]

        print()
        print("=" * 70)
        print("RANKING GERAL APÓS A MLP")
        print("=" * 70)
        print()
        print(ranking[ranking_columns].to_string(index=False))

        decision = analyses["decisao_modelo_fase10"].iloc[0]

        print()
        print("=" * 70)
        print("DECISÃO")
        print("=" * 70)
        print()
        print(f"Modelo líder: {decision['nome_modelo_lider']}")
        print(f"MAE do líder: {decision['mae_lider']:.4f}")
        print(f"RMSE do líder: {decision['rmse_lider']:.4f}")
        print(f"MAPE do líder: {decision['mape_lider']:.4f}%")
        print(f"MAE da MLP: {decision['mae_mlp']:.4f}")
        print(
            "MLP superou o baseline em MAE: "
            f"{'sim' if decision['mlp_superou_baseline_em_mae'] else 'não'}"
        )
        print(
            "Variação percentual do MAE da MLP em relação ao baseline: "
            f"{decision['melhoria_percentual_mae_mlp_sobre_baseline']:.4f}%"
        )

        print()
        print("Salvando previsões e relatórios...")

        table_files = save_mlp_outputs(analyses)

        print(f"{len(table_files)} arquivos CSV gerados.")

        print()
        print("Gerando gráficos...")

        figure_files = generate_mlp_plots(analyses)

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
        print("A MLP foi avaliada nos mesmos 90 casos de teste.")
        print("A seleção da rede utilizou somente dados de treino.")
        print("Todos os sete modelos foram comparados pelo mesmo protocolo.")
        print("Nenhuma recomendação de produção foi gerada nesta fase.")

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERRO")
        print("=" * 70)
        print()
        print(f"Falha na execução da Fase 10: {exc}")


if __name__ == "__main__":
    main()
