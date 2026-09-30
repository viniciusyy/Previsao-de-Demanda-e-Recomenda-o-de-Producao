"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

Distribuição das previsões entre os produtos.

"""

import warnings

from analysis.plots import generate_product_distribution_plots
from database.connection import test_connection
from database.repository import (
    load_historical_data,
    load_referential_integrity_issues,
)
from evaluation.temporal_validation import run_temporal_validation
from forecasting.baselines import run_baseline_evaluation
from forecasting.mlp_model import run_mlp_evaluation
from forecasting.operational_forecast import run_operational_forecast
from forecasting.product_distribution import (
    run_product_distribution,
    save_product_distribution_outputs,
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
    print("DISTRIBUIÇÃO DAS PREVISÕES ENTRE PRODUTOS")
    print()

    try:
        print("Conectando ao banco de dados...")

        if test_connection():
            print("Conexão com MySQL realizada com sucesso.")

        print()
        print("Carregando o snapshot atual do MySQL...")

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
                "A Fase 12 não será executada."
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
        print("Recriando e revalidando toda a etapa de previsão...")

        feature_analyses = run_feature_engineering(dataframe)
        modeling_data = feature_analyses[
            "dataset_modelagem_categoria_feira"
        ]
        category_fair_data = feature_analyses[
            "base_categoria_feira_com_atributos"
        ]

        temporal_analyses = run_temporal_validation(modeling_data)
        fold_details = temporal_analyses["folds_validacao_temporal"]
        baseline_analyses = run_baseline_evaluation(fold_details)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Found unknown categories in columns \\[2\\] during "
                    "transform.*"
                ),
                category=UserWarning,
            )
            phase9_analyses = run_statistical_model_evaluation(
                fold_details,
                baseline_analyses,
            )
            phase10_analyses = run_mlp_evaluation(
                fold_details,
                phase9_analyses,
            )

        print(
            "Categorias de clima ausentes em alguns conjuntos de treino "
            "foram tratadas pelo codificador configurado para categorias "
            "desconhecidas."
        )
        current_decision = phase10_analyses[
            "decisao_modelo_fase10"
        ].iloc[0]

        ranking = phase10_analyses["ranking_modelos_fase10"]
        ranking_columns = [
            "posicao",
            "nome_modelo",
            "mae",
            "rmse",
            "mape",
        ]

        print()
        print("=" * 70)
        print("RANKING ATUALIZADO COM O NOVO SNAPSHOT")
        print("=" * 70)
        print()
        print(ranking[ranking_columns].to_string(index=False))

        print()
        print("=" * 70)
        print("MODELO OPERACIONAL REVALIDADO")
        print("=" * 70)
        print()

        print(
            "Modelo líder revalidado: "
            f"{current_decision['nome_modelo_lider']}."
        )
        print(f"MAE revalidado: {current_decision['mae_lider']:.4f}")
        print(f"RMSE revalidado: {current_decision['rmse_lider']:.4f}")
        print(f"MAPE revalidado: {current_decision['mape_lider']:.4f}%")

        phase11_analyses = run_operational_forecast(
            category_fair_data=category_fair_data,
            phase10_analyses=phase10_analyses,
            source_record_count=len(dataframe),
        )

        category_forecasts = phase11_analyses[
            "previsoes_operacionais_categoria_feira"
        ]
        operational_decision = phase11_analyses[
            "decisao_modelo_operacional"
        ].iloc[0]

        print(
            "Janela histórica aplicada: "
            f"{int(operational_decision['janela_historica'])} ocorrência(s)."
        )
        print(
            "Data de corte atual: "
            f"{category_forecasts['data_corte'].max():%Y-%m-%d}"
        )
        print(
            "Previsões de categoria recriadas: "
            f"{len(category_forecasts)}"
        )

        print()
        print("Calculando proporções e distribuindo entre os produtos...")

        analyses = run_product_distribution(
            source_data=dataframe,
            category_forecasts=category_forecasts,
            source_record_count=len(dataframe),
        )

        validation = analyses["validacao_distribuicao_produtos"]

        print()
        print("=" * 70)
        print("VALIDAÇÃO DA DISTRIBUIÇÃO ENTRE PRODUTOS")
        print("=" * 70)
        print()
        print(validation.to_string(index=False))

        summary = analyses["resumo_distribuicao_produtos"]
        summary_columns = [
            "data_venda_prevista",
            "feira",
            "categoria",
            "previsao_categoria",
            "previsao_distribuida",
            "produtos_catalogo",
            "produtos_permitidos",
            "unidades_adicionadas_maiores_restos",
            "diferenca_reconciliacao",
        ]

        print()
        print("=" * 70)
        print("RESUMO DA DISTRIBUIÇÃO POR CATEGORIA")
        print("=" * 70)
        print()
        print(summary[summary_columns].to_string(index=False))

        restrictions = analyses["restricoes_comerciais_produtos"]
        restriction_columns = [
            "feira",
            "id_produto",
            "produto",
            "categoria",
            "motivo_restricao",
            "previsao_produto",
        ]

        print()
        print("=" * 70)
        print("RESTRIÇÕES COMERCIAIS APLICADAS")
        print("=" * 70)
        print()
        print(restrictions[restriction_columns].to_string(index=False))

        predictions = analyses[
            "previsoes_operacionais_produto_feira"
        ]
        top_predictions = predictions.sort_values(
            ["previsao_produto", "feira", "id_produto"],
            ascending=[False, True, True],
        ).head(15)
        top_columns = [
            "feira",
            "categoria",
            "id_produto",
            "produto",
            "participacao_percentual",
            "previsao_produto",
        ]

        print()
        print("=" * 70)
        print("MAIORES PREVISÕES POR PRODUTO + FEIRA")
        print("=" * 70)
        print()
        top_display = top_predictions[top_columns].copy()
        top_display["participacao_percentual"] = top_display[
            "participacao_percentual"
        ].round(4)
        print(top_display.to_string(index=False))

        decision = analyses["decisao_distribuicao_produtos"].iloc[0]

        print()
        print("=" * 70)
        print("DECISÃO")
        print("=" * 70)
        print()
        print(
            "Registros do MySQL no snapshot: "
            f"{int(decision['registros_origem_mysql'])}"
        )
        print(f"Data de corte: {decision['data_corte']:%Y-%m-%d}")
        print(
            "Produtos no catálogo: "
            f"{int(decision['quantidade_produtos'])}"
        )
        print(
            "Linhas produto + feira: "
            f"{int(decision['linhas_produto_feira'])}"
        )
        print(
            "Combinações permitidas: "
            f"{int(decision['combinacoes_permitidas'])}"
        )
        print(
            "Combinações restritas em zero: "
            f"{int(decision['combinacoes_restritas'])}"
        )
        print(
            "Total previsto nas categorias: "
            f"{int(decision['previsao_total_categorias'])}"
        )
        print(
            "Total distribuído entre produtos: "
            f"{int(decision['previsao_total_produtos'])}"
        )
        print("Recomendação final de produção gerada: não")

        print()
        print("Salvando previsões e relatórios...")

        table_files = save_product_distribution_outputs(analyses)

        print(f"{len(table_files)} arquivos CSV gerados.")

        print()
        print("Gerando gráficos...")

        figure_files = generate_product_distribution_plots(analyses)

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
        print("O snapshot atualizado do MySQL foi utilizado.")
        print("As 18 previsões de categoria foram revalidadas.")
        print("Os totais foram distribuídos entre os 27 produtos.")
        print("As quatro combinações proibidas foram fixadas em zero.")
        print("A soma dos produtos permaneceu igual à das categorias.")
        print("Nenhuma otimização por Simplex foi executada nesta fase.")

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERRO")
        print("=" * 70)
        print()
        print(f"Falha na execução: {exc}")


if __name__ == "__main__":
    main()







