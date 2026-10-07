"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

Consolidação operacional e documentação final.

"""

import warnings

from analysis.plots import generate_production_recommendation_plots
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
)
from forecasting.statistical_models import run_statistical_model_evaluation
from optimization.production_recommendation import (
    run_production_recommendation,
    save_production_recommendation_outputs,
)
from preprocessing.features import run_feature_engineering
from preprocessing.validation import validate_historical_data
from reporting.final_consolidation import (
    run_final_consolidation,
    save_final_consolidation_outputs,
)


def main() -> None:
    

    print("=" * 70)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("E RECOMENDAÇÃO DE PRODUÇÃO")
    print("=" * 70)
    print()
    print("CONSOLIDAÇÃO OPERACIONAL E DOCUMENTAÇÃO FINAL")
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
                "A Fase 14 não será executada."
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
            "Estratégia operacional aplicada: "
            f"{operational_decision['estrategia_operacional']}."
        )
        print(
            "Janela/histórico operacional: "
            f"{operational_decision['janela_historica']}."
        )
        print(
            "Linhas usadas no treinamento final: "
            f"{int(operational_decision['linhas_treino_modelo_final'])}."
        )
        selected_configuration = str(
            operational_decision[
                "configuracao_operacional_selecionada"
            ]
        ).strip()
        if selected_configuration:
            print(
                "Configuração operacional selecionada: "
                f"{selected_configuration}."
            )
        print(
            "Data de corte atual: "
            f"{category_forecasts['data_corte'].max():%Y-%m-%d}"
        )
        print(
            "Previsões de categoria recriadas: "
            f"{len(category_forecasts)}"
        )

        future_context = phase11_analyses["contexto_futuro_previsao"]
        if not future_context.empty:
            print()
            print("Contexto futuro informado:")
            print(
                future_context[
                    [
                        "feira",
                        "data_venda_prevista",
                        "clima",
                        "eh_feriado",
                        "nome_feriado",
                    ]
                ].to_string(index=False)
            )

        operational_validation = phase11_analyses[
            "validacao_previsao_operacional"
        ]
        print(
            "Validações da previsão operacional aprovadas: "
            f"{int(operational_validation['resultado'].eq('OK').sum())} "
            f"de {len(operational_validation)}."
        )

        print()
        print("Recriando a distribuição das previsões entre produtos...")

        phase12_analyses = run_product_distribution(
            source_data=dataframe,
            category_forecasts=category_forecasts,
            source_record_count=len(dataframe),
        )

        predictions = phase12_analyses[
            "previsoes_operacionais_produto_feira"
        ]
        print(f"Previsões por produto recriadas: {len(predictions)}")

        print()
        print("Estimando capacidades com o histórico do MySQL...")
        print("Executando o modelo linear pelo dual Simplex...")

        analyses = run_production_recommendation(
            source_data=dataframe,
            product_forecasts=predictions,
            source_record_count=len(dataframe),
        )

        capacities = analyses["capacidades_estimadas_producao"]
        capacity_columns = [
            "dia_producao",
            "feiras_atendidas",
            "quantidade_datas_historicas",
            "producao_media_historica",
            "capacidade_estimada",
            "data_producao_maxima",
        ]

        print()
        print("=" * 70)
        print("CAPACIDADES ESTIMADAS PELO HISTÓRICO")
        print("=" * 70)
        print()
        capacity_display = capacities[capacity_columns].copy()
        capacity_display["producao_media_historica"] = capacity_display[
            "producao_media_historica"
        ].round(2)
        print(capacity_display.to_string(index=False))
        print()
        print(
            "Esses valores representam máximos observados no histórico, "
            "não capacidades físicas definitivas."
        )

        validation = analyses["validacao_recomendacao_producao"]

        print()
        print("=" * 70)
        print("VALIDAÇÃO DA RECOMENDAÇÃO DE PRODUÇÃO")
        print("=" * 70)
        print()
        print(validation.to_string(index=False))

        day_summary = analyses["resumo_recomendacao_por_dia"]
        day_columns = [
            "data_producao_prevista",
            "dia_producao",
            "feiras_atendidas",
            "previsao_total",
            "capacidade_estimada",
            "recomendacao_total",
            "reducao_total",
            "percentual_atendimento",
            "restricao_capacidade_ativa",
        ]

        print()
        print("=" * 70)
        print("RECOMENDAÇÃO POR DIA DE PRODUÇÃO")
        print("=" * 70)
        print()
        day_display = day_summary[day_columns].copy()
        day_display["percentual_atendimento"] = day_display[
            "percentual_atendimento"
        ].round(4)
        print(day_display.to_string(index=False))

        fair_summary = analyses["resumo_recomendacao_por_feira"]
        fair_columns = [
            "data_producao_prevista",
            "data_venda_prevista",
            "feira",
            "previsao_total",
            "recomendacao_total",
            "reducao_total",
            "percentual_atendimento",
        ]

        print()
        print("=" * 70)
        print("RECOMENDAÇÃO POR FEIRA")
        print("=" * 70)
        print()
        fair_display = fair_summary[fair_columns].copy()
        fair_display["percentual_atendimento"] = fair_display[
            "percentual_atendimento"
        ].round(4)
        print(fair_display.to_string(index=False))

        recommendations = analyses[
            "recomendacoes_producao_produto_feira"
        ]
        reductions = recommendations.loc[
            recommendations["reducao_em_relacao_previsao"] > 0
        ].sort_values(
            ["reducao_em_relacao_previsao", "feira", "id_produto"],
            ascending=[False, True, True],
        ).head(15)
        reduction_columns = [
            "feira",
            "categoria",
            "id_produto",
            "produto",
            "previsao_produto",
            "recomendacao_producao",
            "reducao_em_relacao_previsao",
        ]

        print()
        print("=" * 70)
        print("MAIORES AJUSTES EM RELAÇÃO À PREVISÃO")
        print("=" * 70)
        print()
        if reductions.empty:
            print("Nenhuma redução foi necessária; a capacidade foi suficiente.")
        else:
            print(reductions[reduction_columns].to_string(index=False))

        decision = analyses["decisao_recomendacao_producao"].iloc[0]

        print()
        print("=" * 70)
        print("DECISÃO DA OTIMIZAÇÃO")
        print("=" * 70)
        print()
        print(
            "Registros do MySQL no snapshot: "
            f"{int(decision['registros_origem_mysql'])}"
        )
        print(f"Data de corte: {decision['data_corte']:%Y-%m-%d}")
        print(f"Fonte da capacidade: {decision['fonte_capacidade']}")
        print(f"Solver: {decision['algoritmo_solver']}")
        print(
            "Grupos com restrição de capacidade ativa: "
            f"{int(decision['grupos_com_capacidade_ativa'])}"
        )
        print(
            "Total previsto: "
            f"{int(decision['previsao_total_produtos'])}"
        )
        print(
            "Total recomendado: "
            f"{int(decision['recomendacao_total_producao'])}"
        )
        print(
            "Redução total por capacidade: "
            f"{int(decision['reducao_total_por_capacidade'])}"
        )
        print("Recomendação final de produção gerada: sim")

        print()
        print("Salvando previsões e relatórios...")

        table_files = save_production_recommendation_outputs(analyses)

        print(f"{len(table_files)} arquivos CSV gerados.")

        print()
        print("Gerando gráficos...")

        figure_files = generate_production_recommendation_plots(analyses)

        print(f"{len(figure_files)} gráfico(s) gerado(s).")

        print()
        print("Consolidando o pacote operacional final...")

        final_artifacts = run_final_consolidation(
            ranking=ranking,
            operational_analyses=phase11_analyses,
            distribution_analyses=phase12_analyses,
            recommendation_analyses=analyses,
            censored_count=len(censored_data),
        )
        final_files = save_final_consolidation_outputs(final_artifacts)
        final_summary = final_artifacts["resumo_execucao_final"].iloc[0]
        final_validation = final_artifacts[
            "validacao_consolidacao_final"
        ]

        print()
        print("=" * 70)
        print("CONSOLIDAÇÃO OPERACIONAL FINAL")
        print("=" * 70)
        print()
        print(
            "Modelo operacional: "
            f"{final_summary['nome_modelo_operacional']}"
        )
        print(
            "Registros do snapshot: "
            f"{int(final_summary['registros_origem_mysql'])}"
        )
        print(
            "Itens com produção recomendada: "
            f"{int(final_summary['itens_com_producao_recomendada'])}"
        )
        print(
            "Previsão total: "
            f"{int(final_summary['previsao_total'])}"
        )
        print(
            "Recomendação total: "
            f"{int(final_summary['recomendacao_total'])}"
        )
        print(
            "Redução total: "
            f"{int(final_summary['reducao_total'])}"
        )
        print(
            "Validações da consolidação aprovadas: "
            f"{int(final_validation['resultado'].eq('OK').sum())} "
            f"de {len(final_validation)}."
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
        print("Pacote operacional final:")
        for path in final_files:
            print(f"- {path.name}")

        print()
        print("=" * 70)
        print("EXECUTADA COM SUCESSO")
        print("=" * 70)
        print()
        print("O snapshot atualizado do MySQL foi utilizado.")
        print("As previsões e a distribuição por produto foram revalidadas.")
        print("As capacidades foram estimadas pelo máximo histórico diário.")
        print("O modelo linear foi resolvido pelo dual Simplex do HiGHS.")
        print("A recomendação não ultrapassa a previsão nem a capacidade.")
        print("O mix previsto foi preservado proporcionalmente.")
        print("O plano operacional detalhado foi consolidado.")
        print("As premissas e limitações foram registradas no relatório final.")
        print("A configuração da capacidade poderá ser substituída no futuro.")

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERRO")
        print("=" * 70)
        print()
        print(f"Falha na execução: {exc}")


if __name__ == "__main__":
    main()




