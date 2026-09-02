"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

"""

from database.connection import test_connection
from database.repository import (
    load_historical_data,
    load_referential_integrity_issues,
)
from preprocessing.validation import (
    save_validation_reports,
    validate_historical_data,
)


def main() -> None:
    """
    Executa a Fase 3 do projeto.
    """

    print("=" * 70)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("E RECOMENDAÇÃO DE PRODUÇÃO")
    print("=" * 70)

    print()
    print("Validação dos Dados")
    print()

    try:
        # ====================================================
        # CONEXÃO
        # ====================================================

        print("Conectando ao banco de dados...")

        if test_connection():
            print("Conexão com MySQL realizada com sucesso.")

        # ====================================================
        # CARREGAMENTO DOS DADOS
        # ====================================================

        print()
        print("Carregando dados históricos...")

        df = load_historical_data()

        print("Dados carregados com sucesso.")

        print()
        print(
            f"Quantidade de registros carregados: {len(df)}"
        )

        print(
            f"Quantidade de operações: "
            f"{df['id_operacao'].nunique()}"
        )

        # ====================================================
        # INTEGRIDADE REFERENCIAL
        # ====================================================

        print()
        print("Verificando integridade do banco...")

        referential_issues = (
            load_referential_integrity_issues()
        )

        if referential_issues.empty:
            print(
                "Nenhum problema de integridade "
                "referencial encontrado."
            )
        else:
            print(
                f"Foram encontrados "
                f"{len(referential_issues)} problemas "
                f"de integridade referencial."
            )

        # ====================================================
        # VALIDAÇÃO
        # ====================================================

        print()
        print("Executando validações dos dados...")

        summary, details = validate_historical_data(
            dataframe=df,
            referential_issues=referential_issues,
        )

        print("Validação concluída.")

        # ====================================================
        # RESUMO
        # ====================================================

        print()
        print("=" * 70)
        print("RESUMO DA VALIDAÇÃO")
        print("=" * 70)
        print()

        print(
            summary[
                [
                    "validacao",
                    "nivel",
                    "quantidade",
                    "status",
                ]
            ].to_string(index=False)
        )

        # ====================================================
        # CONTAGEM POR NÍVEL
        # ====================================================

        errors = summary.loc[
            summary["nivel"] == "ERRO",
            "quantidade",
        ].sum()

        warnings = summary.loc[
            summary["nivel"] == "ALERTA",
            "quantidade",
        ].sum()

        informations = summary.loc[
            summary["nivel"] == "INFO",
            "quantidade",
        ].sum()

        print()
        print("=" * 70)
        print("RESULTADO GERAL")
        print("=" * 70)
        print()

        print(f"Erros encontrados: {int(errors)}")
        print(f"Alertas encontrados: {int(warnings)}")
        print(
            f"Informações registradas: "
            f"{int(informations)}"
        )

        # ====================================================
        # RELATÓRIO
        # ====================================================

        summary_path, details_path = (
            save_validation_reports(
                summary,
                details,
            )
        )

        print()
        print("=" * 70)
        print("RELATÓRIOS GERADOS")
        print("=" * 70)
        print()

        print(
            f"Resumo: {summary_path}"
        )

        print(
            f"Detalhes: {details_path}"
        )

        # ====================================================
        # INTERPRETAÇÃO
        # ====================================================

        print()
        print("=" * 70)
        print("INTERPRETAÇÃO")
        print("=" * 70)
        print()

        if errors > 0:
            print(
                "A base possui inconsistências classificadas "
                "como ERRO."
            )

            print(
                "Esses registros deverão ser analisados antes "
                "de serem utilizados na modelagem."
            )

        else:
            print(
                "Nenhuma inconsistência crítica foi encontrada."
            )

        if warnings > 0:
            print(
                "Existem ALERTAS que devem ser revisados, "
                "mas não necessariamente representam erro."
            )

        if informations > 0:
            print(
                "Existem registros informativos, como possíveis "
                "casos de demanda censurada."
            )

        print()
        print(
            "Nenhuma informação foi alterada automaticamente."
        )

        print()
        print("=" * 70)
        print("EXECUTADA")
        print("=" * 70)

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERRO ")
        print("=" * 70)
        print()
        print(exc)


if __name__ == "__main__":
    main()