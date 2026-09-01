"""
Sistema Inteligente de Previsão de Demanda e
Recomendação de Produção para uma Pastelaria.

Trabalho de Conclusão de Curso - Ciência da Computação.

"""

from database.connection import test_connection
from database.repository import load_historical_data


def main() -> None:
    """

    Nesta fase, o sistema:

    1. testa a conexão com o MySQL;
    2. consulta os dados históricos;
    3. carrega os dados em um DataFrame;
    4. apresenta informações básicas sobre o carregamento.
    """

    print("=" * 60)
    print("SISTEMA DE PREVISÃO DE DEMANDA")
    print("E RECOMENDAÇÃO DE PRODUÇÃO")
    print("=" * 60)

    print()
    print("Conexão com MySQL")
    print()

    try:
        # ====================================================
        # TESTE DA CONEXÃO
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

        # ====================================================
        # INFORMAÇÕES BÁSICAS
        # ====================================================

        print()
        print("=" * 60)
        print("INFORMAÇÕES DA BASE")
        print("=" * 60)

        print()
        print(f"Quantidade de registros: {len(df)}")

        if df.empty:
            print()
            print("A consulta não retornou registros.")

        else:
            quantidade_operacoes = df["id_operacao"].nunique()

            print(
                f"Quantidade de operações: "
                f"{quantidade_operacoes}"
            )

            print(
                f"Primeira data de venda: "
                f"{df['data_venda'].min()}"
            )

            print(
                f"Última data de venda: "
                f"{df['data_venda'].max()}"
            )

            print(
                f"Quantidade de feiras: "
                f"{df['feira'].nunique()}"
            )

            print(
                f"Quantidade de produtos encontrados: "
                f"{df['id_produto'].nunique()}"
            )

            print()
            print("=" * 60)
            print("COLUNAS CARREGADAS")
            print("=" * 60)
            print()

            for coluna in df.columns:
                print(f"- {coluna}")

            print()
            print("=" * 60)
            print("PRIMEIROS REGISTROS")
            print("=" * 60)
            print()

            print(df.head())


    except Exception as exc:
        print()
        print("=" * 60)
        print("ERRO")
        print("=" * 60)
        print()
        print(exc)
        print()
        print(
            "Verifique as configurações do arquivo .env "
            "e se o servidor MySQL está em execução."
        )


if __name__ == "__main__":
    main()