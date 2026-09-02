"""
Camada responsável por consultar o banco de dados e transformar
os resultados em estruturas utilizadas pela aplicação.

Nesta etapa, o módulo fornece:

- carregamento dos dados históricos;
- carregamento de possíveis problemas de integridade referencial.
"""

from typing import Any

import pandas as pd

from database.connection import get_connection
from database.queries import (
    QUERY_HISTORICAL_DATA,
    QUERY_REFERENTIAL_INTEGRITY,
)


def _execute_query(query: str) -> pd.DataFrame:
    """
    Executa uma consulta SQL e retorna o resultado em DataFrame.

    Args:
        query: Consulta SQL que será executada.

    Returns:
        pd.DataFrame: Resultado da consulta.

    Raises:
        RuntimeError: Caso ocorra algum erro durante a execução.
    """

    connection = None
    cursor = None

    try:
        connection = get_connection()

        cursor = connection.cursor(dictionary=True)

        cursor.execute(query)

        rows: list[dict[str, Any]] = cursor.fetchall()

        if not rows:
            return pd.DataFrame(columns=cursor.column_names)

        return pd.DataFrame(rows)

    except Exception as exc:
        raise RuntimeError(
            f"Erro ao executar consulta no banco: {exc}"
        ) from exc

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()


def load_historical_data() -> pd.DataFrame:
    """
    Carrega os registros históricos da pastelaria.

    Returns:
        pd.DataFrame: Dados históricos utilizados pelo projeto.
    """

    return _execute_query(QUERY_HISTORICAL_DATA)


def load_referential_integrity_issues() -> pd.DataFrame:
    """
    Procura referências inválidas entre as tabelas do banco.

    Exemplos:

    - operação apontando para feira inexistente;
    - registro apontando para operação inexistente;
    - registro apontando para produto inexistente;
    - produto apontando para categoria inexistente.

    Returns:
        pd.DataFrame: Problemas de integridade encontrados.
    """

    return _execute_query(QUERY_REFERENTIAL_INTEGRITY)