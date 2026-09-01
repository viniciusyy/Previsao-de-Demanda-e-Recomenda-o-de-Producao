"""
Camada responsável por consultar o banco de dados e transformar
os resultados em estruturas utilizadas pela aplicação.

Nesta fase, o principal objetivo é carregar os dados históricos
da pastelaria em um pandas.DataFrame.
"""

from typing import Any

import pandas as pd

from database.connection import get_connection
from database.queries import QUERY_HISTORICAL_DATA


def load_historical_data() -> pd.DataFrame:
    """
    Consulta os registros históricos da pastelaria no MySQL.

    Os dados são convertidos para um pandas.DataFrame para serem
    utilizados posteriormente nas etapas de análise, tratamento
    e modelagem.

    Returns:
        pd.DataFrame: DataFrame contendo os registros históricos.

    Raises:
        RuntimeError: Caso ocorra algum erro durante a consulta.
    """

    connection = None
    cursor = None

    try:
        connection = get_connection()

        cursor = connection.cursor(dictionary=True)

        cursor.execute(QUERY_HISTORICAL_DATA)

        rows: list[dict[str, Any]] = cursor.fetchall()

        if not rows:
            return pd.DataFrame(columns=cursor.column_names)

        dataframe = pd.DataFrame(rows)

        return dataframe

    except Exception as exc:
        raise RuntimeError(
            f"Erro ao carregar os dados históricos: {exc}"
        ) from exc

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()