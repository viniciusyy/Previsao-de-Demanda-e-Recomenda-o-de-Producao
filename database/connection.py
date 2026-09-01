"""
Módulo responsável pela conexão com o banco de dados MySQL.

As configurações de acesso são carregadas a partir do arquivo .env,
evitando que usuário e senha sejam escritos diretamente no código.

"""

import os
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error
from mysql.connector.connection import MySQLConnection


# ============================================================
# CARREGAMENTO DO ARQUIVO .env
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


def get_database_config() -> dict:
    """
    Carrega as configurações do banco de dados a partir do arquivo .env.

    Returns:
        dict: Configurações necessárias para conexão com o MySQL.

    Raises:
        ValueError: Caso alguma configuração obrigatória não tenha sido
        definida no arquivo .env.
    """

    config = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

    campos_faltando = [
        chave
        for chave, valor in config.items()
        if valor is None or str(valor).strip() == ""
    ]

    if campos_faltando:
        raise ValueError(
            "Configurações ausentes no arquivo .env: "
            + ", ".join(campos_faltando)
        )

    try:
        config["port"] = int(config["port"])
    except ValueError as exc:
        raise ValueError(
            "DB_PORT deve possuir um número inteiro válido."
        ) from exc

    return config


def get_connection() -> MySQLConnection:
    """
    Cria e retorna uma conexão com o banco MySQL.

    Returns:
        MySQLConnection: Conexão ativa com o banco de dados.

    Raises:
        ConnectionError: Caso não seja possível realizar a conexão.
    """

    config = get_database_config()

    try:
        connection = mysql.connector.connect(**config)

        if not connection.is_connected():
            raise ConnectionError(
                "A conexão com o MySQL não foi estabelecida."
            )

        return connection

    except Error as exc:
        raise ConnectionError(
            f"Não foi possível conectar ao banco MySQL: {exc}"
        ) from exc


def test_connection() -> bool:
    """
    Testa se a aplicação consegue se conectar ao banco.

    Returns:
        bool: True caso a conexão seja realizada com sucesso.

    Raises:
        ConnectionError: Caso a conexão falhe.
    """

    connection = None

    try:
        connection = get_connection()

        return connection.is_connected()

    finally:
        if connection is not None and connection.is_connected():
            connection.close()