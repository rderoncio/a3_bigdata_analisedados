from enum import Enum
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd

class Utils(Enum):
    """
    Classe utilitária que contém os metodos auxiliares para o EDA do dataset de voos.
    """
    num_voo = {'tipo': str, 'descricao': 'Número do voo'}
    companhia_aerea = {'tipo': str, 'descricao': 'Nome da companhia aérea'}
    codigo_tipo_linha = {'tipo': str, 'descricao': 'Código do tipo de linha (internacional)'}
    aeroporto_origem = {'tipo': str, 'descricao': 'Código do aeroporto de origem'}
    cidade_origem = {'tipo': str, 'descricao': 'Cidade de origem'}
    estado_origem = {'tipo': str, 'descricao': 'Estado de origem'}
    pais_origem = {'tipo': str, 'descricao': 'País de origem'}
    lat_orig = {'tipo': float, 'descricao': 'Latitude do aeroporto de origem'}
    long_orig = {'tipo': float, 'descricao': 'Longitude do aeroporto de origem'}
    partida_prevista = {'tipo': datetime, 'descricao': 'Horário previsto de partida'}
    partida_real = {'tipo': datetime, 'descricao': 'Horário real de partida'}
    aeroporto_destino = {'tipo': str, 'descricao': 'Código do aeroporto de destino'}
    cidade_destino = {'tipo': str, 'descricao': 'Cidade de destino'}
    estado_destino = {'tipo': str, 'descricao': 'Estado de destino'}
    pais_destino = {'tipo': str, 'descricao': 'País de destino'}
    lat_dest = {'tipo': float, 'descricao': 'Latitude do aeroporto de destino'}
    long_dest = {'tipo': float, 'descricao': 'Longitude do aeroporto de destino'}
    chegada_prevista = {'tipo': datetime, 'descricao': 'Horário previsto de chegada'}
    chegada_real = {'tipo': datetime, 'descricao': 'Horário real de chegada'}
    distancia_km = {'tipo': float, 'descricao': 'Total em quilômetros entre o local de origem e destino'}
    situacao_voo = {'tipo': str, 'descricao': 'Situação do voo (realizado, cancelado, etc)'}
    codigo_justificativa = {'tipo': str, 'descricao': 'Código de justificativa (caso o voo tenha sido cancelado)'}

    @staticmethod
    def nomes_colunas() -> List[str]:
        """
        Retorna uma lista com os nomes de todas as colunas definidas como enums na classe Utils.
        
        Returns:
            Uma lista com os nomes de todas as colunas definidas como enums na classe Utils.
        """
        return list(map(lambda coluna: coluna.name, Utils))

    @staticmethod
    def tipo_coluna(nome_coluna: str) -> str | None:
        """
        Retorna o tipo de dados da coluna especificada.

        Args:
        nome_coluna (str): O nome da coluna.

        Returns:
        O tipo de dados da coluna especificada.
        """
        return Utils.__dict__[nome_coluna].__dict__['_value_']['tipo']
    
    @staticmethod
    def descricao_coluna(nome_coluna: str) -> str | None:
        """
        Retorna a descrição da coluna especificada.

        Args:
        nome_coluna (str): O nome da coluna.

        Returns:
        A descrição da coluna especificada.
        """
        return Utils.__dict__[nome_coluna].__dict__['_value_']['descricao']

    @staticmethod
    def calcular_distancia_km(lat_orig:float, long_orig:float, lat_dest:float, long_dest:float) -> float:
        """
        Calcula a distância em km entre dois pontos (latitude, longitude) utilizando a fórmula Haversine.

        Parameters
        ----------
        lat_orig : float
            Latitude do ponto de origem em graus.
        long_orig : float
            Longitude do ponto de origem em graus.
        lat_dest : float
            Latitude do ponto de destino em graus.
        long_dest : float
            Longitude do ponto de destino em graus.

        Returns
        -------
        float
            Distância em km entre os dois pontos.

        """
        # raio da Terra em km
        R = 6371  

        # Converte para radianos
        lat_orig, long_orig, lat_dest, long_dest = map(np.radians, [lat_orig, long_orig, lat_dest, long_dest])

        # Calcula a diferença entre as latitudes e longitudes
        dlat = lat_dest - lat_orig
        dlong = long_dest - long_orig

        # Calcula a fórmula Haversine
        a = np.sin(dlat/2)**2 + np.cos(lat_orig) * np.cos(lat_dest) * np.sin(dlong/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Retorna a distância em km
        return R * c
    
    @staticmethod
    def formatar_colunas_datetime(dataframe: pd.DataFrame, nome_coluna: str) -> pd.DataFrame:
        """
        Formata uma coluna de um DataFrame contendo datas/horas como uma string no formato '%d/%m/%Y %H:%M:%S'.

        Args:
            dataframe: O DataFrame contendo a coluna a ser formatada.
            nome_coluna: O nome da coluna a ser formatada.

        Returns:
            O DataFrame com a coluna formatada.
        """
        dataframe[nome_coluna] = pd.to_datetime(dataframe[nome_coluna])
        dataframe[nome_coluna] = dataframe[nome_coluna].dt.strftime('%d/%m/%Y %H:%M:%S')
        return dataframe
