import pandas as pd
from enum import Enum
from datetime import datetime
import numpy as np

class DescricaoColuna(Enum):
    """
    Classe de Enums para armazenar as descrições das colunas do DataFrame.
    """
    voos = 'Número do voo'
    companhia_aerea = 'Nome da companhia aérea'
    codigo_tipo_linha = 'Código do tipo de linha (internacional)'
    partida_prevista = 'Horário previsto de partida'
    partida_real = 'Horário real de partida'
    chegada_prevista = 'Horário previsto de chegada'
    chegada_real = 'Horário real de chegada'
    situacao_voo = 'Situação do voo (realizado, cancelado, etc)'
    codigo_justificativa = 'Código de justificativa (caso o voo tenha sido cancelado)'
    aeroporto_origem = 'Código do aeroporto de origem'
    cidade_origem = 'Cidade de origem'
    estado_origem = 'Estado de origem'
    pais_origem = 'País de origem'
    aeroporto_destino = 'Código do aeroporto de destino'
    cidade_destino = 'Cidade de destino'
    estado_destino = 'Estado de destino'
    pais_destino = 'País de destino'
    lat_orig = 'Latitude do aeroporto de origem'
    long_orig = 'Longitude do aeroporto de origem'
    lat_dest = 'Latitude do aeroporto de destino'
    long_dest = 'Longitude do aeroporto de destino'
    distancia_km = 'Total em Kilômetros entre o local Origem e Destino'

    @staticmethod
    def get(nome_coluna):
        """
        Método estático para recuperar a descrição da coluna a partir do nome.

        :param nome_coluna: Nome da coluna.
        :return: Descrição da coluna.
        """
        return DescricaoColuna[nome_coluna].value if nome_coluna in DescricaoColuna.__members__ else None
    
class DescricaoTipo(Enum):
    """
    Classe de Enums para armazenar os tipos das colunas do DataFrame.
    """
    voos = str
    companhia_aerea = str
    codigo_tipo_linha = str
    partida_prevista = datetime
    partida_real = datetime
    chegada_prevista = datetime
    chegada_real = datetime
    situacao_voo = str
    codigo_justificativa = str
    aeroporto_origem = str
    cidade_origem = str
    estado_origem = str
    pais_origem = str
    aeroporto_destino = str
    cidade_destino = str
    estado_destino = str
    pais_destino = str
    lat_orig = int
    long_orig = int
    lat_dest = int
    long_dest = int
    distancia_km = float

    @staticmethod
    def get(nome_coluna):
        """
        Método estático para recuperar a descrição da coluna a partir do nome.

        :param nome_coluna: Nome da coluna.
        :return: Descrição da coluna.
        """
        return DescricaoTipo[nome_coluna].value if nome_coluna in DescricaoTipo.__members__ else None

class Utils:
    
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
    def formatar_data_hora(dataframe: pd.DataFrame, coluna: str) -> pd.Series:
        """
        Formata uma coluna de um DataFrame como uma string no formato '%d/%m/%Y %H:%M:%S'.

        Args:
            dataframe: O DataFrame contendo a coluna a ser formatada.
            coluna: O nome da coluna a ser formatada.

        Returns:
            A coluna formatada como uma Series do pandas.
        """
        dataframe[coluna] = pd.to_datetime(dataframe[coluna])
        dataframe[coluna] = dataframe[coluna].dt.strftime('%d/%m/%Y %H:%M:%S')
        return dataframe[coluna]
