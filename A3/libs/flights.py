import os
import re
import pandas as pd
from enum import Enum
from datetime import datetime


class CSV:
    """
    Classe estática para carregar arquivos CSV em um DataFrame.
    """

    _encoding = "utf-8"
    _extensao_base = ".csv"
    _caminho_base = ["bases"]

    @staticmethod
    def carregar_base(arquivo_csv: str) -> pd.DataFrame:
        """
        Carrega um arquivo CSV e retorna um DataFrame.

        :param arquivo_csv: O caminho do arquivo CSV a ser carregado.
        :type arquivo_csv: str
        :return: Um DataFrame contendo os dados do arquivo CSV.
        :rtype: pd.DataFrame
        :example:
        >>> df = CSV.carregar_base("dados.csv")
        """
        if not arquivo_csv.lower().endswith(CSV._extensao_base):
            arquivo_csv = arquivo_csv.split(".")[0] + CSV._extensao_base

        caminho_arquivo = CSV._recuperar_caminho_arquivo(arquivo_csv)
        return pd.read_csv(caminho_arquivo, encoding=CSV._encoding)

    @staticmethod
    def carregar_bases() -> pd.DataFrame:
        """
        Carrega vários arquivos CSV de um diretório e os concatena em um único DataFrame.

        :return: Um DataFrame contendo os dados de todos os arquivos CSV carregados.
        :rtype: pd.DataFrame
        :example:
        >>> df = CSV.carregar_bases()
        """

        caminho = CSV._recuperar_caminho_base()
        arquivos = [arquivo for arquivo in os.listdir(caminho) if arquivo.endswith(".csv")]
        dataframes = [pd.read_csv(os.path.join(caminho, arquivo)) for arquivo in arquivos]
        return pd.concat(dataframes, ignore_index=True)

    @classmethod
    def _recuperar_caminho_arquivo(self, arquivo_csv: str) -> str:
        """
        Retorna o caminho completo do arquivo CSV.

        :param arquivo_csv: O nome do arquivo CSV a ser carregado.
        :type arquivo_csv: str
        :return: Uma string representando o caminho completo do arquivo CSV.
        :rtype: str
        """
        return os.path.abspath(os.path.join(self._recuperar_caminho_base(), arquivo_csv))

    @classmethod
    def _recuperar_caminho_base(self) -> str:
        """
        Retorna o caminho completo da pasta de bases.

        :return: Uma string representando o caminho completo da pasta de bases.
        :rtype: str
        """
        current_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))

        return os.path.join(current_dir, *self._caminho_base)

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
    longdest = 'Longitude do aeroporto de destino'
    latdest = 'Latitude do aeroporto de destino'
    longorig = 'Longitude do aeroporto de origem'
    latorig = 'Latitude do aeroporto de origem'

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
    longdest = int
    latdest = int
    longorig = int
    latorig = int

    @staticmethod
    def get(nome_coluna):
        """
        Método estático para recuperar a descrição da coluna a partir do nome.

        :param nome_coluna: Nome da coluna.
        :return: Descrição da coluna.
        """
        return DescricaoTipo[nome_coluna].value.__name__ if nome_coluna in DescricaoTipo.__members__ else None

class Flights:

    _dataframe_base = None
    _dataframe_description = None
    _dataframe_props = [
                            dict(selector='th', props=[('text-align', 'center')]),
                            dict(selector='td', props=[('text-align', 'center')])
                        ]

    @classmethod
    def _carregar_bases(cls) -> None:
        """
        Load data from CSV files and returns a DataFrame.

        :return: A pandas DataFrame containing the data of all loaded CSV files.
        :rtype: pd.DataFrame
        """
        if cls._dataframe_base is None:
            cls._dataframe_base = CSV.carregar_bases()

    @classmethod
    def _tratar_colunas(cls) -> None:
        """
        Normaliza os nomes das colunas do DataFrame, deixando-os mais amigáveis e atribuindo descrição a elas.
        Retorna None caso a base ainda não tenha sido carregada.

        :return: None
        """
        if cls._dataframe_base is None:
            print('A base ainda não foi carregada.')
            return
        
        colunas_tratadas = []

        for coluna in cls._dataframe_base.columns:
            coluna = coluna.lower().replace('.', '_')
            if any(char.isupper() for char in coluna):
                coluna = '_'.join(word.lower() for word in re.findall('[A-Z][^A-Z]*', coluna))
            colunas_tratadas.append(coluna)
        
        cls._dataframe_base.columns = colunas_tratadas

    @classmethod
    def _carregar_descricao(cls) -> None:
        """
        Carrega a descrição das colunas do dataframe e aplica um estilo de formatação para a visualização.

        :return: None.
        """
        colunas = ["Coluna", "Tipo", "Descrição"]
        dados = []

        for col in cls._dataframe_base.columns:
            nome_coluna = cls._dataframe_base[col].name
            tipo_coluna = DescricaoTipo.get(nome_coluna)
            descricao_coluna = DescricaoColuna.get(nome_coluna)

            dados.append([nome_coluna, tipo_coluna, descricao_coluna])

        cls._dataframe_description = pd.DataFrame(data=dados, columns=colunas)

    def __init__(self, carregar_bases: bool, tratar_colunas: bool):

        if carregar_bases:
            self._carregar_bases()
        
        if tratar_colunas:
            self._tratar_colunas()
        
        if self._dataframe_description is None:
            self._carregar_descricao()

    def carregar_base(self, arquivo_csv: str) -> pd.DataFrame:
        """
        Carrega um arquivo CSV através da classe CSV, se ainda não foi carregado, e retorna um objeto DataFrame.

        :param arquivo_csv: O caminho do arquivo CSV a ser carregado.
        :type arquivo_csv: str
        :return: Um objeto DataFrame contendo os dados do arquivo repositorio_csv.
        :rtype: pd.DataFrame
        """
        if self._dataframe_base is None:
            self._dataframe_base = CSV.carregar_base(self._pd, arquivo_csv)

        return self._dataframe_base

    @property
    def descricao(self):
        """
        Retorna um DataFrame com informações descritivas sobre as colunas do DataFrame carregado,
        incluindo seus tipos e descrições.

        Returns
        -------
        pandas.DataFrame
            DataFrame contendo informações descritivas das colunas.
        """
        if self._dataframe_base is None:
            print('A base ainda não foi carregada.')
            return

        self._dataframe_description = self._dataframe_description.style.set_table_styles(self._dataframe_props)

        return self._dataframe_description.hide(axis='index')

    @property
    def base(self):
        """
        Retorna o DataFrame da base de dados.

        :return: Um DataFrame contendo os dados da base de dados.
        :rtype: pd.DataFrame ou None
        """
        if self._dataframe_base is None:
            print('A base ainda não foi carregada.')
            return None
        
        self._dataframe_base = self._dataframe_base.head().style.set_table_styles(self._daataframe_props)

        return self._dataframe_base
