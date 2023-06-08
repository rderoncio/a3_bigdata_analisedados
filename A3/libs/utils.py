from enum import Enum
from datetime import datetime
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Utils(Enum):
    """
    Classe utilitária que contém os metodos auxiliares para o EDA do dataset de voos.
    """
    num_voo = {'tipo': str, 'descricao': 'Número do voo'}
    companhia_aerea = {'tipo': str, 'descricao': 'Nome da companhia aérea'}
    codigo_tipo_linha = {'tipo': str, 'descricao': 'Código do tipo de linha (internacional)'}

    rota = {'tipo': str, 'descricao': 'Origem x Destino'}
    periodo_ferias = {'tipo': str, 'descricao': 'Contém o mês condizente a um período de férias (Janeiro, Julho ou Dezembro)'}

    aeroporto_origem = {'tipo': str, 'descricao': 'Código do aeroporto de origem'}
    cidade_origem = {'tipo': str, 'descricao': 'Cidade de origem'}
    estado_origem = {'tipo': str, 'descricao': 'Estado de origem'}
    pais_origem = {'tipo': str, 'descricao': 'País de origem'}
    lat_orig = {'tipo': float, 'descricao': 'Latitude do aeroporto de origem'}
    long_orig = {'tipo': float, 'descricao': 'Longitude do aeroporto de origem'}
    partida_prevista = {'tipo': datetime, 'descricao': 'Horário previsto de partida'}
    partida_real = {'tipo': datetime, 'descricao': 'Horário real de partida'}
    partida_atrasou = {'tipo': bool, 'descricao': 'Houve atraso na partida do voo?'}

    aeroporto_destino = {'tipo': str, 'descricao': 'Código do aeroporto de destino'}
    cidade_destino = {'tipo': str, 'descricao': 'Cidade de destino'}
    estado_destino = {'tipo': str, 'descricao': 'Estado de destino'}
    pais_destino = {'tipo': str, 'descricao': 'País de destino'}
    lat_dest = {'tipo': float, 'descricao': 'Latitude do aeroporto de destino'}
    long_dest = {'tipo': float, 'descricao': 'Longitude do aeroporto de destino'}
    chegada_prevista = {'tipo': datetime, 'descricao': 'Horário previsto de chegada'}
    chegada_real = {'tipo': datetime, 'descricao': 'Horário real de chegada'}
    chegada_atrasou = {'tipo': bool, 'descricao': 'Houve atraso na chegada do voo?'}

    distancia_km = {'tipo': float, 'descricao': 'Total em quilômetros entre o local de origem e destino'}
    
    situacao_voo = {'tipo': str, 'descricao': 'Situação do voo (realizado, cancelado, etc)'}
    codigo_justificativa = {'tipo': str, 'descricao': 'Código de justificativa (caso o voo tenha sido cancelado)'}
    justificativa_atraso = {'tipo': str, 'descricao': 'Informação e correlata à justificativa no atraso do voo'}
    justificativa_cancelamento = {'tipo': str, 'descricao': 'Informação e correlata à justificativa do cancelamento do voo'}

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
    def calcular_distancia_km(lat_orig: float, long_orig: float, lat_dest: float, long_dest: float) -> float:
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

    @staticmethod
    def atribuir_periodo_ferias(data: pd.Series) -> pd.Series:
        """
        Atribui o valor do período de férias com base no mês da data fornecida.

        Parâmetros:
        - data: pd.Series
            Coluna do tipo pd.Series contendo as datas formatadas no formato '%d/%m/%Y %H:%M:%S' para análise do mês.

        Retorna:
        - pd.Series
            Coluna do tipo pd.Series contendo os valores correspondentes ao período de férias ('janeiro', 'julho' ou 'dezembro')
            se o mês da data fornecida estiver na lista [1, 7, 12]. Caso contrário, retorna uma string vazia.

        Exemplo:
        >>> atribuir_periodo_ferias(pd.Series(['15/01/2023 10:30:00', '10/02/2023 15:45:00']))
        0    janeiro
        1           
        dtype: object
        """
        ferias = {'janeiro': 1, 'julho': 7, 'dezembro': 12}
        
        def atribuir_periodo(data):
            mes = datetime.strptime(data, '%d/%m/%Y %H:%M:%S').month
            if mes in ferias.values():
                return next(key for key, value in ferias.items() if value == mes)
            else:
                return ''
        return data.apply(atribuir_periodo)

    @staticmethod
    def validar_atraso(situacao: str, dt_prevista: str, dt_real: str) -> str:
        """
        Verifica se um voo está atrasado comparando os horários previsto e real de partida/chegada.
        Se a situação do voo for 'cancelado', retorna uma string vazia.
        Se os horários previsto e real forem iguais ou se o horário previsto for maior ou igual ao horário real,
        retorna 'N', caso contrário, retorna 'S'.

        Args:
            situacao (str): Situação do voo ('cancelado' ou 'realizado').
            prevista (str): Horário previsto de partida/chegada no formato 'dd/mm/aaaa hh:mm:ss'.
            real (str): Horário real de partida/chegada no formato 'dd/mm/aaaa hh:mm:ss'.

        Returns:
            str: 'N' caso o voo esteja atrasado ou 'S' caso contrário. Retorna uma string vazia se o voo foi cancelado.
        """

        if situacao.lower() == 'cancelado':
            return ''
        
        prevista_datetime = datetime.strptime(dt_prevista, '%d/%m/%Y %H:%M:%S')
        real_datetime = datetime.strptime(dt_real, '%d/%m/%Y %H:%M:%S')

        if prevista_datetime >= real_datetime or prevista_datetime == real_datetime:
            return 'N'
        else:
            return 'S'

    @staticmethod
    def motivo_atraso(motivo: str) -> str:
        atrasos = {
            'AEROPORTO COM RESTRICOES OPERACIONAIS': 'Restrições operacionais no aeroporto',
            'ALTERNATIVA ABAIXO DOS LIMITES': 'Alternativa abaixo dos limites',
            'ANTECIPACAO DE HORARIO AUTORIZADA - ESPECIFICO VOOS INTERNACIONAIS': 'Antecipação de horário',
            'ATRASO AEROPORTO DE ALTERNATIVA - CONDICOES METEOROLOGICAS': 'Condições meteorológicas',
            'ATRASO AEROPORTO DE ALTERNATIVA - ORDEM TECNICA': 'Ordem técnica',
            'ATRASOS NAO ESPECIFICOS - OUTROS': 'Outros',
            'CONEXAO AERONAVE/VOLTA - VOO DE IDA NAO PENALIZADO AEROPORTO INTERDITADO': 'Interdição do aeroporto',
            'CONEXAO AERONAVE/VOLTA - VOO DE IDA NAO PENALIZADO CONDICOES METEOROLOGICAS': 'Condições meteorológicas',
            'CONEXAO DE AERONAVE': 'Conexão de aeronave',
            'DEFEITOS DA AERONAVE': 'Defeitos da aeronave',
            'DEGELO E REMOCAO DE NEVE E/OU LAMA EM AERONAVE': 'Degelo/Limpeza',
            'INCLUSAO DE ETAPA (AEROPORTO DE ALTERNATIVA) DEVIDO A UM VOO ESPECIAL RETORNO': 'Inclusão de etapa',
            'OPERACAO DE VOO COM MAIS DE 04 HORAS DE ATRASO PANE AERONAVE': 'Pane na aeronave',
            'TROCA DE AERONAVE': 'Troca de aeronave'
        }

        return atrasos.get(motivo, '')

    @staticmethod
    def motivo_cancelamento(motivo: str) -> str:
        cancelamentos = {
            'CANCELAMENTO - CONEXAO AERONAVE/VOLTA - VOO DE IDA CANCELADO - AEROPORTO INTERDITADO': 'Interdição do aeroporto',
            'CANCELAMENTO - CONEXAO AERONAVE/VOLTA - VOO DE IDA CANCELADO - CONDICOES METEOROLOGICAS': 'Condições meteorológicas',
            'CANCELAMENTO POR MOTIVOS TECNICOS - OPERACIONAIS': 'Motivos técnicos-operacionais',
            'FALTA PAX COM PASSAGEM MARCADA - ( APENAS PARA AS LINHAS AEREAS DOMESTICAS REGIONAIS)': 'Falta de passageiros com passagem marcada',
            'PROGRAMADO - FERIADO NACIONAL': 'Feriado nacional'
        }

        return cancelamentos.get(motivo, '')

    @staticmethod
    def nome_companhia_aerea_normalizado(companhia_aerea: str) -> str:
        """
        Retorna o nome normalizado de uma companhia aérea.

        Parâmetros:
            - companhia_aerea (str): O nome da companhia aérea.

        Retorna:
            str: O nome normalizado da companhia aérea, se estiver presente na lista de normalizados.
                Caso contrário, retorna o próprio nome sem alterações.

        Exemplo:
            nome_companhia_aerea_normalizado('AEROLINEAS ARGENTINAS')
            Saída: 'Aerolineas Argentinas'
        """
        normalizados = [
                {'AEROLINEAS ARGENTINAS': 'Aerolineas Argentinas'},
                {'AIR CANADA': 'Air Canada'},
                {'AIR CHINA': 'Air China'},
                {'AIR EUROPA S/A': 'Air Europa'},
                {'AIR FRANCE': 'Air France'},
                {'AIRES - LAN COLOMBIA': 'Aires'},
                {'ALITALIA': 'Alitalia'},
                {'AMERICAN AIRLINES INC': 'American Airlines'},
                {'AUSTRAL LINEAS AÉREAS CIELOS DEL SUR S.A': 'Austral Lineas Aereas'},
                {'AVIANCA': 'Avianca'},
                {'AVIANCA BRASIL': 'Avianca Brasil'},
                {'AZUL': 'Azul'},
                {'BOLIVIANA DE AVIACION': 'Boliviana de Aviacion'},
                {'BRITISH AIRWAYS PLC': 'British Airways'},
                {'CONDOR FLUGDINST': 'Condor Flugdinst'},
                {'COPA -COMPANIA PANAMENA DE AVIACION': 'Copa'},
                {'DELTA AIRLINES': 'Delta Airlines'},
                {'EDELWEISS': 'Edelweiss'},
                {'EMIRATES': 'Emirates'},
                {'EMPRESA DE TRANSPORTES AEREOS DE CABO VERDE S.A.': 'Empresa de Transportes Aereos de Cabo Verde'},
                {'ETHIOPIAN': 'Ethiopian'},
                {'ETIHAD': 'Etihad'},
                {'FLYWAYS': 'Flyways'},
                {'GOL': 'Gol'},
                {'IBERIA': 'Iberia'},
                {'INSELAIR': 'InselAir'},
                {'KLM ROYAL DUTCH AIRLINES': 'KLM Royal Dutch Airlines'},
                {'KOREAN AIRLINES': 'Korean Airlines'},
                {'LAN ARGENTINA S/A': 'LAN Argentina'},
                {'LAN CHILE': 'LAN Chile'},
                {'LAN PERU S/A': 'LAN Peru'},
                {'LUFTHANSA': 'Lufthansa'},
                {'MAP LINHAS AEREAS': 'MAP Linhas Aereas'},
                {'MERIDIANA': 'Meridiana'},
                {'NAO INFORMADO': 'Nao Informado'},
                {'PASSAREDO': 'Passaredo'},
                {'QATAR AIRWAYS': 'Qatar Airways'},
                {'ROYAL AIR MAROC': 'Royal Air Maroc'},
                {'SINGAPORE AIRLINES': 'Singapore Airlines'},
                {'SOUTH AFRICAN AIRWAYS': 'South African Airways'},
                {'SURINAM AIRWAYS': 'Surinam Airways'},
                {'SWISSAIR': 'Swissair'},
                {'TAAG LINHAS AEREAS DE ANGOLA': 'TAAG Linhas Aereas de Angola'},
                {'TAM': 'TAM'},
                {'TAM TRANSP. AR. DEL. MERCOS': 'TAM Transp. Ar. Del. Mercos'},
                {'TAP AIR PORTUGAL': 'TAP Air Portugal'},
                {'TOTAL': 'Total'},
                {'TRASAMERICA  AIRLINES-TACAPERU': 'Trasamerica Airlines-TACA Peru'},
                {'TURKISH AIRLINES INC.': 'Turkish Airlines'},
                {'UNITED AIRLINES': 'United Airlines'}
            ]
        
        for item in normalizados:
            if companhia_aerea in item:
                return item[companhia_aerea]
        return companhia_aerea
    
    @staticmethod
    def centralizar_dataframe(dataframe: pd.DataFrame, limit: int | None = 10) -> None:
        """
        Centraliza um dataframe exibindo-o com alinhamento centralizado das células.
        
        Parâmetros:
        - dataframe: pandas.DataFrame: O dataframe a ser centralizado.
        - limit: int | None: Opcional. O número máximo de linhas a serem exibidas. Se None, exibe todas as linhas.
        
        Retorna:
        - pandas.DataFrame
        
        """
        dataframe_limit = dataframe.head(limit) if limit is not None else dataframe

        styled_dataframe = dataframe_limit.style.set_table_styles([
            dict(selector='th', props=[('text-align', 'center')]), 
            dict(selector='td', props=[('text-align', 'center')])
        ])

        return styled_dataframe
    
class Plot:
    pass

class AnacVoos:
    """Classe que representa dados de voos da ANAC (Agência Nacional de Aviação Civil)."""

    dados = None
    dados_solidos = None
    total_arquivos = 0
    total_registros = 0
    dados_solidos = False

    @classmethod
    def filtrar_tipo_linha(cls, codigo_tipo_linha):
        """
        Filtra os voos com base no código do tipo de linha e aeroporto de destino.

        Parâmetros:
        - codigo_tipo_linha: Código do tipo de linha a ser filtrado (por exemplo, 'Nacional', 'Internacional')
        - aeroporto_destino: Aeroporto de destino a ser filtrado

        Retorna:
        - DataFrame filtrado com os voos correspondentes ao código do tipo de linha e aeroporto de destino fornecidos.
        """
        return cls.dados[cls.dados['codigo_tipo_linha'].isin([codigo_tipo_linha])]

        voos_filtrados = cls.dados[(cls.dados['codigo_tipo_linha'] == codigo_tipo_linha) & (cls.dados['aeroporto_destino'] == aeroporto_destino)]
        return voos_filtrados