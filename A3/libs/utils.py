from enum import Enum
from datetime import datetime
import time
from typing import Any, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk

class Utils(Enum):
    """
    Classe utilitária que contém os metodos auxiliares para o EDA do dataset de voos.
    """
    num_voo = {'tipo': str, 'descricao': 'Número do voo'}
    companhia_aerea = {'tipo': str, 'descricao': 'Nome da companhia aérea'}
    codigo_tipo_linha = {'tipo': str, 'descricao': 'Código do tipo de linha (internacional)'}

    rota = {'tipo': str, 'descricao': 'Origem x Destino'}
    periodo_ferias = {'tipo': str, 'descricao': 'Contém o mês condizente a um período de férias (Janeiro, Julho ou Dezembro)'}

    dia_semana = {'tipo': str, 'descricao': 'Contém o dia da semana para data de partida'}

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
    def atualizar_situacao_voo(situacao: str, partida_atrasou: str, chegada_atrasou: str) -> str:
        """
        Atualiza a situação do voo com base nas informações de atraso de partida e chegada.

        Args:
            situacao (str): Situação atual do voo.
            partida_atrasou (str): Indica se houve atraso na partida do voo.
            chegada_atrasou (str): Indica se houve atraso na chegada do voo.

        Returns:
            str: Situação atualizada do voo.
        """
        if situacao == 'Realizado' and (partida_atrasou == 'S' or chegada_atrasou == 'S'):
            return 'Realizado Com Atraso'
        elif situacao == 'Realizado' and partida_atrasou == 'N' and chegada_atrasou == 'N':
            return 'Realizado Sem Atraso'
        else:
            return situacao

    @staticmethod
    def atualizar_justificativa_atraso(partida_atrasou: str, chegada_atrasou: str, codigo_justificativa: str) -> str:
        """
        Atualiza a justificativa de atraso com base nas informações de atraso de partida e chegada.

        Args:
            partida_atrasou (str): Indica se houve atraso na partida do voo.
            chegada_atrasou (str): Indica se houve atraso na chegada do voo.
            codigo_justificativa (str): Código da justificativa de atraso.

        Returns:
            str: Justificativa de atraso atualizada.
        """
        if partida_atrasou == 'S' or chegada_atrasou == 'S':
            return Utils.motivo_atraso(codigo_justificativa)
        return ''
    
    @staticmethod
    def atualizar_justificativa_cancelamento(situacao_voo: str, codigo_justificativa: str) -> str:
        """
        Atualiza a justificativa de cancelamento com base na situação do voo.

        Args:
            situacao_voo (str): Situação atual do voo.
            codigo_justificativa (str): Código da justificativa de cancelamento.

        Returns:
            str: Justificativa de cancelamento atualizada.
        """
        if situacao_voo == 'Cancelado':
            return Utils.motivo_cancelamento(codigo_justificativa)
        return ''

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
    
    @staticmethod
    def atribuir_dia_semana(dia_semana: str) -> str:
        """
        Atribui o nome do dia da semana em português com base no dia da semana em inglês fornecido.

        Parâmetros:
            dia_semana (str): O nome do dia da semana em inglês.

        Retorna:
            str: O nome do dia da semana em português.

        Exemplo:
            >>> atribuir_dia_semana('Monday')
            'segunda-feira'
        """
        dias_semana = {
            'Monday': 'segunda-feira', 
            'Tuesday': 'terça-feira', 
            'Wednesday': 'quarta-feira', 
            'Thursday': 'quinta-feira', 
            'Friday': 'sexta-feira', 
            'Saturday': 'sábado', 
            'Sunday': 'domingo'
        }
        
        return dias_semana[dia_semana]

    @staticmethod
    def criar_rota(codigo_tipo_linha, pais_origem, pais_destino, estado_origem, estado_destino, cidade_origem, cidade_destino):
        """
        Cria a rota com base no tipo de linha.

        Parâmetros:
        - codigo_tipo_linha (str): O código do tipo de linha.
        - pais_origem (str): O país de origem.
        - pais_destino (str): O país de destino.
        - estado_origem (str): O estado de origem.
        - estado_destino (str): O estado de destino.
        - cidade_origem (str): A cidade de origem.
        - cidade_destino (str): A cidade de destino.

        Retorna:
        - str: A rota criada com base no tipo de linha.

        Exemplo de uso:
        rota = Utils.criar_rota('Nacional', 'Brasil', 'Argentina', 'São Paulo', 'Buenos Aires', 'São Paulo', 'Buenos Aires')
        print(rota)  # Saída: "São Paulo - Buenos Aires"
        """
        if codigo_tipo_linha == 'Internacional':
            return pais_origem + ' - ' + pais_destino
        elif codigo_tipo_linha == 'Nacional':
            return estado_origem + ' - ' + estado_destino
        else:
            return cidade_origem + ' - ' + cidade_destino


class Plot:
    
    @staticmethod
    def periodo_ferias_geral(dataframe: pd.DataFrame, periodo_ferias: List[str], grid: bool, context: str, figsize, suptitle: str) -> Any:
        """
        Gera gráficos de barras para os períodos de férias especificados.

        Args:
            dataframe (pd.DataFrame): O dataframe contendo os dados.
            periodo_ferias (List[str]): Uma lista com os períodos de férias a serem plotados.
            grid (bool): Indica se as linhas de grade devem ser exibidas nos gráficos.
            context (str): O estilo de contexto do matplotlib a ser aplicado aos gráficos.
            figsize (Tuple[str]): Uma tupla indicando as dimensões da figura dos gráficos.
            suptitle (str): O título da figura principal que envolve todos os gráficos.

        Returns:
            Any: Os objetos de gráfico gerados.
        """
        with plt.style.context(context):
            fig = plt.figure(figsize=figsize)
            
            ax1 = plt.subplot2grid((2, 3), (0, 0))
            ax2 = plt.subplot2grid((2, 3), (0, 1))
            ax3 = plt.subplot2grid((2, 3), (0, 2))

            # Dados para os gráficos de barras
            periodo = periodo_ferias[0]
            df = dataframe.query("periodo_ferias == @periodo").reset_index(drop=True)
            bar_values = df['codigo_tipo_linha']
            bar_heights_1 = df['realizados_s_atraso']
            bar_heights_2 = df['realizados_c_atraso']
            bar_heights_3 = df['cancelados']
            bar_width = 0.2

            bar_pos_1 = np.arange(len(bar_values))
            bar_pos_2 = bar_pos_1 + bar_width + 0.1
            bar_pos_3 = bar_pos_1 + 2*(bar_width + 0.1)

            # Certifique-se de que as barras estejam deslocadas uma da outra
            ax1.barh(y=bar_pos_1, width=bar_heights_1, height=bar_width, label='Realizados S/Atraso')
            ax1.barh(y=bar_pos_2, width=bar_heights_2, height=bar_width, label='Realizados C/Atraso')
            ax1.barh(y=bar_pos_3, width=bar_heights_3, height=bar_width, label='Cancelados')
            ax1.set_yticks(bar_pos_2)
            ax1.set_yticklabels(bar_values)
            ax1.set_xticks([])
            ax1.grid(grid)
            ax1.set_title(f"Período de {periodo.title()}", fontsize=14)

            # Adicionar totais acima das barras 'top', 'bottom', 'center', 'baseline', 'center_baseline'
            for i, v in enumerate(bar_heights_1):
                ax1.annotate(str(v), xy=(v + 0.2, bar_pos_1[i] + bar_width/2), xytext=(5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_2):
                ax1.annotate(str(v), xy=(v + 0.2, bar_pos_2[i] + bar_width/2), xytext=(5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_3):
                ax1.annotate(str(v), xy=(v + 0.2, bar_pos_3[i] + bar_width/2), xytext=(5, 0), textcoords="offset points", color='white', ha='left', va='top')

            # Dados para os gráficos de barras
            periodo = periodo_ferias[1]
            df = dataframe.query("periodo_ferias == @periodo").reset_index(drop=True)
            bar_values = df['codigo_tipo_linha']
            bar_heights_1 = df['realizados_s_atraso']
            bar_heights_2 = df['realizados_c_atraso']
            bar_heights_3 = df['cancelados']
            bar_width = 0.2

            bar_pos_1 = np.arange(len(bar_values))
            bar_pos_2 = bar_pos_1 + bar_width + 0.1
            bar_pos_3 = bar_pos_1 + 2*(bar_width + 0.1)

            # Certifique-se de que as barras estejam deslocadas uma da outra
            ax2.barh(y=bar_pos_1, width=bar_heights_1, height=bar_width, label='Realizados S/Atraso')
            ax2.barh(y=bar_pos_2, width=bar_heights_2, height=bar_width, label='Realizados C/Atraso')
            ax2.barh(y=bar_pos_3, width=bar_heights_3, height=bar_width, label='Cancelados')
            ax2.set_yticks(bar_pos_2)
            ax2.set_yticklabels([])
            ax2.set_xticks([])  # Remove os valores do eixo x
            ax2.grid(grid)
            ax2.set_title(f"Período de {periodo.title()}", fontsize=14)

            # Adicionar totais acima das barras 'top', 'bottom', 'center', 'baseline', 'center_baseline'
            for i, v in enumerate(bar_heights_1):
                ax2.annotate(str(v), xy=(v + 0.2, bar_pos_1[i] + bar_width/2), xytext=(5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_2):
                ax2.annotate(str(v), xy=(v + 0.2, bar_pos_2[i] + bar_width/2), xytext=(5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_3):
                ax2.annotate(str(v), xy=(v + 0.2, bar_pos_3[i] + bar_width/2), xytext=(5, 0), textcoords="offset points", color='white', ha='left', va='top')

            # Dados para os gráficos de barras
            periodo = periodo_ferias[2]
            df = dataframe.query("periodo_ferias == @periodo").reset_index(drop=True)
            bar_values = df['codigo_tipo_linha']
            bar_heights_1 = df['realizados_s_atraso']
            bar_heights_2 = df['realizados_c_atraso']
            bar_heights_3 = df['cancelados']
            bar_width = 0.2

            bar_pos_1 = np.arange(len(bar_values))
            bar_pos_2 = bar_pos_1 + bar_width + 0.1
            bar_pos_3 = bar_pos_1 + 2*(bar_width + 0.1)

            # Certifique-se de que as barras estejam deslocadas uma da outra
            ax3.barh(y=bar_pos_1, width=bar_heights_1, height=bar_width, label='Realizados S/Atraso')
            ax3.barh(y=bar_pos_2, width=bar_heights_2, height=bar_width, label='Realizados C/Atraso')
            ax3.barh(y=bar_pos_3, width=bar_heights_3, height=bar_width, label='Cancelados')
            ax3.set_yticks(bar_pos_2)
            ax3.set_yticklabels([])
            ax3.set_xticks([])
            ax3.grid(grid)
            ax3.set_title(f"Período de {periodo.title()}", fontsize=14)

            # Adicionar totais acima das barras 'top', 'bottom', 'center', 'baseline', 'center_baseline'
            for i, v in enumerate(bar_heights_1):
                ax3.annotate(str(v), xy=(v + 0.2, bar_pos_1[i] + bar_width/2), xytext=(5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_2):
                ax3.annotate(str(v), xy=(v + 0.2, bar_pos_2[i] + bar_width/2), xytext=(5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_3):
                ax3.annotate(str(v), xy=(v + 0.2, bar_pos_3[i] + bar_width/2), xytext=(5, 0), textcoords="offset points", color='white', ha='left', va='top')

            plt.suptitle(suptitle, fontsize=25)
            plt.legend(["Realizados s/ Atraso", "Realizados c/ Atraso", "Cancelados"], loc='best') 
            plt.show()



class AnacVoos:
    """Classe que representa dados de voos da ANAC (Agência Nacional de Aviação Civil)."""

    dados = None
    dados_solidos = None
    total_arquivos = 0
    total_registros = 0
    dados_solidos = False
    tempo_execucao = 0
    
    @classmethod
    @property
    def periodo_ferias(cls):
        if cls.dados_solidos:
            periodos = cls.dados[cls.dados['periodo_ferias'] != '']
            return periodos['periodo_ferias'].unique().tolist()
        return []
    
    @classmethod
    def get_voos_ferias_agg(cls, filtrar_periodo_ferias: bool, percentuais, round: int, cols_groupby) -> pd.DataFrame:

        if not cls.dados_solidos:
            raise ValueError("Os dados não foram carregados ou foram comprometidos na transformação.")
            
        if filtrar_periodo_ferias: df = cls.__get_voos_ferias_regras(get_voos_ferias(), cols_groupby)
        else: df = cls.__get_voos_ferias_regras(AnacVoos.dados, cols_groupby)

        if len(percentuais) > 0:
            for cols in percentuais:
                df[cols[0]] = (df[cols[1]] / df['voos']).round(round)

        return df
    
    @classmethod
    def __get_voos_ferias_regras(dataframe: pd.DataFrame, cols_groupby) -> pd.DataFrame:
        return dataframe.groupby(cols_groupby).agg(
            voos=(cols_groupby[0], 'size'),
            realizados_s_atraso=('situacao_voo', lambda x: (x == 'Realizado Sem Atraso').sum()),
            realizados_c_atraso=('situacao_voo', lambda x: (x == 'Realizado Com Atraso').sum()),
            cancelados=('situacao_voo', lambda x: (x == 'Cancelado').sum())
        ).reset_index()
    
    @classmethod
    def get_voos_ferias(cls) -> pd.DataFrame:
        if not cls.dados_solidos:
            raise ValueError("Os dados não foram carregados ou foram comprometidos na transformação.")

        return cls.dados[cls.dados['periodo_ferias'] != '']

    @classmethod
    def get_voos_ferias_tipo_linha(cls, codigo_tipo_linha: str, percentuais: List[List[str]], round: int) -> pd.DataFrame:
        """
        Retorna um DataFrame filtrado pelos voos de férias de acordo com o tipo de linha especificado.
        
        Parâmetros:
            - codigo_tipo_linha (str): O tipo de linha dos voos a serem filtrados.
            - percentuais (List[List[str]]): Uma lista de listas contendo os pares de colunas que serão calculados como percentuais.
            - round (int): O número de casas decimais para arredondar os percentuais.
            - reindex (bool): Indica se as colunas do DataFrame devem ser reindexadas na ordem especificada.
        
        Retorno:
            Um DataFrame filtrado com as colunas especificadas e os percentuais calculados.
        """
        if not cls.dados_solidos:
            raise ValueError("Os dados não foram carregados ou foram comprometidos na transformação.")

        dataframe: pd.DataFrame = cls.dados

        df_filtrado = dataframe[dataframe['codigo_tipo_linha'] == codigo_tipo_linha] \
            .groupby(['periodo_ferias', 'rota']) \
            .agg(
                distancia_media_km=('distancia_km', 'mean'),
                voos=('distancia_km', 'size'),
                realizados_s_atraso=('situacao_voo', lambda x: (x == 'Realizado').sum()),
                realizados_c_atraso=('partida_atrasou', lambda x: (x == 'S').sum()),
                cancelados=('situacao_voo', lambda x: (x == 'Cancelado').sum()),
            ) \
            .reset_index().sort_values(by=['voos', 'distancia_media_km'], ascending=[False, True])
        
        for cols in percentuais:
            df_filtrado[cols[0]] = (df_filtrado[cols[1]] / df_filtrado['voos']).round(round)
        
        df_filtrado = df_filtrado.reindex(columns=['periodo_ferias', 'rota', 'distancia_media_km', 'voos', 'realizados_s_atraso', 'tx_realizados', 'realizados_c_atraso', 'tx_atrasos', 'cancelados', 'tx_cancelados'])
        
        return df_filtrado
    
