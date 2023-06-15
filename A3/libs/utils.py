from enum import Enum
from datetime import datetime
from typing import Any, Callable, List, Dict, Tuple
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
    codigo_tipo_linha = {
        'tipo': str, 'descricao': 'Código do tipo de linha (internacional)'}

    rota = {'tipo': str, 'descricao': 'Origem x Destino'}
    periodo_ferias = {
        'tipo': str, 'descricao': 'Contém o mês condizente a um período de férias (Janeiro, Julho ou Dezembro)'}

    dia_semana = {'tipo': str,
                  'descricao': 'Contém o dia da semana para data de partida'}

    aeroporto_origem = {'tipo': str,
                        'descricao': 'Código do aeroporto de origem'}
    cidade_origem = {'tipo': str, 'descricao': 'Cidade de origem'}
    estado_origem = {'tipo': str, 'descricao': 'Estado de origem'}
    pais_origem = {'tipo': str, 'descricao': 'País de origem'}
    lat_orig = {'tipo': float, 'descricao': 'Latitude do aeroporto de origem'}
    long_orig = {'tipo': float,
                 'descricao': 'Longitude do aeroporto de origem'}
    partida_prevista = {'tipo': datetime,
                        'descricao': 'Horário previsto de partida'}
    partida_real = {'tipo': datetime, 'descricao': 'Horário real de partida'}
    partida_atrasou = {'tipo': bool,
                       'descricao': 'Houve atraso na partida do voo?'}
    tempo_atraso_partida = {'tipo': str,
                            'descricao': 'Tempo total do atraso da partida.'}

    aeroporto_destino = {'tipo': str,
                         'descricao': 'Código do aeroporto de destino'}
    cidade_destino = {'tipo': str, 'descricao': 'Cidade de destino'}
    estado_destino = {'tipo': str, 'descricao': 'Estado de destino'}
    pais_destino = {'tipo': str, 'descricao': 'País de destino'}
    lat_dest = {'tipo': float, 'descricao': 'Latitude do aeroporto de destino'}
    long_dest = {'tipo': float,
                 'descricao': 'Longitude do aeroporto de destino'}
    chegada_prevista = {'tipo': datetime,
                        'descricao': 'Horário previsto de chegada'}
    chegada_real = {'tipo': datetime, 'descricao': 'Horário real de chegada'}
    chegada_atrasou = {'tipo': bool,
                       'descricao': 'Houve atraso na chegada do voo?'}
    tempo_atraso_chegada = {'tipo': str,
                            'descricao': 'Tempo total do atraso da chegada.'}

    distancia_km = {
        'tipo': float, 'descricao': 'Total em quilômetros entre o local de origem e destino'}

    situacao_voo = {
        'tipo': str, 'descricao': 'Situação do voo (realizado, cancelado, etc)'}
    codigo_justificativa = {
        'tipo': str, 'descricao': 'Código de justificativa (caso o voo tenha sido cancelado)'}
    justificativa_atraso = {
        'tipo': str, 'descricao': 'Informação e correlata à justificativa no atraso do voo'}
    justificativa_cancelamento = {
        'tipo': str, 'descricao': 'Informação e correlata à justificativa do cancelamento do voo'}

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
        lat_orig, long_orig, lat_dest, long_dest = map(
            np.radians, [lat_orig, long_orig, lat_dest, long_dest])

        # Calcula a diferença entre as latitudes e longitudes
        dlat = lat_dest - lat_orig
        dlong = long_dest - long_orig

        # Calcula a fórmula Haversine
        a = np.sin(dlat/2)**2 + np.cos(lat_orig) * \
            np.cos(lat_dest) * np.sin(dlong/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Retorna a distância em km
        return R * c

    @staticmethod
    def calcular_atraso(atrasou: str, prevista: str, real: str) -> str:
        """
        Calcula o tempo de atraso com base em uma chegada prevista e uma chegada real.

        Parâmetros:
            - chegada_prevista (str): String no formato 'dd/mm/yyyy HH:MM:SS' representando a chegada prevista.
            - chegada_real (str): String no formato 'dd/mm/yyyy HH:MM:SS' representando a chegada real.

        Retorna:
            str: Tempo de atraso no formato 'HH:MM:SS'.

        Exemplo de uso:
            chegada_prevista = '02/01/2015 06:30:00'
            chegada_real = '02/01/2015 06:35:15'

            atraso = calcular_atraso(chegada_prevista, chegada_real)
            print(atraso)  # Output: 00:05:15
        """
        if atrasou.upper() != 'S':
            return ''

        dt_prevista = datetime.strptime(prevista, '%d/%m/%Y %H:%M:%S')
        dt_real = datetime.strptime(real, '%d/%m/%Y %H:%M:%S')
        diferenca = dt_real - dt_prevista

        segundos_total = diferenca.total_seconds()
        horas, segundos_total = divmod(segundos_total, 3600)
        minutos, segundos = divmod(segundos_total, 60)

        return f'{int(horas):02}:{int(minutos):02}:{int(segundos):02}'

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
        dataframe[nome_coluna] = dataframe[nome_coluna].dt.strftime(
            '%d/%m/%Y %H:%M:%S')
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
        """
        Retorna o motivo de atraso com base no código informado.

        Parâmetros:
            motivo (str): O código do motivo de atraso.

        Retorna:
            str: O motivo de atraso correspondente ao código.

        Exemplo de uso:
            motivo = Utils.motivo_atraso('AEROPORTO COM RESTRICOES OPERACIONAIS')
            print(motivo)  # Saída: "Restrições operacionais no aeroporto"
        """
        default = 'Atraso não especificado'

        atrasos = {
            'AEROPORTO COM RESTRICOES OPERACIONAIS': 'Restrições operacionais no aeroporto',
            'ALTERNATIVA ABAIXO DOS LIMITES': 'Alternativa abaixo dos limites',
            'ANTECIPACAO DE HORARIO AUTORIZADA - ESPECIFICO VOOS INTERNACIONAIS': 'Antecipação de horário',
            'ATRASO AEROPORTO DE ALTERNATIVA - CONDICOES METEOROLOGICAS': 'Condições meteorológicas',
            'ATRASO AEROPORTO DE ALTERNATIVA - ORDEM TECNICA': 'Ordem técnica',
            'ATRASOS NAO ESPECIFICOS - OUTROS': default,
            'CONEXAO AERONAVE/VOLTA - VOO DE IDA NAO PENALIZADO AEROPORTO INTERDITADO': 'Interdição do aeroporto',
            'CONEXAO AERONAVE/VOLTA - VOO DE IDA NAO PENALIZADO CONDICOES METEOROLOGICAS': 'Condições meteorológicas',
            'CONEXAO DE AERONAVE': 'Conexão de aeronave',
            'DEFEITOS DA AERONAVE': 'Defeitos da aeronave',
            'DEGELO E REMOCAO DE NEVE E/OU LAMA EM AERONAVE': 'Degelo/Limpeza',
            'INCLUSAO DE ETAPA (AEROPORTO DE ALTERNATIVA) DEVIDO A UM VOO ESPECIAL RETORNO': 'Inclusão de etapa',
            'OPERACAO DE VOO COM MAIS DE 04 HORAS DE ATRASO PANE AERONAVE': 'Pane na aeronave',
            'TROCA DE AERONAVE': 'Troca de aeronave'
        }

        return atrasos.get(motivo, default)

    @staticmethod
    def motivo_cancelamento(motivo: str) -> str:
        """
        Retorna o motivo de cancelamento com base no código informado.

        Parâmetros:
            motivo (str): O código do motivo de cancelamento.

        Retorna:
            str: O motivo de cancelamento correspondente ao código.

        Exemplo de uso:
            motivo = Utils.motivo_cancelamento('CANCELAMENTO - CONEXAO AERONAVE/VOLTA - VOO DE IDA CANCELADO - AEROPORTO INTERDITADO')
            print(motivo)  # Saída: "Interdição do aeroporto"
        """
        default = 'Cancelamento não especificado'

        cancelamentos = {
            'CANCELAMENTO - CONEXAO AERONAVE/VOLTA - VOO DE IDA CANCELADO - AEROPORTO INTERDITADO': 'Interdição do aeroporto',
            'CANCELAMENTO - CONEXAO AERONAVE/VOLTA - VOO DE IDA CANCELADO - CONDICOES METEOROLOGICAS': 'Condições meteorológicas',
            'CANCELAMENTO POR MOTIVOS TECNICOS - OPERACIONAIS': 'Motivos técnicos-operacionais',
            'FALTA PAX COM PASSAGEM MARCADA - ( APENAS PARA AS LINHAS AEREAS DOMESTICAS REGIONAIS)': 'Falta de passageiros com passagem marcada',
            'PROGRAMADO - FERIADO NACIONAL': 'Feriado nacional'
        }

        return cancelamentos.get(motivo, default)

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
            {'EMPRESA DE TRANSPORTES AEREOS DE CABO VERDE S.A.':
             'Empresa de Transportes Aereos de Cabo Verde'},
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
        dataframe_limit = dataframe.head(
            limit) if limit is not None else dataframe

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
    def criar_rota(
        codigo_tipo_linha: str,
        pais_origem: str,
        pais_destino: str,
        estado_origem: str,
        estado_destino: str,
        cidade_origem: str,
        cidade_destino: str
    ) -> str:
        """
        Cria a rota com base no tipo de linha.

        Parâmetros:
            codigo_tipo_linha (str): O código do tipo de linha.
            pais_origem (str): O país de origem.
            pais_destino (str): O país de destino.
            estado_origem (str): O estado de origem.
            estado_destino (str): O estado de destino.
            cidade_origem (str): A cidade de origem.
            cidade_destino (str): A cidade de destino.

        Retorna:
            str: A rota criada com base no tipo de linha.

        Exemplo de uso:
            rota = Utils.criar_rota(
                'Nacional',
                'Brasil',
                'Argentina',
                'São Paulo',
                'Buenos Aires',
                'São Paulo',
                'Buenos Aires'
            )
            print(rota)  # Saída: "São Paulo - Buenos Aires"
        """
        if codigo_tipo_linha == 'Internacional':
            return pais_origem + ' - ' + pais_destino
        elif codigo_tipo_linha == 'Nacional':
            return estado_origem + ' - ' + estado_destino
        else:
            return cidade_origem + ' - ' + cidade_destino

    @staticmethod
    def converter_tempo_para_segundos(tempo: str) -> int:
        """
        Converte uma string de tempo no formato 'HH:MM:SS' para o total de segundos.

        Parâmetros:
            tempo (str): A string de tempo a ser convertida.

        Retorna:
            int: O total de segundos.

        Levanta:
            None.

        Exemplos:
            >>> converter_tempo_para_segundos('02:30:45')
            9045
            >>> converter_tempo_para_segundos('00:45:15')
            2715
        """
        if isinstance(tempo, str) and tempo.count(':') == 2:
            horas, minutos, segundos = map(int, tempo.split(':'))
            total_segundos = horas * 3600 + minutos * 60 + segundos
            return total_segundos
        else:
            return 0

    @staticmethod
    def converter_segundos_para_tempo(segundos: int) -> str:
        """
        Converte um número inteiro de segundos para uma string de tempo no formato 'HH:MM:SS'.

        Parâmetros:
            segundos (int): O número de segundos a ser convertido.

        Retorna:
            str: A string de tempo no formato 'HH:MM:SS'.

        Levanta:
            None.

        Exemplos:
            >>> converter_segundos_para_tempo(9045)
            '02:30:45'
            >>> converter_segundos_para_tempo(2715)
            '00:45:15'
        """
        try:
            horas = int(segundos) // 3600
            minutos = (int(segundos) % 3600) // 60
            segundos = int(segundos) % 60
            return f"{horas:02d}:{minutos:02d}:{segundos:02d}"
        except:
            return "00:00:00"

    @staticmethod
    def formatar_tempo_execucao(tempo: float) -> str:
        """
        Formata um tempo em segundos para o formato 'HH:MM:SS'.

        Parâmetros:
            tempo (float): O tempo em segundos a ser formatado.

        Retorna:
            str: O tempo formatado no formato 'HH:MM:SS'.

        Exemplo:
            >>> formatar_tempo_execucao(3610)
            '01:00:10'
        """
        horas = int(tempo) // 3600
        minutos = (int(tempo) % 3600) // 60
        segundos = int(tempo) % 60

        return f"{horas:02d}:{minutos:02d}:{segundos:02d}"


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
            df = dataframe.query(
                "periodo_ferias == @periodo").reset_index(drop=True)
            bar_values = df['codigo_tipo_linha']
            bar_heights_1 = df['realizados_s_atraso']
            bar_heights_2 = df['realizados_c_atraso']
            bar_heights_3 = df['cancelados']
            bar_width = 0.2

            bar_pos_1 = np.arange(len(bar_values))
            bar_pos_2 = bar_pos_1 + bar_width + 0.1
            bar_pos_3 = bar_pos_1 + 2*(bar_width + 0.1)

            # Certifique-se de que as barras estejam deslocadas uma da outra
            ax1.barh(y=bar_pos_1, width=bar_heights_1,
                     height=bar_width, label='Realizados S/Atraso')
            ax1.barh(y=bar_pos_2, width=bar_heights_2,
                     height=bar_width, label='Realizados C/Atraso')
            ax1.barh(y=bar_pos_3, width=bar_heights_3,
                     height=bar_width, label='Cancelados')
            ax1.set_yticks(bar_pos_2)
            ax1.set_yticklabels(bar_values)
            ax1.set_xticks([])
            ax1.grid(grid)
            ax1.set_title(f"Período de {periodo.title()}", fontsize=14)

            # Adicionar totais acima das barras 'top', 'bottom', 'center', 'baseline', 'center_baseline'
            for i, v in enumerate(bar_heights_1):
                ax1.annotate(str(v), xy=(v + 0.2, bar_pos_1[i] + bar_width/2), xytext=(
                    5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_2):
                ax1.annotate(str(v), xy=(v + 0.2, bar_pos_2[i] + bar_width/2), xytext=(
                    5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_3):
                ax1.annotate(str(v), xy=(v + 0.2, bar_pos_3[i] + bar_width/2), xytext=(
                    5, 0), textcoords="offset points", color='white', ha='left', va='top')

            # Dados para os gráficos de barras
            periodo = periodo_ferias[1]
            df = dataframe.query(
                "periodo_ferias == @periodo").reset_index(drop=True)
            bar_values = df['codigo_tipo_linha']
            bar_heights_1 = df['realizados_s_atraso']
            bar_heights_2 = df['realizados_c_atraso']
            bar_heights_3 = df['cancelados']
            bar_width = 0.2

            bar_pos_1 = np.arange(len(bar_values))
            bar_pos_2 = bar_pos_1 + bar_width + 0.1
            bar_pos_3 = bar_pos_1 + 2*(bar_width + 0.1)

            # Certifique-se de que as barras estejam deslocadas uma da outra
            ax2.barh(y=bar_pos_1, width=bar_heights_1,
                     height=bar_width, label='Realizados S/Atraso')
            ax2.barh(y=bar_pos_2, width=bar_heights_2,
                     height=bar_width, label='Realizados C/Atraso')
            ax2.barh(y=bar_pos_3, width=bar_heights_3,
                     height=bar_width, label='Cancelados')
            ax2.set_yticks(bar_pos_2)
            ax2.set_yticklabels([])
            ax2.set_xticks([])  # Remove os valores do eixo x
            ax2.grid(grid)
            ax2.set_title(f"Período de {periodo.title()}", fontsize=14)

            # Adicionar totais acima das barras 'top', 'bottom', 'center', 'baseline', 'center_baseline'
            for i, v in enumerate(bar_heights_1):
                ax2.annotate(str(v), xy=(v + 0.2, bar_pos_1[i] + bar_width/2), xytext=(
                    5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_2):
                ax2.annotate(str(v), xy=(v + 0.2, bar_pos_2[i] + bar_width/2), xytext=(
                    5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_3):
                ax2.annotate(str(v), xy=(v + 0.2, bar_pos_3[i] + bar_width/2), xytext=(
                    5, 0), textcoords="offset points", color='white', ha='left', va='top')

            # Dados para os gráficos de barras
            periodo = periodo_ferias[2]
            df = dataframe.query(
                "periodo_ferias == @periodo").reset_index(drop=True)
            bar_values = df['codigo_tipo_linha']
            bar_heights_1 = df['realizados_s_atraso']
            bar_heights_2 = df['realizados_c_atraso']
            bar_heights_3 = df['cancelados']
            bar_width = 0.2

            bar_pos_1 = np.arange(len(bar_values))
            bar_pos_2 = bar_pos_1 + bar_width + 0.1
            bar_pos_3 = bar_pos_1 + 2*(bar_width + 0.1)

            # Certifique-se de que as barras estejam deslocadas uma da outra
            ax3.barh(y=bar_pos_1, width=bar_heights_1,
                     height=bar_width, label='Realizados S/Atraso')
            ax3.barh(y=bar_pos_2, width=bar_heights_2,
                     height=bar_width, label='Realizados C/Atraso')
            ax3.barh(y=bar_pos_3, width=bar_heights_3,
                     height=bar_width, label='Cancelados')
            ax3.set_yticks(bar_pos_2)
            ax3.set_yticklabels([])
            ax3.set_xticks([])
            ax3.grid(grid)
            ax3.set_title(f"Período de {periodo.title()}", fontsize=14)

            # Adicionar totais acima das barras 'top', 'bottom', 'center', 'baseline', 'center_baseline'
            for i, v in enumerate(bar_heights_1):
                ax3.annotate(str(v), xy=(v + 0.2, bar_pos_1[i] + bar_width/2), xytext=(
                    5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_2):
                ax3.annotate(str(v), xy=(v + 0.2, bar_pos_2[i] + bar_width/2), xytext=(
                    5, 0), textcoords="offset points", color='white', ha='left', va='top')

            for i, v in enumerate(bar_heights_3):
                ax3.annotate(str(v), xy=(v + 0.2, bar_pos_3[i] + bar_width/2), xytext=(
                    5, 0), textcoords="offset points", color='white', ha='left', va='top')

            plt.suptitle(suptitle, fontsize=25)
            plt.legend(["Realizados s/ Atraso",
                       "Realizados c/ Atraso", "Cancelados"], loc='best')
            plt.show()

    @staticmethod
    def periodo_ferias_tipo_linha(dataframe: pd.DataFrame, periodo_ferias: str, grid: bool, context: str, figsize, suptitle: str) -> Any:
        with plt.style.context(context):
            fig = plt.figure(figsize=figsize)

            ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)

            bar_values = dataframe['rota']
            lin_heights_1 = dataframe['voos'].values[0]
            bar_heights_1 = dataframe['realizados_s_atraso']
            bar_heights_2 = dataframe['realizados_c_atraso']
            bar_heights_3 = dataframe['cancelados']
            bar_width = 0.2

            bar_pos = np.arange(len(bar_values))
            bars1 = ax1.bar(x=bar_pos - bar_width, height=bar_heights_1,
                            width=bar_width, label='Realizados S/Atraso')
            bars2 = ax1.bar(x=bar_pos, height=bar_heights_2,
                            width=bar_width, label='Realizados C/Atraso')
            bars3 = ax1.bar(x=bar_pos + bar_width, height=bar_heights_3,
                            width=bar_width, label='Cancelados')
            ax1.set_xticks(bar_pos)
            ax1.set_xticklabels(bar_values, rotation=30, ha='right')
            ax1.set_yticklabels([])
            ax1.grid(grid)

            ax2 = ax1.twinx()
            line_heights = dataframe['voos']
            ax2.plot(bar_pos, line_heights, color='red',
                     linestyle=':', label='Totais')
            ax2.set_ylabel('Voos')
            ax2.set_yticklabels([])
            ax2.grid(grid)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')

            for bar1, bar2, bar3, height1, height2, height3 in zip(bars1, bars2, bars3, bar_heights_1, bar_heights_2, bar_heights_3):
                ax1.annotate(f'{height1/lin_heights_1:.1%}', xy=(bar1.get_x() + bar1.get_width() / 2, height1),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='baseline')
                ax1.annotate(f'{height2/lin_heights_1:.1%}', xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='baseline')
                ax1.annotate(f'{height3/lin_heights_1:.1%}', xy=(bar3.get_x() + bar3.get_width() / 2, height3),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='baseline')

            mplcyberpunk.make_lines_glow(ax=ax2)

            plt.suptitle(suptitle, fontsize=25)
            plt.tight_layout()
            plt.show()

    @staticmethod
    def atrasos_periodo_ferias(dataframe: pd.DataFrame, periodo: str, linha, grid: bool, context: str, figsize):
        with plt.style.context(context):
            fig = plt.figure(figsize=figsize)

            ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3)

            # dados ax1
            bar_values = dataframe['justificativa_atraso']
            bar_heights = dataframe['total_atrasos']
            bar_width = 0.35

            # plot ax1
            ax1.barh(y=bar_values, width=bar_heights, height=bar_width)
            ax1.set_yticks(np.arange(len(bar_values)))
            ax1.set_yticklabels(bar_values)
            ax1.set_xticklabels([])
            ax1.grid(grid)
            ax1.set_title(f"{periodo.title()} | {linha.title()}", fontsize=20)

            # values ax1
            for i, v in enumerate(bar_heights):
                ax1.text(v, i, str(v), va='center')

            plt.tight_layout()
            plt.show()

    @staticmethod
    def cancelamentos_periodo_ferias(dataframe: pd.DataFrame, periodo: str, linha, grid: bool, context: str, figsize):
        with plt.style.context(context):
            fig = plt.figure(figsize=figsize)

            ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3)

            # dados ax1
            bar_values = dataframe['justificativa_cancelamento']
            bar_heights = dataframe['total_cancelamentos']
            bar_width = 0.35

            # plot ax1
            ax1.barh(y=bar_values, width=bar_heights, height=bar_width)
            ax1.set_yticks(np.arange(len(bar_values)))
            ax1.set_yticklabels(bar_values)
            ax1.set_xticklabels([])
            ax1.grid(grid)
            ax1.set_title(f"{periodo.title()} | {linha.title()}", fontsize=20)

            # values ax1
            for i, v in enumerate(bar_heights):
                ax1.text(v, i, str(v), va='center')

            plt.tight_layout()
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
    def periodo_ferias(cls) -> List[str]:
        """
        Retorna uma lista dos períodos de férias presentes nos dados.

        Retorna:
        - List[str]: Lista dos períodos de férias.

        Exemplo de uso:
        periodos = AnacVoos.periodo_ferias
        print(periodos)

        """

        if cls.dados_solidos:
            periodos = cls.dados[cls.dados['periodo_ferias'] != '']
            return periodos['periodo_ferias'].unique().tolist()
        return []

    @classmethod
    def tipo_linha(cls) -> List[str]:
        """
        Retorna uma lista dos períodos de férias presentes nos dados.

        Retorna:
        - List[str]: Lista dos tipos de linha.

        Exemplo de uso:
        periodos = AnacVoos.tipo_linha
        print(periodos)

        """

        if cls.dados_solidos:
            periodos = cls.dados[cls.dados['codigo_tipo_linha'] != '']
            return periodos['codigo_tipo_linha'].unique().tolist()
        return []

    @classmethod
    def __get_voos_ferias_regras(cls, dataframe: pd.DataFrame, cols_groupby: List[str], aggregation_columns_added: Dict[str, Tuple[str, Callable]]) -> pd.DataFrame:
        """
        Aplica regras de agregação ao DataFrame dos voos de férias.

        Parâmetros:
        - dataframe (pd.DataFrame): DataFrame dos voos de férias.
        - cols_groupby (List[str]): Lista das colunas de agrupamento.
        - aggregation_columns (Dict[str, Tuple[str, Callable]]): Dicionário das colunas de agregação e suas funções de agregação.

        Retorna:
        - pd.DataFrame: DataFrame com as estatísticas agregadas dos voos de férias.

        """
        aggregation_columns = {
            'realizados_s_atraso': ('situacao_voo', lambda x: (x == 'Realizado Sem Atraso').sum()),
            'realizados_c_atraso': ('situacao_voo', lambda x: (x == 'Realizado Com Atraso').sum()),
            'cancelados': ('situacao_voo', lambda x: (x == 'Cancelado').sum())
        }

        aggregation_columns.update(aggregation_columns_added)

        return dataframe.groupby(cols_groupby).agg(**aggregation_columns).reset_index()

    @classmethod
    def get_voos_ferias(cls) -> pd.DataFrame:
        """
        Retorna um DataFrame contendo os voos durante o período de férias.

        Raises:
            ValueError: Se os dados não foram carregados ou foram comprometidos na transformação.

        Returns:
            pd.DataFrame: DataFrame contendo os voos durante o período de férias.
        """
        if not cls.dados_solidos:
            raise ValueError(
                "Os dados não foram carregados ou foram comprometidos na transformação.")

        return cls.dados[cls.dados['periodo_ferias'] != '']

    @classmethod
    def get_voos_ferias_geral(cls, filtrar_periodo_ferias: bool, percentuais: List[List[str]], round: int, cols_groupby: List[str]) -> pd.DataFrame:
        """
        Retorna um DataFrame com informações gerais sobre os voos durante o período de férias.

        Args:
            filtrar_periodo_ferias (bool): Indica se os dados devem ser filtrados apenas para o período de férias.
            percentuais: Lista de tuplas contendo as colunas que devem ser calculadas como percentuais.
            round (int): Número de casas decimais para arredondar os percentuais.
            cols_groupby (List[str]): Lista de colunas utilizadas para agrupar os dados.

        Raises:
            ValueError: Se os dados não foram carregados ou foram comprometidos na transformação.

        Returns:
            pd.DataFrame: DataFrame com informações gerais sobre os voos durante o período de férias.
        """
        if not cls.dados_solidos:
            raise ValueError(
                "Os dados não foram carregados ou foram comprometidos na transformação.")

        add_aggregation = {
            'voos': ('distancia_km', 'size')
        }

        if filtrar_periodo_ferias:
            df = cls.__get_voos_ferias_regras(
                cls.get_voos_ferias(), cols_groupby, add_aggregation)
        else:
            df = cls.__get_voos_ferias_regras(
                cls.dados, cols_groupby, add_aggregation)

        if len(percentuais) > 0:
            for cols in percentuais:
                df[cols[0]] = (df[cols[1]] / df['voos']).round(round)

        return df

    @classmethod
    def get_voos_ferias_linha_periodo(cls, cols_groupyby: List[str], cols_percentuais: List[List[str]], round: int, ranking: int = 10, linha: str = None, periodo: str = None) -> pd.DataFrame:
        """
        Obtém os voos de férias de um determinado tipo de linha com base nos filtros especificados.

        Parâmetros:
        - cols_groupyby (List[str]): Lista das colunas a serem agrupadas.
        - cols_percentuais (List[List[str]]): Lista de pares de colunas para cálculo de percentuais.
        - round (int): Número de casas decimais para arredondamento.
        - ranking (int): Número máximo de registros no resultado.
        - linha (str): Tipo de linha para filtrar os voos de férias (opcional).
        - periodo (str): Período de férias para filtrar os voos de férias (opcional).

        Retorna:
        - pd.DataFrame: DataFrame contendo os voos de férias filtrados e agregados, classificados pelo número de voos em ordem decrescente.

        Exemplo de uso:
        nacionais = AnacVoos.get_voos_ferias_tipo_linha(cols_groupyby=['codigo_tipo_linha', 'periodo_ferias', 'rota'], cols_percentuais=[['tx_realizados', 'realizados_s_atraso'], ['tx_atrasos', 'realizados_c_atraso'], ['tx_cancelados', 'cancelados']], round=2, ranking=10, linha='Nacional', periodo='2022-01')
        print(nacionais)
        """
        if not cls.dados_solidos:
            raise ValueError("Os dados não foram carregados ou foram comprometidos na transformação.")

        dataframe: pd.DataFrame = cls.dados
        if linha is not None:
            dataframe = dataframe.query("codigo_tipo_linha == @linha")
        if periodo is not None:
            dataframe = dataframe.query("periodo_ferias == @periodo")

        add_aggregation = {
            'distancia_media_km': ('distancia_km', 'mean'),
            'voos': ('distancia_km', 'size')
        }

        dataframe_agg = cls.__get_voos_ferias_regras(dataframe=dataframe, cols_groupby=cols_groupyby, aggregation_columns_added=add_aggregation)

        for cols in cols_percentuais:
            dataframe_agg[cols[0]] = (dataframe_agg[cols[1]] / dataframe_agg['voos']).round(round)

        return dataframe_agg.nlargest(ranking, 'voos')

    @classmethod
    def get_atrasos_voos_ferias(cls, cols_groupby, converter_segundos_para_tempo, filtro_periodo_ferias=None, filtro_codigo_tipo_linha=None) -> pd.DataFrame:
        if not cls.dados_solidos:
            raise ValueError(
                "Os dados não foram carregados ou foram comprometidos na transformação.")

        atrasos = cls.get_voos_ferias().query(
            "partida_atrasou == 'S' or chegada_atrasou == 'S' or situacao_voo == 'Realizado Com Atraso'")

        if filtro_periodo_ferias is not None:
            atrasos = atrasos.query("periodo_ferias == @filtro_periodo_ferias")

        if filtro_codigo_tipo_linha is not None:
            atrasos = atrasos.query(
                "codigo_tipo_linha == @filtro_codigo_tipo_linha")

        atrasos.loc[atrasos['tempo_atraso_partida'].notnull(), 'tempo_atraso_partida'] = atrasos.loc[atrasos['tempo_atraso_partida'].notnull(
        ), 'tempo_atraso_partida'].apply(Utils.converter_tempo_para_segundos)
        atrasos.loc[atrasos['tempo_atraso_chegada'].notnull(), 'tempo_atraso_chegada'] = atrasos.loc[atrasos['tempo_atraso_chegada'].notnull(
        ), 'tempo_atraso_chegada'].apply(Utils.converter_tempo_para_segundos)

        atrasos = atrasos.groupby(cols_groupby).agg(
            total_atrasos=(cols_groupby[0], 'size'),
            media_atrasos_partida=('tempo_atraso_partida', 'mean'),
            media_atrasos_chegada=('tempo_atraso_chegada', 'mean')
        ).reset_index()

        if converter_segundos_para_tempo:
            atrasos.loc[:, 'media_atrasos_partida'] = atrasos.loc[:,
                                                                  'media_atrasos_partida'].apply(Utils.converter_segundos_para_tempo)
            atrasos.loc[:, 'media_atrasos_chegada'] = atrasos.loc[:,
                                                                  'media_atrasos_chegada'].apply(Utils.converter_segundos_para_tempo)

        return atrasos.sort_values(by=['periodo_ferias', 'total_atrasos'], ascending=False)

    @classmethod
    def get_cancelamentos_voos_ferias(cls, cols_groupby, filtro_periodo_ferias=None, filtro_codigo_tipo_linha=None) -> pd.DataFrame:
        if not cls.dados_solidos:
            raise ValueError(
                "Os dados não foram carregados ou foram comprometidos na transformação.")

        atrasos = cls.get_voos_ferias().query(
            "situacao_voo == 'Cancelado'")

        if filtro_periodo_ferias is not None:
            atrasos = atrasos.query("periodo_ferias == @filtro_periodo_ferias")

        if filtro_codigo_tipo_linha is not None:
            atrasos = atrasos.query(
                "codigo_tipo_linha == @filtro_codigo_tipo_linha")

        atrasos = atrasos.groupby(cols_groupby).agg(
            total_cancelamentos=(cols_groupby[0], 'size'),
        ).reset_index()

        return atrasos.sort_values(by=['periodo_ferias', 'total_cancelamentos'], ascending=False)

    @classmethod
    def get_voos_ferias_resumo(cls, txs_columns, groupby_columns, round):
        if not cls.dados_solidos:
            raise ValueError(
                "Os dados não foram carregados ou foram comprometidos na transformação.")

        aggregation_columns = {
            'voos': ('situacao_voo', 'size'),
            'realizados_s_atraso': ('situacao_voo', lambda x: (x == 'Realizado Sem Atraso').sum()),
            'realizados_c_atraso': ('situacao_voo', lambda x: (x == 'Realizado Com Atraso').sum()),
            'cancelados': ('situacao_voo', lambda x: (x == 'Cancelado').sum())
        }

        dataframe = cls.get_voos_ferias().groupby(
            groupby_columns).agg(**aggregation_columns).reset_index()

        for cols in txs_columns:
            dataframe[cols[0]] = (
                dataframe[cols[1]] / dataframe['voos']).round(round)

        return dataframe
