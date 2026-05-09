import os
import logging
import numpy as np
from skimage import io, segmentation, color
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# CONFIGURAÇÕES GLOBAIS
# ==========================================
# Defina o número máximo de segmentos desejados para o algoritmo SLIC
MAXIMO_SEGMENTOS = 600000
# ==========================================

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ProcessadorImagem:
    """
    Classe que atua como o 'Worker' de processamento local.
    Contém a lógica de processamento isolada para facilitar futura migração para gRPC.
    """

    @staticmethod
    def processar_bloco(argumentos: tuple) -> tuple:
        """
        Aplica o algoritmo SLIC em um bloco específico da imagem.
        
        Args:
            argumentos (tuple): Tupla contendo (id_bloco, array_imagem, n_segmentos, compacidade)
            
        Returns:
            tuple: (id_bloco, matriz_processada)
        """
        id_bloco, imagem_array, n_segmentos, compacidade = argumentos
        
        try:
            logging.info(f"Worker processando bloco {id_bloco} com alvo de {n_segmentos} segmentos.")
            
            segmentos_slic = segmentation.slic(
                imagem_array, 
                n_segments=n_segmentos, 
                compactness=compacidade, 
                start_label=1
            )
            
            imagem_processada = color.label2rgb(segmentos_slic, imagem_array, kind='avg')
            
            # Correção: Garante que os valores fiquem no limite seguro de 8 bits (0 a 255) antes da conversão
            imagem_final_uint8 = np.clip(imagem_processada, 0, 255).astype(np.uint8)
            
            return id_bloco, imagem_final_uint8
            
        except Exception as e:
            logging.error(f"Erro no processamento do bloco {id_bloco}: {str(e)}")
            raise


class OrquestradorProcessamento:
    """
    Classe responsável por gerenciar a divisão da carga de trabalho,
    alocação de núcleos da CPU e reconstrução da imagem.
    """

    def __init__(self, segmentos_totais: int):
        self.segmentos_totais = segmentos_totais
        
        # Identifica núcleos lógicos e define o uso pela metade
        total_nucleos = os.cpu_count() or 2
        self.workers = max(1, total_nucleos // 2)
        logging.info(f"Hardware detectado: {total_nucleos} núcleos lógicos. Alocando {self.workers} workers.")

    def _fatiar_imagem(self, imagem: np.ndarray) -> list:
        """Divide a imagem horizontalmente com base no número de workers."""
        return np.array_split(imagem, self.workers, axis=0)

    def _juntar_fatias(self, fatias: list) -> np.ndarray:
        """Reconstrói a imagem empilhando as fatias verticalmente."""
        return np.vstack(fatias)

    def executar_paralelo(self, imagem_original: np.ndarray) -> np.ndarray:
        """
        Orquestra o fluxo de divisão, envio para os workers e reconstrução.
        """
        fatias = self._fatiar_imagem(imagem_original)
        
        # Divide a quantidade de segmentos proporcionalmente para cada bloco
        segmentos_por_fatia = max(1, self.segmentos_totais // self.workers)
        compacidade = 10.0

        # Prepara os pacotes de dados para envio aos processos
        argumentos_workers = [
            (i, fatia, segmentos_por_fatia, compacidade) 
            for i, fatia in enumerate(fatias)
        ]

        fatias_processadas = [None] * self.workers

        # Inicia o pool de processos paralelos
        logging.info("Iniciando processamento paralelo em múltiplos núcleos...")
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # O método map garante a execução paralela e coleta os retornos
            resultados = executor.map(ProcessadorImagem.processar_bloco, argumentos_workers)

            # Posiciona cada resultado em seu respectivo índice para garantir a ordem correta
            for id_bloco, resultado_fatia in resultados:
                fatias_processadas[id_bloco] = resultado_fatia

        logging.info("Todos os blocos processados. Reconstruindo imagem final.")
        return self._juntar_fatias(fatias_processadas)


def processar_arquivo(caminho_entrada: str, caminho_saida: str) -> None:
    """Função principal de entrada/saída de dados."""
    
    if not os.path.exists(caminho_entrada):
        logging.error(f"Arquivo não encontrado: {caminho_entrada}")
        return

    logging.info(f"Carregando imagem: {caminho_entrada}")
    try:
        imagem_original = io.imread(caminho_entrada)
        if imagem_original is None or imagem_original.size == 0:
            raise ValueError("O arquivo lido não contém dados de imagem válidos.")
    except Exception as e:
        logging.error(f"Falha ao ler o arquivo de imagem: {str(e)}")
        return

    try:
        # Instancia o orquestrador utilizando a variável global
        orquestrador = OrquestradorProcessamento(segmentos_totais=MAXIMO_SEGMENTOS)
        imagem_resultado = orquestrador.executar_paralelo(imagem_original)
    except Exception as e:
        logging.error(f"O processamento falhou: {str(e)}")
        return

    logging.info(f"Salvando resultado em: {caminho_saida}")
    try:
        io.imsave(caminho_saida, imagem_resultado)
        logging.info("Arquivo salvo com sucesso.")
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo de saída: {str(e)}")


if __name__ == "__main__":
    DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
    IMAGEM_ENTRADA = os.path.join(DIRETORIO_ATUAL, "HHpCVrzWkAQTAm6.jpg")
    IMAGEM_SAIDA = os.path.join(DIRETORIO_ATUAL, "imagem_processada_paralela.jpg")

    print("--- Pipeline de Processamento Paralelo ---")
    processar_arquivo(IMAGEM_ENTRADA, IMAGEM_SAIDA)