import grpc
import numpy as np
from PIL import Image
import time

import segmentacao_pb2
import segmentacao_pb2_grpc

from lamport_clock import LamportClock

WORKERS = [
    "localhost:50051",
    "localhost:50052"
]

CAMINHO_IMAGEM = "teste.jpg"
CAMINHO_SAIDA = "resultado_segmentado.jpg"

MAXIMO_SEGMENTOS = 1000
COMPACTNESS = 10.0


def dividir_imagem_em_blocos(imagem_array, quantidade_blocos):
    altura = imagem_array.shape[0]
    blocos = []

    linhas_por_bloco = altura // quantidade_blocos

    for i in range(quantidade_blocos):
        inicio = i * linhas_por_bloco

        if i == quantidade_blocos - 1:
            fim = altura
        else:
            fim = (i + 1) * linhas_por_bloco

        bloco = imagem_array[inicio:fim, :, :]
        blocos.append((i, inicio, fim, bloco))

    return blocos


def _tentar_worker(
    worker_address,
    id_bloco,
    bloco,
    clock,
    n_segmentos,
    compactness,
    max_tentativas=3
):
    """Tenta processar o bloco em um worker com retry."""

    for tentativa in range(max_tentativas):

        canal = None

        try:

            print(
                f"Tentativa {tentativa + 1} - "
                f"Enviando bloco {id_bloco} para {worker_address}"
            )

            canal = grpc.insecure_channel(
                worker_address,
                options=[
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ]
            )

            stub = segmentacao_pb2_grpc.SegmentacaoServiceStub(canal)

            altura, largura, _ = bloco.shape

            request = segmentacao_pb2.BlocoImagemRequest(
                id_bloco=id_bloco,
                largura=largura,
                altura=altura,
                imagem=bloco.tobytes(),
                timestamp=clock.get_time(),
                n_segmentos=n_segmentos,
                compactness=compactness
            )

            response = stub.ProcessarBloco(
                request,
                timeout=30
            )

            clock.update(response.timestamp)

            bloco_segmentado = Image.frombytes(
                "RGB",
                (response.largura, response.altura),
                response.imagem_segmentada
            )

            return np.array(bloco_segmentado)

        except grpc.RpcError as e:

            print(
                f"Falha na tentativa {tentativa + 1} "
                f"em {worker_address}: {e.code()}"
            )

            if tentativa < max_tentativas - 1:
                print("Tentando novamente...")
                time.sleep(2)
            else:
                print(f"Worker {worker_address} esgotou as tentativas.")
                return None

        finally:
            if canal:
                canal.close()


def enviar_bloco_com_failover(
    id_bloco,
    bloco,
    clock,
    lista_workers,
    worker_preferido,
    n_segmentos,
    compactness
):

    ordem_workers = [
        worker_preferido
    ] + [
        w for w in lista_workers
        if w != worker_preferido
    ]

    for worker_address in ordem_workers:

        clock.increment()

        resultado = _tentar_worker(
            worker_address,
            id_bloco,
            bloco,
            clock,
            n_segmentos,
            compactness
        )

        if resultado is not None:

            if worker_address != worker_preferido:
                print(
                    f"Failover bem-sucedido: "
                    f"bloco {id_bloco} -> {worker_address}"
                )

            return resultado

        print(
            f"Worker {worker_address} indisponível."
        )

    print(
        f"ERRO CRÍTICO: nenhum worker conseguiu "
        f"processar o bloco {id_bloco}."
    )

    return None


def processar_imagem_distribuida(
    imagem_array,
    workers=None,
    max_segmentos=MAXIMO_SEGMENTOS,
    compactness=COMPACTNESS,
    progresso_callback=None
):
    """
    Núcleo do processamento distribuído, reutilizável tanto pelo
    script de linha de comando (main) quanto pelo web service.

    progresso_callback (opcional): função chamada como
    progresso_callback(blocos_concluidos, blocos_totais) a cada
    bloco finalizado, para permitir acompanhar o andamento (ex: via
    endpoint de status do web service).

    Retorna: (imagem_final_array, tempo_total_segundos) ou (None, tempo)
    em caso de falha total.
    """

    inicio_tempo = time.time()

    if workers is None:
        workers = WORKERS

    clock = LamportClock()

    quantidade_blocos = len(workers)

    blocos = dividir_imagem_em_blocos(
        imagem_array,
        quantidade_blocos
    )

    segmentos_por_bloco = max(
        1,
        max_segmentos // quantidade_blocos
    )

    resultados = []

    for indice, (worker, (id_bloco, inicio, fim, bloco)) in enumerate(
        zip(workers, blocos)
    ):

        bloco_segmentado = enviar_bloco_com_failover(
            id_bloco=id_bloco,
            bloco=bloco,
            clock=clock,
            lista_workers=workers,
            worker_preferido=worker,
            n_segmentos=segmentos_por_bloco,
            compactness=compactness
        )

        if bloco_segmentado is None:
            print("Abortando processamento.")
            return None, time.time() - inicio_tempo

        resultados.append(
            (
                inicio,
                fim,
                bloco_segmentado
            )
        )

        if progresso_callback:
            progresso_callback(indice + 1, quantidade_blocos)

    resultados.sort(key=lambda x: x[0])

    imagem_final = np.vstack(
        [
            bloco
            for _, _, bloco in resultados
        ]
    )

    tempo_total = time.time() - inicio_tempo

    return imagem_final, tempo_total


def main():

    imagem = Image.open(CAMINHO_IMAGEM).convert("RGB")
    imagem_array = np.array(imagem)

    imagem_final, tempo_total = processar_imagem_distribuida(imagem_array)

    if imagem_final is None:
        return

    Image.fromarray(imagem_final).save(CAMINHO_SAIDA)

    print(f"\nImagem salva em: {CAMINHO_SAIDA}")
    print(f"Tempo total: {tempo_total:.2f} segundos")


if __name__ == "__main__":
    main()