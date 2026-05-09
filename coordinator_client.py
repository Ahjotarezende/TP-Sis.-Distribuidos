import grpc
import numpy as np
from PIL import Image
import time

import segmentacao_pb2
import segmentacao_pb2_grpc


WORKERS = [
    "192.168.0.15:50051",
    "192.168.0.22:50051"
]

CAMINHO_IMAGEM = "imagem_teste.jpg"
CAMINHO_SAIDA = "resultado_segmentado.jpg"


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


def enviar_bloco_para_worker(worker_address, id_bloco, bloco):
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
        imagem=Image.fromarray(bloco.astype(np.uint8), "RGB").tobytes()
    )

    response = stub.ProcessarBloco(request)

    bloco_segmentado = Image.frombytes(
        "RGB",
        (response.largura, response.altura),
        response.imagem_segmentada
    )

    return np.array(bloco_segmentado)


def main():
    inicio_tempo = time.time()

    imagem = Image.open(CAMINHO_IMAGEM).convert("RGB")
    imagem_array = np.array(imagem)

    quantidade_blocos = len(WORKERS)
    blocos = dividir_imagem_em_blocos(imagem_array, quantidade_blocos)

    resultados = []

    for worker, (id_bloco, inicio, fim, bloco) in zip(WORKERS, blocos):
        print(f"Enviando bloco {id_bloco} para {worker}")

        bloco_segmentado = enviar_bloco_para_worker(
            worker,
            id_bloco,
            bloco
        )

        resultados.append((inicio, fim, bloco_segmentado))

    resultados.sort(key=lambda x: x[0])

    imagem_final = np.vstack([bloco for _, _, bloco in resultados])

    Image.fromarray(imagem_final).save(CAMINHO_SAIDA)

    fim_tempo = time.time()

    print(f"Imagem segmentada salva em: {CAMINHO_SAIDA}")
    print(f"Tempo total: {fim_tempo - inicio_tempo:.2f} segundos")


if __name__ == "__main__":
    main()