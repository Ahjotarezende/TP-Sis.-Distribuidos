import grpc
from concurrent import futures
import socket
import sys

import segmentacao_pb2
import segmentacao_pb2_grpc

from slic_processor import aplicar_slic, bytes_para_array, array_para_bytes

from lamport_clock import LamportClock


class SegmentacaoService(segmentacao_pb2_grpc.SegmentacaoServiceServicer):

    def __init__(self):
        self.clock = LamportClock()

    def Status(self, request, context):
        return segmentacao_pb2.StatusResponse(
            status="online",
            nome_maquina=socket.gethostname()
        )

    def ProcessarBloco(self, request, context):
        self.clock.update(request.timestamp)

        print(
            f"[t={self.clock.get_time()}] "
            f"Recebi o bloco {request.id_bloco}"
        )

        bloco_array = bytes_para_array(
            request.imagem,
            request.largura,
            request.altura
        )

        bloco_segmentado = aplicar_slic(bloco_array,request.n_segmentos,request.compactness)
        self.clock.increment()

        bloco_segmentado_bytes = array_para_bytes(bloco_segmentado)

        return segmentacao_pb2.BlocoImagemResponse(
            id_bloco=request.id_bloco,
            largura=request.largura,
            altura=request.altura,
            imagem_segmentada=bloco_segmentado_bytes,
            timestamp=self.clock.get_time()
        )


def iniciar_servidor(porta):
    servidor = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
    )

    segmentacao_pb2_grpc.add_SegmentacaoServiceServicer_to_server(
        SegmentacaoService(),
        servidor
    )

    servidor.add_insecure_port(f"[::]:{porta}")
    servidor.start()

    print(f"Worker gRPC rodando na porta {porta}...")
    servidor.wait_for_termination()


if __name__ == "__main__":
    # Uso: python worker_server.py [porta]
    # Se nao passar nada, usa 50051 como padrao
    porta = sys.argv[1] if len(sys.argv) > 1 else "50051"
    iniciar_servidor(porta)