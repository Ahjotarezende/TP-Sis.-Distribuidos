import grpc
import socket
import sys
import threading
import time
from concurrent import futures

import segmentacao_pb2
import segmentacao_pb2_grpc

from lamport_clock import LamportClock
from slic_processor import aplicar_slic, array_para_bytes, bytes_para_array

WORKERS = [
    "localhost:50051",
    "localhost:50052",
    "localhost:50053",
]

PEER_TIMEOUT = 5
HEARTBEAT_INTERVAL = 10

_lock = threading.Lock()
_leader_address: str | None = None
_leader_id: int | None = None
_election_in_progress = False

def _get_id(address: str) -> int:
    return int(address.split(":")[1])


def _make_channel(address: str):
    return grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )


def _worker_alive(address: str) -> bool:
    try:
        ch = _make_channel(address)
        stub = segmentacao_pb2_grpc.SegmentacaoServiceStub(ch)
        stub.Status(segmentacao_pb2.StatusRequest(), timeout=PEER_TIMEOUT)
        ch.close()
        return True
    except grpc.RpcError:
        return False

def iniciar_eleicao(minha_porta: int) -> None:
    global _election_in_progress, _leader_address, _leader_id

    with _lock:
        if _election_in_progress:
            return
        _election_in_progress = True

    print(f"[Eleição] Nó :{minha_porta} iniciando eleição (Bully)")

    meu_id = minha_porta
    peers_maiores = [w for w in WORKERS if _get_id(w) > meu_id]

    recebeu_ok = False

    for peer in peers_maiores:
        try:
            ch = _make_channel(peer)
            stub = segmentacao_pb2_grpc.SegmentacaoServiceStub(ch)

            resp = stub.Election(
                segmentacao_pb2.ElectionRequest(sender_id=meu_id),
                timeout=PEER_TIMEOUT,
            )

            ch.close()

            if resp.alive:
                recebeu_ok = True
                break

        except grpc.RpcError:
            pass

    if not recebeu_ok:
        _tornar_lider(minha_porta)
    else:
        _aguardar_coordenador(minha_porta)

    with _lock:
        _election_in_progress = False


def _tornar_lider(minha_porta: int) -> None:
    global _leader_address, _leader_id

    meu_address = f"localhost:{minha_porta}"

    with _lock:
        _leader_address = meu_address
        _leader_id = minha_porta

    print(f"[Eleição] *** Sou o novo líder: {meu_address} ***")

    for peer in WORKERS:
        if peer == meu_address:
            continue

        try:
            ch = _make_channel(peer)
            stub = segmentacao_pb2_grpc.SegmentacaoServiceStub(ch)

            stub.Coordinator(
                segmentacao_pb2.CoordinatorRequest(
                    leader_id=minha_porta,
                    leader_address=meu_address,
                ),
                timeout=PEER_TIMEOUT,
            )

            ch.close()

        except grpc.RpcError:
            pass


def _aguardar_coordenador(minha_porta: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout

    while time.time() < deadline:
        with _lock:
            if _leader_id is not None:
                return
        time.sleep(0.5)

    iniciar_eleicao(minha_porta)


def _heartbeat_loop(minha_porta: int) -> None:
    global _leader_address, _leader_id

    while True:
        time.sleep(HEARTBEAT_INTERVAL)

        with _lock:
            lider = _leader_address
            lider_id = _leader_id

        if lider is None:
            continue

        if lider_id == minha_porta:
            continue

        if not _worker_alive(lider):
            print(f"[Heartbeat] Líder {lider} falhou! Re-eleição...")

            with _lock:
                _leader_address = None
                _leader_id = None

            iniciar_eleicao(minha_porta)

class SegmentacaoService(segmentacao_pb2_grpc.SegmentacaoServiceServicer):

    def __init__(self, minha_porta: int) -> None:
        self.clock = LamportClock()
        self.minha_porta = minha_porta

    def Status(self, request, context):
        return segmentacao_pb2.StatusResponse(
            status="online",
            nome_maquina=socket.gethostname(),
        )

    def ProcessarBloco(self, request, context):
        self.clock.update(request.timestamp)

        print(
            f"[RECEBIDO] Bloco {request.id_bloco} | "
            f"worker={self.minha_porta if hasattr(self, 'minha_porta') else 'unknown'} | "
            f"t={self.clock.get_time()}"
        )

        bloco_array = bytes_para_array(
            request.imagem,
            request.largura,
            request.altura,
        )

        bloco_segmentado = aplicar_slic(
            bloco_array,
            request.n_segmentos,
            request.compactness,
        )

        self.clock.increment()

        return segmentacao_pb2.BlocoImagemResponse(
            id_bloco=request.id_bloco,
            largura=request.largura,
            altura=request.altura,
            imagem_segmentada=array_para_bytes(bloco_segmentado),
            timestamp=self.clock.get_time(),
        )

    def Election(self, request, context):
        threading.Thread(
            target=iniciar_eleicao,
            args=(self.minha_porta,),
            daemon=True,
        ).start()

        return segmentacao_pb2.ElectionResponse(alive=True)

    def Coordinator(self, request, context):
        global _leader_address, _leader_id

        with _lock:
            _leader_address = request.leader_address
            _leader_id = request.leader_id

        print(
            f"[Eleição] Novo líder registrado: "
            f"{request.leader_address} (ID={request.leader_id})"
        )

        return segmentacao_pb2.CoordinatorResponse(acknowledged=True)

def iniciar_servidor(porta: int) -> None:
    servidor = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )

    segmentacao_pb2_grpc.add_SegmentacaoServiceServicer_to_server(
        SegmentacaoService(porta),
        servidor,
    )

    servidor.add_insecure_port(f"[::]:{porta}")
    servidor.start()

    print(f"Worker gRPC rodando na porta {porta}...")

    threading.Thread(
        target=_heartbeat_loop,
        args=(porta,),
        daemon=True,
    ).start()

    servidor.wait_for_termination()


if __name__ == "__main__":
    porta = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    iniciar_servidor(porta)