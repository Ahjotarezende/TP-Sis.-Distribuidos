import grpc
import numpy as np
from PIL import Image
import time
import threading

import segmentacao_pb2
import segmentacao_pb2_grpc
from lamport_clock import LamportClock

WORKERS = [
    "localhost:50051",
    "localhost:50052",
    "localhost:50053",
]

CAMINHO_IMAGEM = "teste.jpg"
CAMINHO_SAIDA = "resultado_segmentado.jpg"

MAXIMO_SEGMENTOS = 300
COMPACTNESS = 10.0

PEER_TIMEOUT = 5
ELECTION_WAIT = 20.0

_lock = threading.Lock()
_leader_address: str | None = None
_leader_id: int | None = None


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


def _aguardar_workers(workers, tentativas=10, intervalo=2.0):
    online = []
    for w in workers:
        for _ in range(tentativas):
            try:
                ch = _make_channel(w)
                stub = segmentacao_pb2_grpc.SegmentacaoServiceStub(ch)
                stub.Status(segmentacao_pb2.StatusRequest(), timeout=PEER_TIMEOUT)
                ch.close()
                online.append(w)
                print(f"[Init] {w} online")
                break
            except grpc.RpcError:
                time.sleep(intervalo)
        else:
            print(f"[Init] {w} ignorado")
    return online


def _disparar_eleicao(workers_online):
    global _leader_address, _leader_id

    print("\n[Eleição] iniciando...")

    for w in workers_online:
        try:
            ch = _make_channel(w)
            stub = segmentacao_pb2_grpc.SegmentacaoServiceStub(ch)
            stub.Election(segmentacao_pb2.ElectionRequest(sender_id=0), timeout=PEER_TIMEOUT)
            ch.close()
        except grpc.RpcError:
            pass

    deadline = time.time() + ELECTION_WAIT
    while time.time() < deadline:
        with _lock:
            if _leader_address:
                return _leader_address
        time.sleep(0.5)

    if workers_online:
        lider = max(workers_online, key=_get_id)
        with _lock:
            _leader_address = lider
            _leader_id = _get_id(lider)
        return lider

    return None


def _re_eleger(workers_online):
    global _leader_address, _leader_id

    with _lock:
        lider = _leader_address

    if not lider:
        return _disparar_eleicao(workers_online)

    try:
        ch = _make_channel(lider)
        stub = segmentacao_pb2_grpc.SegmentacaoServiceStub(ch)
        stub.Status(segmentacao_pb2.StatusRequest(), timeout=PEER_TIMEOUT)
        ch.close()
        return lider
    except grpc.RpcError:
        with _lock:
            _leader_address = None
            _leader_id = None

        return _disparar_eleicao([w for w in workers_online if w != lider])


def dividir_imagem_em_blocos(img, n):
    h = img.shape[0]
    step = h // n
    blocos = []

    for i in range(n):
        ini = i * step
        fim = h if i == n - 1 else (i + 1) * step
        blocos.append((i, ini, fim, img[ini:fim]))

    return blocos


def _tentar_worker(addr, id_bloco, bloco, clock, n_seg, comp, max_tent=3):
    for _ in range(max_tent):
        canal = None
        try:
            canal = _make_channel(addr)
            stub = segmentacao_pb2_grpc.SegmentacaoServiceStub(canal)

            h, w, _ = bloco.shape

            req = segmentacao_pb2.BlocoImagemRequest(
                id_bloco=id_bloco,
                largura=w,
                altura=h,
                imagem=bloco.tobytes(),
                timestamp=clock.get_time(),
                n_segmentos=n_seg,
                compactness=comp,
            )

            resp = stub.ProcessarBloco(req, timeout=30)
            clock.update(resp.timestamp)

            img = Image.frombytes(
                "RGB",
                (resp.largura, resp.altura),
                resp.imagem_segmentada,
            )

            return np.array(img)

        except grpc.RpcError:
            time.sleep(2)

        finally:
            if canal:
                canal.close()

    return None


def enviar_bloco_com_failover(
    id_bloco,
    bloco,
    clock,
    workers,
    preferido,
    n_seg,
    comp,
):
    with _lock:
        lider = _leader_address

    ordem = [preferido]

    if lider and lider != preferido:
        ordem.append(lider)

    ordem += [w for w in workers if w not in ordem]

    for w in ordem:
        clock.increment()

        res = _tentar_worker(w, id_bloco, bloco, clock, n_seg, comp)

        if res is not None:
            return res

        if w == lider:
            threading.Thread(
                target=_re_eleger,
                args=(workers,),
                daemon=True,
            ).start()

    return None


def processar_imagem_distribuida(
    imagem_array,
    workers=None,
    max_segmentos=MAXIMO_SEGMENTOS,
    compactness=COMPACTNESS,
    progresso_callback=None,
):
    inicio = time.time()

    if workers is None:
        workers = WORKERS

    workers_online = _aguardar_workers(workers)
    if not workers_online:
        return None, 0

    _disparar_eleicao(workers_online)

    clock = LamportClock()

    blocos = dividir_imagem_em_blocos(imagem_array, len(workers_online))
    seg_por_bloco = max(1, max_segmentos // len(workers_online))

    resultados = []

    for i, (worker, (idb, ini, fim, bloco)) in enumerate(zip(workers_online, blocos)):
        res = enviar_bloco_com_failover(
            idb,
            bloco,
            clock,
            workers_online,
            worker,
            seg_por_bloco,
            compactness,
        )

        if res is None:
            return None, time.time() - inicio

        resultados.append((ini, fim, res))

        if progresso_callback:
            progresso_callback(i + 1, len(workers_online))

    resultados.sort(key=lambda x: x[0])
    final = np.vstack([b for _, _, b in resultados])

    return final, time.time() - inicio


def main():
    img = Image.open(CAMINHO_IMAGEM).convert("RGB")
    arr = np.array(img)

    final, tempo = processar_imagem_distribuida(arr)

    if final is None:
        print("Falha geral")
        return

    Image.fromarray(final).save(CAMINHO_SAIDA)

    print(f"OK -> {CAMINHO_SAIDA}")
    print(f"Tempo: {tempo:.2f}s")


if __name__ == "__main__":
    main()