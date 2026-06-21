"""
Web Service - Gateway HTTP para o sistema de segmentação distribuída.

Este serviço NÃO substitui o gRPC entre coordinator <-> workers.
Ele atua como uma "porta de entrada" externa: recebe imagens via HTTP,
delega o processamento para o pipeline distribuído já existente
(coordinator_client.processar_imagem_distribuida) e permite acompanhar
o status/baixar o resultado depois.

Como rodar:
    pip install fastapi uvicorn python-multipart
    uvicorn web_service:app --reload --port 8000

Depois, acesse http://localhost:8000/docs para testar pela interface
automática do FastAPI (ou use curl/Postman).
"""

import io
import uuid
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from coordinator_client import (
    processar_imagem_distribuida,
    WORKERS,
    MAXIMO_SEGMENTOS,
    COMPACTNESS,
)

app = FastAPI(
    title="Segmentação Distribuída - Web Service",
    description="Gateway HTTP para o pipeline de segmentação SLIC distribuído via gRPC",
)

# Diretório onde os resultados ficam salvos, para download posterior
DIRETORIO_RESULTADOS = Path("resultados_web")
DIRETORIO_RESULTADOS.mkdir(exist_ok=True)

# "Banco de dados" de jobs em memória.
# Em produção isso seria um banco real ou Redis, mas para o trabalho
# prático isso é suficiente e simples de explicar no relatório.
# Estrutura de cada job:
# {
#   "status": "pendente" | "processando" | "concluido" | "erro",
#   "blocos_concluidos": int,
#   "blocos_totais": int,
#   "tempo_total": float | None,
#   "erro": str | None,
#   "caminho_resultado": str | None,
# }
jobs = {}
jobs_lock = threading.Lock()


class StatusJobResponse(BaseModel):
    job_id: str
    status: str
    blocos_concluidos: int
    blocos_totais: int
    tempo_total: float | None = None
    erro: str | None = None


def _atualizar_job(job_id: str, **campos):
    with jobs_lock:
        jobs[job_id].update(campos)


def _processar_em_background(job_id: str, imagem_array: np.ndarray):
    """
    Executa o pipeline distribuído numa thread separada, para que a
    requisição HTTP de upload não fique bloqueada esperando o
    processamento inteiro terminar (que pode levar vários segundos).
    """

    def callback_progresso(concluidos, total):
        _atualizar_job(
            job_id,
            blocos_concluidos=concluidos,
            blocos_totais=total,
        )

    try:
        _atualizar_job(job_id, status="processando")

        imagem_final, tempo_total = processar_imagem_distribuida(
            imagem_array,
            workers=WORKERS,
            max_segmentos=MAXIMO_SEGMENTOS,
            compactness=COMPACTNESS,
            progresso_callback=callback_progresso,
        )

        if imagem_final is None:
            _atualizar_job(
                job_id,
                status="erro",
                erro="Nenhum worker conseguiu processar a imagem.",
                tempo_total=tempo_total,
            )
            return

        caminho_saida = DIRETORIO_RESULTADOS / f"{job_id}.jpg"
        Image.fromarray(imagem_final).save(caminho_saida)

        _atualizar_job(
            job_id,
            status="concluido",
            tempo_total=tempo_total,
            caminho_resultado=str(caminho_saida),
        )

    except Exception as e:
        _atualizar_job(job_id, status="erro", erro=str(e))


@app.post("/processar", response_model=StatusJobResponse)
async def processar_imagem(arquivo: UploadFile = File(...)):
    """
    Recebe uma imagem, dispara o processamento distribuído em segundo
    plano e devolve imediatamente um job_id para acompanhamento.
    """

    if not arquivo.content_type or not arquivo.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Arquivo enviado não parece ser uma imagem.",
        )

    conteudo = await arquivo.read()

    try:
        imagem = Image.open(io.BytesIO(conteudo)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler a imagem enviada.",
        )

    imagem_array = np.array(imagem)
    job_id = str(uuid.uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "status": "pendente",
            "blocos_concluidos": 0,
            "blocos_totais": len(WORKERS),
            "tempo_total": None,
            "erro": None,
            "caminho_resultado": None,
        }

    # Processamento em thread separada: a chamada HTTP retorna logo,
    # e o cliente consulta /status/{job_id} para saber quando terminou.
    thread = threading.Thread(
        target=_processar_em_background,
        args=(job_id, imagem_array),
        daemon=True,
    )
    thread.start()

    return StatusJobResponse(
        job_id=job_id,
        status="pendente",
        blocos_concluidos=0,
        blocos_totais=len(WORKERS),
    )


@app.get("/status/{job_id}", response_model=StatusJobResponse)
def consultar_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="job_id não encontrado.")

    return StatusJobResponse(job_id=job_id, **{
        k: v for k, v in job.items() if k != "caminho_resultado"
    })


@app.get("/resultado/{job_id}")
def baixar_resultado(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="job_id não encontrado.")

    if job["status"] != "concluido":
        raise HTTPException(
            status_code=409,
            detail=f"Job ainda não concluído (status atual: {job['status']}).",
        )

    return FileResponse(
        path=job["caminho_resultado"],
        media_type="image/jpeg",
        filename=f"resultado_{job_id}.jpg",
    )


@app.get("/")
def raiz():
    return {
        "servico": "Segmentação Distribuída - Web Service",
        "endpoints": {
            "POST /processar": "Envia uma imagem (multipart/form-data, campo 'arquivo')",
            "GET /status/{job_id}": "Consulta o andamento do processamento",
            "GET /resultado/{job_id}": "Baixa a imagem segmentada final",
        },
        "docs": "/docs",
    }
