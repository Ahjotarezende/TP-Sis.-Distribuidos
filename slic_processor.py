import numpy as np
from skimage import segmentation, color
import numpy as np
from PIL import Image


def aplicar_slic(
    imagem_array: np.ndarray,
    n_segmentos: int,
    compactness: float
) -> np.ndarray:

    segmentos = segmentation.slic(
        imagem_array,
        n_segments=n_segmentos,
        compactness=compactness,
        start_label=1
    )

    imagem_processada = color.label2rgb(
        segmentos,
        imagem_array,
        kind="avg"
    )

    return np.clip(imagem_processada, 0, 255).astype(np.uint8)


def bytes_para_array(imagem_bytes: bytes, largura: int, altura: int) -> np.ndarray:
    imagem = Image.frombytes("RGB", (largura, altura), imagem_bytes)
    return np.array(imagem)


def array_para_bytes(imagem_array: np.ndarray) -> bytes:
    imagem = Image.fromarray(imagem_array.astype(np.uint8), "RGB")
    return imagem.tobytes()