import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from PIL import Image


def aplicar_slic(imagem_array: np.ndarray) -> np.ndarray:
    """
    Recebe um bloco RGB em formato NumPy e devolve o bloco segmentado.
    """

    imagem_float = img_as_float(imagem_array)

    segmentos = slic(
        imagem_float,
        n_segments=100,
        compactness=10,
        sigma=1,
        start_label=1,
        channel_axis=-1
    )

    imagem_com_bordas = mark_boundaries(imagem_float, segmentos)

    imagem_saida = (imagem_com_bordas * 255).astype(np.uint8)

    return imagem_saida


def bytes_para_array(imagem_bytes: bytes, largura: int, altura: int) -> np.ndarray:
    imagem = Image.frombytes("RGB", (largura, altura), imagem_bytes)
    return np.array(imagem)


def array_para_bytes(imagem_array: np.ndarray) -> bytes:
    imagem = Image.fromarray(imagem_array.astype(np.uint8), "RGB")
    return imagem.tobytes()