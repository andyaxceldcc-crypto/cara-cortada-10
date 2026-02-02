# --- INICIO DEL ARCHIVO face_enhancer.py ---

from typing import Any, List
import cv2
import threading
import gfpgan
import os
import platform
import torch # Asegurarse de que torch está importado

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    """
    Comprueba que los modelos necesarios están descargados; si no, los descarga condicionalmente.
    """
    download_directory_path = models_dir
    conditional_download(
        download_directory_path,
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        ],
    )
    return True


def pre_start() -> bool:
    """
    Verifica que la ruta objetivo es una imagen o vídeo antes de iniciar.
    """
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Selecciona una imagen o vídeo para la ruta objetivo.", NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    """
    Inicializa y devuelve la instancia global de GFPGANer,
    priorizando CUDA, luego MPS (Mac) y finalmente CPU.
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
            device = None
            try:
                # Prioridad 1: CUDA
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    print(f"{NAME}: Usando dispositivo CUDA.")
                # Prioridad 2: MPS (Mac Silicon)
                elif platform.system() == "Darwin" and torch.backends.mps.is_available():
                    device = torch.device("mps")
                    print(f"{NAME}: Usando dispositivo MPS.")
                # Prioridad 3: CPU
                else:
                    device = torch.device("cpu")
                    print(f"{NAME}: Usando dispositivo CPU.")

                FACE_ENHANCER = gfpgan.GFPGANer(
                    model_path=model_path,
                    upscale=1,  # upscale=1 significa sólo mejora, sin cambio de tamaño
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=device
                )
                print(f"{NAME}: GFPGANer inicializado correctamente en {device}.")

            except Exception as e:
                print(f"{NAME}: Error inicializando GFPGANer: {e}")
                # Intento de fallback a CPU si falla la inicialización con GPU
                if device is not None and device.type != 'cpu':
                    print(f"{NAME}: Retroceso a CPU debido al error.")
                    try:
                        device = torch.device("cpu")
                        FACE_ENHANCER = gfpgan.GFPGANer(
                            model_path=model_path,
                            upscale=1,
                            arch='clean',
                            channel_multiplier=2,
                            bg_upsampler=None,
                            device=device
                        )
                        print(f"{NAME}: GFPGANer inicializado correctamente en CPU tras fallback.")
                    except Exception as fallback_e:
                         print(f"{NAME}: FATAL: No se pudo inicializar GFPGANer ni siquiera en CPU: {fallback_e}")
                         FACE_ENHANCER = None # Asegurar None si falla totalmente
                else:
                     print(f"{NAME}: FATAL: No se pudo inicializar GFPGANer en CPU: {e}")
                     FACE_ENHANCER = None # Asegurar None si falla totalmente


    # Comprobar si el enhancer sigue siendo None después del intento
    if FACE_ENHANCER is None:
        raise RuntimeError(f"{NAME}: Falló la inicialización de GFPGANer. Revisa los registros para más detalles.")

    return FACE_ENHANCER


def enhance_face(temp_frame: Frame) -> Frame:
    """Mejora las caras en un único frame usando la instancia global de GFPGANer."""
    # Asegurarse de que el enhancer está listo
    enhancer = get_face_enhancer()
    try:
        with THREAD_SEMAPHORE:
            # El método enhance devuelve: _, restored_faces, restored_img
            _, _, restored_img = enhancer.enhance(
                temp_frame,
                has_aligned=False, # Asumir que las caras no están prealineadas
                only_center_face=False, # Mejorar todas las caras detectadas
                paste_back=True # Pegar las caras mejoradas de vuelta en la imagen original
            )
        # GFPGAN puede devolver None si no detecta cara o ocurre un error
        if restored_img is None:
            # print(f"{NAME}: Advertencia: GFPGAN devolvió None. Devolviendo frame original.")
            return temp_frame
        return restored_img
    except Exception as e:
        print(f"{NAME}: Error durante la mejora de la cara: {e}")
        # Devolver el frame original en caso de error durante la mejora
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame) -> Frame:
    """Procesa un frame: mejora la cara si se detecta."""
    # No necesitamos estrictamente source_face para la mejora
    # Podemos confiar en enhance_face que intentará mejorar si hay cara
    temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Procesa múltiples frames a partir de rutas de archivo."""
    for temp_frame_path in temp_frame_paths:
        if not os.path.exists(temp_frame_path):
            print(f"{NAME}: Advertencia: Ruta de frame no encontrada {temp_frame_path}, saltando.")
            if progress:
                progress.update(1)
            continue

        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(f"{NAME}: Advertencia: Error leyendo frame {temp_frame_path}, saltando.")
            if progress:
                progress.update(1)
            continue

        result_frame = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result_frame)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    """Procesa un único archivo de imagen."""
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: No se pudo leer la imagen objetivo {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Imagen mejorada guardada en {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    """Procesa frames de vídeo usando el núcleo de procesamiento de frames."""
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

# --- FIN DEL ARCHIVO face_enhancer.py ---
