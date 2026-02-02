import cv2
import numpy as np
from typing import Optional, Tuple, Callable
import platform
import threading

# Solo importar librerías específicas de Windows cuando se ejecute en Windows
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None

        # Inicializar componentes específicos de Windows si corresponde
        if platform.system() == "Windows":
            self.graph = FilterGraph()
            # Verificar que el dispositivo exista
            devices = self.graph.get_input_devices()
            if self.device_index >= len(devices):
                raise ValueError(
                    f"Índice de dispositivo inválido {device_index}. Dispositivos disponibles: {len(devices)}"
                )

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        """Inicializa y arranca la captura de video"""
        try:
            if platform.system() == "Windows":
                # Métodos de captura específicos para Windows
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),  # Intentar DirectShow primero
                    (self.device_index, cv2.CAP_ANY),  # Luego backend por defecto
                    (-1, cv2.CAP_ANY),  # Fallback con -1
                    (0, cv2.CAP_ANY),  # Finalmente 0 sin backend específico
                ]

                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue
            else:
                # Sistemas Unix-like (Linux/Mac)
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("No se pudo abrir la cámara")

            # Configurar formato
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

            self.is_running = True
            return True

        except Exception as e:
            print(f"Error al iniciar la captura: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Leer un frame desde la cámara"""
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def release(self) -> None:
        """Detener captura y liberar recursos"""
        if self.is_running and self.cap is not None:
            self.cap.release()
            self.is_running = False
            self.cap = None

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Registrar callback para procesamiento de frames"""
        self.frame_callback = callback