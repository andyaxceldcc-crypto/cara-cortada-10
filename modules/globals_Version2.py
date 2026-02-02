# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Datos para el mapeo de caras
source_target_map: List[Dict[str, Any]] = [] # Almacena el mapeo detallado para procesamiento de imagen/video
simple_map: Dict[str, Any] = {}             # Almacena mapeo simplificado (embeddings/caras) para modo en vivo/simple

# Rutas
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Opciones de procesamiento
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Procesar todas las caras detectadas con la fuente por defecto
map_faces: bool = False          # Usar source_target_map o simple_map para swaps específicos
poisson_blend: bool = False      # Habilitar mezcla Poisson para swaps más suaves
color_correction: bool = False   # Habilitar corrección de color (implementación específica)
nsfw_filter: bool = False

# Opciones de salida de video
video_encoder: str | None = None
video_quality: int | None = None # Normalmente valor CRF o bitrate

# Opciones de modo en vivo
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder para elemento UI si es necesario
webcam_preview_running: bool = False
show_fps: bool = False

# Configuración del sistema
max_memory: int | None = None        # Límite de memoria en GB (si se usa)
execution_providers: List[str] = []  # p.ej., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Número de hilos para ejecución en CPU
headless: bool | None = None         # Ejecutar sin UI?
log_level: str = "error"             # Nivel de logging ('debug','info','warning','error')

# Conmutadores UI del procesador facial (ejemplo)
fp_ui: Dict[str, bool] = {"face_enhancer": False}

# Opciones específicas del face swapper
face_swapper_enabled: bool = True # Interruptor general para el procesador swapper
opacity: float = 1.0              # Factor de mezcla para la cara swappeada (0.0-1.0)
sharpness: float = 0.0            # Mejora de nitidez para la cara swappeada (0.0-1.0+)

# Opciones de máscara para la boca
mouth_mask: bool = False           # Habilitar enmascaramiento de la zona de la boca
show_mouth_mask_box: bool = False  # Visualizar la caja de la máscara de la boca (para debug)
mask_feather_ratio: int = 12       # Denominador para cálculo de difuminado (mayor = difuminado más pequeño)
mask_down_size: float = 0.1        # Factor de expansión para labio inferior (relativo)
mask_size: float = 1.0             # Factor de expansión para labio superior (relativo)

# --- START: Añadido para interpolación de frames ---
enable_interpolation: bool = True # Activar suavizado temporal
interpolation_weight: float = 0  # Peso de mezcla para el frame actual (0.0-1.0). Más bajo = más suavizado.
# --- END: Añadido para interpolación de frames ---

# --- END OF FILE globals.py ---