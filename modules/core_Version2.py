import os
import sys
# Hilo único duplica el rendimiento en CUDA - debe establecerse antes de importar torch
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# Reducir el nivel de logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

# Si se usa ROCMExecutionProvider, liberar referncias a torch para evitar problemas
if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    # Capturar Ctrl+C y llamar a destroy()
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='Seleccionar una imagen de origen', dest='source_path')
    program.add_argument('-t', '--target', help='Seleccionar una imagen o video objetivo', dest='target_path')
    program.add_argument('-o', '--output', help='Seleccionar archivo o directorio de salida', dest='output_path')
    program.add_argument('--frame-processor', help='Pipeline de procesadores por frame', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='Conservar fps originales', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='Conservar audio original', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='Conservar frames temporales', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='Procesar todas las caras detectadas', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='Filtrar imágenes o videos NSFW', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='Mapear caras origen/destino', dest='map_faces', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='Enmascarar la región de la boca', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--video-encoder', help='Ajustar el encoder de salida de video', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='Ajustar calidad de video (CRF)', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('-l', '--lang', help='Idioma de la interfaz', default="en")
    program.add_argument('--live-mirror', help='Mostrar espejo en modo cámara en vivo', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='El marco de cámara en vivo es redimensionable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='Cantidad máxima de RAM en GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='Proveedor de ejecución', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='Número de hilos de ejecución', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # Registrar argumentos obsoletos (ocultos)
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang

    # Para el conmutador (tumbler) del ENHANCER:
    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False

    # Traducir argumentos obsoletos
    if args.source_path_deprecated:
        print('\033[33mEl argumento -f y --face está obsoleto. Use -s y --source en su lugar.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mEl argumento --cpu-cores está obsoleto. Use --execution-threads en su lugar.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mEl argumento --gpu-vendor apple está obsoleto. Use --execution-provider coreml en su lugar.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mEl argumento --gpu-vendor nvidia está obsoleto. Use --execution-provider cuda en su lugar.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mEl argumento --gpu-vendor amd está obsoleto. Use --execution-provider cuda en su lugar.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mEl argumento --gpu-threads está obsoleto. Use --execution-threads en su lugar.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    # Recomienda memoria según la plataforma
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # Evitar fugas de memoria de tensorflow
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # Limitar uso de memoria si está configurado
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    # Vaciar caché de CUDA si corresponde
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    # Comprobaciones previas: versión de Python y ffmpeg
    if sys.version_info < (3, 9):
        update_status('La versión de Python no es compatible - por favor actualice a 3.9 o superior.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg no está instalado.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    # Muestra o envía estado a la interfaz
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)

def start() -> None:
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status('Procesando...')
    # Procesar imagen a imagen
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print("Error copiando archivo:", str(e))
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progresando...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            update_status('¡Procesamiento a imagen completado!')
        else:
            update_status('¡El procesamiento a imagen falló!')
        return
    # Procesar imagen a video
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
        return

    if not modules.globals.map_faces:
        update_status('Creando recursos temporales...')
        create_temp(modules.globals.target_path)
        update_status('Extrayendo frames...')
        extract_frames(modules.globals.target_path)

    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progresando...', frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        release_resources()
    # Maneja fps
    if modules.globals.keep_fps:
        update_status('Detectando fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creando video con {fps} fps...')
        create_video(modules.globals.target_path, fps)
    else:
        update_status('Creando video con 30.0 fps...')
        create_video(modules.globals.target_path)
    # Manejo de audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restaurando audio...')
        else:
            update_status('Restaurar audio puede causar problemas si no se mantiene el fps...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    # Limpiar y validar
    clean_temp(modules.globals.target_path)
    if is_video(modules.globals.target_path):
        update_status('¡Procesamiento a video completado!')
    else:
        update_status('¡El procesamiento a video falló!')


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit: quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()