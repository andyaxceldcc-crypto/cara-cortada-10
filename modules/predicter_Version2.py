import numpy
import opennsfw2
from PIL import Image
import cv2  # Import de OpenCV
import modules.globals  # Importa globals para acceder al interruptor de corrección de color

from modules.typing import Frame

MAX_PROBABILITY = 0.85

# Pre-cargar el modelo para eficiencia
model = None

def predict_frame(target_frame: Frame) -> bool:
    # Convertir el frame a RGB antes de procesar si la corrección de color está habilitada
    if modules.globals.color_correction:
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        
    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
    global model
    if model is None: 
        model = opennsfw2.make_open_nsfw_model()
        
    views = numpy.expand_dims(image, axis=0)
    _, probability = model.predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    return opennsfw2.predict_image(target_path) > MAX_PROBABILITY


def predict_video(target_path: str) -> bool:
    _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
    return any(probability > MAX_PROBABILITY for probability in probabilities)