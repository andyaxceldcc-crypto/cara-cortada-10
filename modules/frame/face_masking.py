import cv2
import numpy as np
from modules.typing import Face, Frame
import modules.globals

def apply_color_transfer(source, target):
    """
    Aplicar transferencia de color desde la imagen target a la imagen source.
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    # Reorganizar mean y std para que sean broadcastables
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Realizar la transferencia de color
    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Convertir landmarks a int32
        landmarks = landmarks.astype(np.int32)

        # Extraer rasgos faciales
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        # Calcular padding
        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )  # 5% del ancho de la cara

        # Crear un hull convexo ligeramente mayor para el padding
        hull = cv2.convexHull(face_outline)
        hull_padded = []
        for point in hull:
            x, y = point[0]
            center = np.mean(face_outline, axis=0)
            direction = np.array([x, y]) - center
            direction = direction / np.linalg.norm(direction)
            padded_point = np.array([x, y]) + direction * padding
            hull_padded.append(padded_point)

        hull_padded = np.array(hull_padded, dtype=np.int32)

        # Rellenar el hull convexo con padding
        cv2.fillConvexPoly(mask, hull_padded, 255)

        # Suavizar los bordes de la máscara
        mask = cv2.GaussianBlur(mask, (5, 5), 3)

    return mask

def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        #                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
        lower_lip_order = [
            65,
            66,
            62,
            70,
            69,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            0,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            65,
        ]
        lower_lip_landmarks = landmarks[lower_lip_order].astype(
            np.float32
        )  # Usar float para cálculos más precisos

        # Calcular el centro de los landmarks
        center = np.mean(lower_lip_landmarks, axis=0)

        # Expandir los landmarks hacia afuera usando mouth_mask_size
        expansion_factor = (
            1 + modules.globals.mask_down_size * modules.globals.mouth_mask_size
        )  # Ajuste de expansión según slider
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

        # Extender la parte superior del labio
        toplip_indices = [
            20,
            0,
            1,
            2,
            3,
            4,
            5,
        ]  # Índices para landmarks 2, 65, 66, 62, 70, 69, 18
        toplip_extension = (
            modules.globals.mask_size * modules.globals.mouth_mask_size * 0.5
        )  # Ajuste de extensión según slider
        for idx in toplip_indices:
            direction = expanded_landmarks[idx] - center
            direction = direction / np.linalg.norm(direction)
            expanded_landmarks[idx] += direction * toplip_extension

        # Extender la parte inferior (zona del mentón)
        chin_indices = [
            11,
            12,
            13,
            14,
            15,
            16,
        ]  # Índices para landmarks 21, 22, 23, 24, 0, 8
        chin_extension = 2 * 0.2  # Ajustar este factor para controlar la extensión
        for idx in chin_indices:
            expanded_landmarks[idx][1] += (
                expanded_landmarks[idx][1] - center[1]
            ) * chin_extension

        # Convertir de nuevo a coordenadas enteras
        expanded_landmarks = expanded_landmarks.astype(np.int32)

        # Calcular bounding box para la parte inferior expandida de la boca
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Añadir un poco de padding al bounding box
        padding = int((max_x - min_x) * 0.1)  # 10% de padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        # Asegurar que las dimensiones del bounding box sean válidas
        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1

        # Crear la máscara
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [expanded_landmarks - [min_x, min_y]], 255)

        # Aplicar Gaussian blur para suavizar los bordes de la máscara
        mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 5)

        # Colocar la máscara ROI en la máscara completa
        mask[min_y:max_y, min_x:max_x] = mask_roi

        # Extraer el área enmascarada del frame
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

        # Devolver el polígono expandido de la parte inferior del labio en coordenadas del frame original
        lower_lip_polygon = expanded_landmarks

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon

def create_eyes_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyes_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Landmarks ojo izquierdo (87-96) y ojo derecho (33-42)
        left_eye = landmarks[87:96]
        right_eye = landmarks[33:42]
        
        # Calcular centros y dimensiones para cada ojo
        left_eye_center = np.mean(left_eye, axis=0).astype(np.int32)
        right_eye_center = np.mean(right_eye, axis=0).astype(np.int32)
        
        # Calcular dimensiones del ojo con ajuste de tamaño
        def get_eye_dimensions(eye_points):
            x_coords = eye_points[:, 0]
            y_coords = eye_points[:, 1]
            width = int((np.max(x_coords) - np.min(x_coords)) * (1 + modules.globals.mask_down_size * modules.globals.eyes_mask_size))
            height = int((np.max(y_coords) - np.min(y_coords)) * (1 + modules.globals.mask_down_size * modules.globals.eyes_mask_size))
            return width, height
        
        left_width, left_height = get_eye_dimensions(left_eye)
        right_width, right_height = get_eye_dimensions(right_eye)
        
        # Añadir padding extra
        padding = int(max(left_width, right_width) * 0.2)
        
        # Calcular bounding box para ambos ojos
        min_x = min(left_eye_center[0] - left_width//2, right_eye_center[0] - right_width//2) - padding
        max_x = max(left_eye_center[0] + left_width//2, right_eye_center[0] + right_width//2) + padding
        min_y = min(left_eye_center[1] - left_height//2, right_eye_center[1] - right_height//2) - padding
        max_y = max(left_eye_center[1] + left_height//2, right_eye_center[1] + right_height//2) + padding
        
        # Asegurar que las coordenadas estén dentro del frame
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(frame.shape[1], max_x)
        max_y = min(frame.shape[0], max_y)
        
        # Crear máscara para la región de ojos
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        
        # Dibujar elipses para ambos ojos
        left_center = (left_eye_center[0] - min_x, left_eye_center[1] - min_y)
        right_center = (right_eye_center[0] - min_x, right_eye_center[1] - min_y)
        
        # Calcular ejes (mitad de width y height)
        left_axes = (left_width//2, left_height//2)
        right_axes = (right_width//2, right_height//2)
        
        # Dibujar elipses rellenadas
        cv2.ellipse(mask_roi, left_center, left_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(mask_roi, right_center, right_axes, 0, 0, 360, 255, -1)
        
        # Aplicar Gaussian blur para suavizar los bordes
        mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 5)
        
        # Colocar la máscara ROI en la máscara completa
        mask[min_y:max_y, min_x:max_x] = mask_roi
        
        # Extraer el recorte de la imagen para los ojos
        eyes_cutout = frame[min_y:max_y, min_x:max_x].copy()
        
        # Crear puntos poligonales para visualización
        def create_ellipse_points(center, axes):
            t = np.linspace(0, 2*np.pi, 32)
            x = center[0] + axes[0] * np.cos(t)
            y = center[1] + axes[1] * np.sin(t)
            return np.column_stack((x, y)).astype(np.int32)
        
        # Generar puntos para ambas elipses
        left_points = create_ellipse_points((left_eye_center[0], left_eye_center[1]), (left_width//2, left_height//2))
        right_points = create_ellipse_points((right_eye_center[0], right_eye_center[1]), (right_width//2, right_height//2))
        
        # Combinar puntos de ambos ojos
        eyes_polygon = np.vstack([left_points, right_points])
        
    return mask, eyes_cutout, (min_x, min_y, max_x, max_y), eyes_polygon

def create_curved_eyebrow(points):
    if len(points) >= 5:
        # Ordenar puntos por coordenada x
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        # Calcular dimensiones
        x_min, y_min = np.min(sorted_points, axis=0)
        x_max, y_max = np.max(sorted_points, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        
        # Crear más puntos para una curva más suave
        num_points = 50
        x = np.linspace(x_min, x_max, num_points)
        
        # Ajustar una curva cuadrática para un arco más natural
        coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 2)
        y = np.polyval(coeffs, x)
        
        # Offset aumentados para crear más separación
        top_offset = height * 0.5  # Aumentado desde 0.3 para subir más
        bottom_offset = height * 0.2  # Aumentado desde 0.1 para bajar más
        
        # Crear curvas suaves
        top_curve = y - top_offset
        bottom_curve = y + bottom_offset
        
        # Crear extremos curvados con mayor taper
        end_points = 5
        start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)  # Taper aumentado
        end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)  # Taper aumentado
        
        # Crear extremos afinados
        start_curve = np.column_stack((
            start_x,
            np.linspace(bottom_curve[0], top_curve[0], end_points)
        ))
        end_curve = np.column_stack((
            end_x,
            np.linspace(bottom_curve[-1], top_curve[-1], end_points)
        ))
        
        # Combinar todos los puntos para formar un contorno suave
        contour_points = np.vstack([
            start_curve,
            np.column_stack((x, top_curve)),
            end_curve,
            np.column_stack((x[::-1], bottom_curve[::-1]))
        ])
        
        # Añadir un pequeño padding para mejor cobertura
        center = np.mean(contour_points, axis=0)
        vectors = contour_points - center
        padded_points = center + vectors * 1.2  # Padding ligeramente aumentado
        
        return padded_points
    return points

def create_eyebrows_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyebrows_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Landmarks ceja izquierda (97-105) y ceja derecha (43-51)
        left_eyebrow = landmarks[97:105].astype(np.float32)
        right_eyebrow = landmarks[43:51].astype(np.float32)
        
        # Calcular centros y dimensiones de cada ceja
        left_center = np.mean(left_eyebrow, axis=0)
        right_center = np.mean(right_eyebrow, axis=0)
        
        # Calcular bounding box con padding ajustado por tamaño
        all_points = np.vstack([left_eyebrow, right_eyebrow])
        padding_factor = modules.globals.eyebrows_mask_size
        min_x = np.min(all_points[:, 0]) - 25 * padding_factor
        max_x = np.max(all_points[:, 0]) + 25 * padding_factor
        min_y = np.min(all_points[:, 1]) - 20 * padding_factor
        max_y = np.max(all_points[:, 1]) + 15 * padding_factor
        
        # Asegurar que las coordenadas estén dentro del frame
        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(frame.shape[1], int(max_x))
        max_y = min(frame.shape[0], int(max_y))
        
        # Crear máscara para la región de las cejas
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        
        try:
            # Convertir puntos a coordenadas locales
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]
            
            def create_curved_eyebrow(points):
                if len(points) >= 5:
                    # Ordenar puntos por x
                    sorted_idx = np.argsort(points[:, 0])
                    sorted_points = points[sorted_idx]
                    
                    # Calcular dimensiones
                    x_min, y_min = np.min(sorted_points, axis=0)
                    x_max, y_max = np.max(sorted_points, axis=0)
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Crear más puntos para una curva suave
                    num_points = 50
                    x = np.linspace(x_min, x_max, num_points)
                    
                    # Ajustar cuadrática para el arco
                    coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 2)
                    y = np.polyval(coeffs, x)
                    
                    # Offsets aumentados para separación
                    top_offset = height * 0.5
                    bottom_offset = height * 0.2
                    
                    top_curve = y - top_offset
                    bottom_curve = y + bottom_offset
                    
                    end_points = 5
                    start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)
                    end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)
                    
                    start_curve = np.column_stack((
                        start_x,
                        np.linspace(bottom_curve[0], top_curve[0], end_points)
                    ))
                    end_curve = np.column_stack((
                        end_x,
                        np.linspace(bottom_curve[-1], top_curve[-1], end_points)
                    ))
                    
                    contour_points = np.vstack([
                        start_curve,
                        np.column_stack((x, top_curve)),
                        end_curve,
                        np.column_stack((x[::-1], bottom_curve[::-1]))
                    ])
                    
                    center = np.mean(contour_points, axis=0)
                    vectors = contour_points - center
                    padded_points = center + vectors * 1.2
                    
                    return padded_points
                return points
            
            # Generar y dibujar formas de cejas
            left_shape = create_curved_eyebrow(left_local)
            right_shape = create_curved_eyebrow(right_local)
            
            # Aplicar multi-etapa de desenfoque para un feathering natural
            # Primero, Gaussian fuerte para suavizado inicial
            mask_roi = cv2.GaussianBlur(mask_roi, (21, 21), 7)
            
            # Segundo, desenfoque medio para áreas de transición
            mask_roi = cv2.GaussianBlur(mask_roi, (11, 11), 3)
            
            # Finalmente, desenfoque ligero para detalles finos
            mask_roi = cv2.GaussianBlur(mask_roi, (5, 5), 1)
            
            # Normalizar valores de la máscara
            mask_roi = cv2.normalize(mask_roi, None, 0, 255, cv2.NORM_MINMAX)
            
            # Colocar la máscara ROI en la máscara completa
            mask[min_y:max_y, min_x:max_x] = mask_roi
            
            # Extraer el recorte de la imagen para las cejas
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
            
            # Combinar puntos para visualización
            eyebrows_polygon = np.vstack([
                left_shape + [min_x, min_y],
                right_shape + [min_x, min_y]
            ]).astype(np.int32)
            
        except Exception as e:
            # Fallback a polígonos simples si el ajuste de curva falla
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]
            cv2.fillPoly(mask_roi, [left_local.astype(np.int32)], 255)
            cv2.fillPoly(mask_roi, [right_local.astype(np.int32)], 255)
            mask_roi = cv2.GaussianBlur(mask_roi, (21, 21), 7)
            mask[min_y:max_y, min_x:max_x] = mask_roi
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
            eyebrows_polygon = np.vstack([left_eyebrow, right_eyebrow]).astype(np.int32)
        
    return mask, eyebrows_cutout, (min_x, min_y, max_x, max_y), eyebrows_polygon

def apply_mask_area(
    frame: np.ndarray,
    cutout: np.ndarray,
    box: tuple,
    face_mask: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or polygon is None
    ):
        return frame

    try:
        resized_cutout = cv2.resize(cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_cutout.shape:
            resized_cutout = cv2.resize(
                resized_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_area = apply_color_transfer(resized_cutout, roi)

        # Crear máscara para el área
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        
        # Dividir puntos para parte izquierda y derecha si es necesario
        if len(polygon) > 50:  # Umbral arbitrario para detectar si hay múltiples partes
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point] - [min_x, min_y]
            right_points = polygon[mid_point:] - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [left_points], 255)
            cv2.fillPoly(polygon_mask, [right_points], 255)
        else:
            adjusted_polygon = polygon - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        # Aplicar feathering inicial fuerte
        polygon_mask = cv2.GaussianBlur(polygon_mask, (21, 21), 7)

        # Aplicar feathering adicional
        feather_amount = min(
            30,
            box_width // modules.globals.mask_feather_ratio,
            box_height // modules.globals.mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(float), (0, 0), feather_amount
        )
        feathered_mask = feathered_mask / feathered_mask.max()

        # Aplicar suavizado adicional a los bordes de la máscara
        feathered_mask = cv2.GaussianBlur(feathered_mask, (5, 5), 1)

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi / 255.0)

        combined_mask = combined_mask[:, :, np.newaxis]
        blended = (
            color_corrected_area * combined_mask + roi * (1 - combined_mask)
        ).astype(np.uint8)

        # Aplicar máscara de cara al resultado mezclado
        face_mask_3channel = (
            np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
        )
        final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        pass

    return frame

def draw_mask_visualization(
    frame: Frame,
    mask_data: tuple,
    label: str,
    draw_method: str = "polygon"
) -> Frame:
    mask, cutout, (min_x, min_y, max_x, max_y), polygon = mask_data

    vis_frame = frame.copy()

    # Asegurar coordenadas dentro de los límites del frame
    height, width = vis_frame.shape[:2]
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(width, max_x), min(height, max_y)

    if draw_method == "ellipse" and len(polygon) > 50:  # Para ojos
        # Dividir puntos para la parte izquierda y derecha
        mid_point = len(polygon) // 2
        left_points = polygon[:mid_point]
        right_points = polygon[mid_point:]
        
        try:
            # Ajustar elipses a puntos - se necesitan al menos 5 puntos
            if len(left_points) >= 5 and len(right_points) >= 5:
                # Convertir puntos al formato float32 correcto para fitEllipse
                left_points = left_points.astype(np.float32)
                right_points = right_points.astype(np.float32)
                
                # Ajustar elipses
                left_ellipse = cv2.fitEllipse(left_points)
                right_ellipse = cv2.fitEllipse(right_points)
                
                # Dibujar las elipses
                cv2.ellipse(vis_frame, left_ellipse, (0, 255, 0), 2)
                cv2.ellipse(vis_frame, right_ellipse, (0, 255, 0), 2)
        except Exception as e:
            # Si falla fitEllipse, dibujar rectángulos como fallback
            left_rect = cv2.boundingRect(left_points)
            right_rect = cv2.boundingRect(right_points)
            cv2.rectangle(vis_frame, 
                        (left_rect[0], left_rect[1]), 
                        (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), 
                        (0, 255, 0), 2)
            cv2.rectangle(vis_frame,
                        (right_rect[0], right_rect[1]),
                        (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]),
                        (0, 255, 0), 2)
    else:  # Para boca y cejas
        # Dibujar el polígono
        if len(polygon) > 50:  # Si tenemos múltiples partes
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point]
            right_points = polygon[mid_point:]
            cv2.polylines(vis_frame, [left_points], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.polylines(vis_frame, [right_points], True, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.polylines(vis_frame, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    # Añadir etiqueta
    cv2.putText(
        vis_frame,
        label,
        (min_x, min_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return vis_frame 
