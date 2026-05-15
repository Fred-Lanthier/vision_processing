import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import math
import numpy as np
import pyrealsense2 as rs

# Initialisation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- PARAMETRAGE VISUEL ---
COLOR_CLOSED = (50, 255, 150)
COLOR_OPEN = (50, 150, 255)
COLOR_TARGET = (0, 255, 255)
COLOR_HUD_BG = (30, 30, 30)

# --- CLASSE DE GÉOMÉTRIE CORRIGÉE ---
class CameraGeometry:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        
        # Constante physique : Ecart moyen entre les coins des yeux (en cm)
        self.REAL_EYE_DIST_CM = 6.4 

    def get_metric_coordinates(self, pixel_point, eye_distance_pixels):
        """
        Convertit un point (u, v) en (X, Y, Z) métrique (cm) en utilisant fx et fy séparément.
        """
        if eye_distance_pixels == 0: return (0, 0, 0)

        # 1. Calcul de Z (Profondeur)
        # On utilise fx car les yeux sont alignés horizontalement
        Z_cm = (self.fx * self.REAL_EYE_DIST_CM) / eye_distance_pixels

        # 2. Récupération des coordonnées pixels (u, v)
        u, v = pixel_point

        # 3. Calcul de X (Horizontal) avec fx
        # Formule : X = (Pixel - CentreX) * Z / fx
        X_cm = (u - self.cx) * Z_cm / self.fx

        # 4. Calcul de Y (Vertical) avec fy
        # Formule : Y = (Pixel - CentreY) * Z / fy
        Y_cm = (v - self.cy) * Z_cm / self.fy

        return (X_cm, Y_cm, Z_cm)


class RealSenseColorCapture:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        stream_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = stream_profile.get_intrinsics()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        return True, np.asanyarray(color_frame.get_data())

    def isOpened(self):
        return True

    def release(self):
        self.pipeline.stop()


def draw_complete_interface(image, distance, status, face_landmarks, center_point, metric_coords):
    h, w, _ = image.shape
    overlay = image.copy()
    main_color = COLOR_CLOSED if status == "Fermee" else COLOR_OPEN

    if face_landmarks:
        # A. Moulage
        for connection in mp_face_mesh.FACEMESH_LIPS:
            pt1 = face_landmarks.landmark[connection[0]]
            pt2 = face_landmarks.landmark[connection[1]]
            cv2.line(image, (int(pt1.x*w), int(pt1.y*h)), (int(pt2.x*w), int(pt2.y*h)), main_color, 1, cv2.LINE_AA)

        # B. Viseur
        if center_point:
            cx, cy = center_point
            cv2.line(image, (cx-10, cy), (cx+10, cy), COLOR_TARGET, 1)
            cv2.line(image, (cx, cy-10), (cx, cy+10), COLOR_TARGET, 1)

    # --- HUD LATÉRAL ---
    cv2.rectangle(overlay, (20, 20), (280, 280), COLOR_HUD_BG, -1)
    
    cv2.putText(image, "COORD PIXELS (u, v)", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    if center_point:
        cv2.putText(image, f"u: {center_point[0]}  v: {center_point[1]}", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TARGET, 1)

    # --- AFFICHAGE MÉTRIQUE ---
    cv2.putText(image, "COORD REELLES (X, Y)", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    X, Y, Z = metric_coords
    
    # Affichage clair X et Y
    cv2.putText(image, f"X: {X:.1f} cm", (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(image, f"Y: {Y:.1f} cm", (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    
    # Affichage Profondeur
    cv2.putText(image, f"Dist (Z): {Z:.1f} cm", (40, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    # Statut
    cv2.putText(image, status.upper(), (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, main_color, 2)

    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    return image


try:
    cap = RealSenseColorCapture()
except RuntimeError as exc:
    raise SystemExit(
        "Cannot start RealSense color stream. Close realsense-viewer/ROS camera nodes "
        f"or check USB connection. Details: {exc}"
    )

success, frame = cap.read()
if not success:
    cap.release()
    raise SystemExit("RealSense started, but no color frame was received.")

h, w, _ = frame.shape
intr = cap.intrinsics
geom = CameraGeometry(w, h, intr.fx, intr.fy, intr.ppx, intr.ppy)
print(f"Géométrie RealSense : fx={geom.fx:.2f}, fy={geom.fy:.2f}, Centre=({geom.cx:.2f}, {geom.cy:.2f})")

while cap.isOpened():
    success, image = cap.read()
    if not success: break
    image = cv2.flip(image, 1)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    status = "..."
    center_px = None
    metric_coords = (0, 0, 0)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        
        # 1. Calcul du centre pixel (u, v)
        top, bottom = lm.landmark[13], lm.landmark[14]
        cx, cy = int((top.x + bottom.x)/2 * w), int((top.y + bottom.y)/2 * h)
        center_px = (cx, cy)
        
        # 2. Distance des yeux (Pixels) pour calculer Z
        eye_left = lm.landmark[33]   # Coin ext gauche
        eye_right = lm.landmark[263] # Coin ext droit
        
        # Pythagore pour la distance précise en pixels
        eye_dist_px = math.sqrt((eye_left.x*w - eye_right.x*w)**2 + (eye_left.y*h - eye_right.y*h)**2)
        
        # 3. Calcul métrique (Cœur du sujet)
        metric_coords = geom.get_metric_coordinates(center_px, eye_dist_px)

        # Calcul statut ouverture
        mouth_dist = math.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
        status = "Ouverte" if mouth_dist > 0.04 else "Fermee"

    draw_complete_interface(image, 0, status, 
                          results.multi_face_landmarks[0] if results.multi_face_landmarks else None, 
                          center_px, metric_coords)

    cv2.imshow('Tracking Metrique (fx/fy)', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
