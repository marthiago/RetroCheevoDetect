import cv2
import tensorflow as tf
import numpy as np
import csv
import sys
import pytesseract
import os
from datetime import datetime

# Carregar modelo
model_path = '/Users/thiagomartins/Projetos/RetroCheevoDetect/models/retroachievements_model.keras'
model = tf.keras.models.load_model(model_path)

# Função para salvar frames
def save_frame(frame, frame_number, folder='samples'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    frame_filename = os.path.join(folder, f'frame_{frame_number}.png')
    cv2.imwrite(frame_filename, frame)

# Função para converter segundos para formato h:mm:ss
def seconds_to_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02}:{int(s):02}"

# Função para analisar vídeo
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.basename(video_path)
    video_dir = os.path.dirname(video_path)
    csv_filename = f"{video_name[:20]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(video_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'Start Time', 'End Time', 'Notification Text', 'Frame Inicial', 'Frame Final', 'Percentual'])
        
        threshold = 0.3  # Ajuste do threshold de predição
        
        notification_detected = False
        start_frame = None
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width, _ = frame.shape
            # Definir a caixa delimitadora para os 25% superiores da tela a partir do canto superior esquerdo
            x, y, w, h = 0, 0, int(width * 0.5), int(height * 0.5)
            notification_area = frame[y:y+h, x:x+w]

            frame_resized = cv2.resize(frame, (150, 150))
            frame_normalized = frame_resized / 255.0
            frame_reshaped = np.reshape(frame_normalized, (1, 150, 150, 3))
            
            prediction = model.predict(frame_reshaped)
            if prediction[0] > threshold:
                # Extração de texto usando OCR na área da notificação
                notification_text = pytesseract.image_to_string(notification_area).strip()
                
                # Verificar se texto foi detectado
                if notification_text:
                    if not notification_detected:
                        start_frame = i
                        save_frame(frame, start_frame, folder='samples')
                        notification_detected = True
                    
                    end_frame = i
                    start_time_seconds = start_frame / fps
                    end_time_seconds = end_frame / fps
                    start_time = seconds_to_time(start_time_seconds)
                    end_time = seconds_to_time(end_time_seconds)
                    
                    # Calcular percentual do vídeo
                    percentual = (i / frame_count) * 100
                    
                    csvwriter.writerow([video_name, start_time, end_time, notification_text, start_frame, end_frame, f"{percentual:.2f}%"])
                    print(f"Notification found at {start_time} (Frame {start_frame}) to {end_time} (Frame {end_frame}): {notification_text}")
                else:
                    notification_detected = False
            else:
                notification_detected = False

            progress = (i / frame_count) * 100
            print(f"Processing: {progress:.2f}%")
    
    cap.release()
    print(f"Analysis complete. CSV saved at {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_video.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    analyze_video(video_path)
