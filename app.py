from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, session
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2
import pygame
import pyttsx3
import torch
import datetime
import warnings
from twilio.rest import Client
from geopy.geocoders import Nominatim
import geocoder
import logging
from flask_mysqldb import MySQL
import json
from waitress import serve

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('comtypes').setLevel(logging.INFO)

warnings.filterwarnings('ignore', category=FutureWarning)

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'database'
app.config['SECRET_KEY'] = 'your-secret-key-here'

mysql = MySQL(app)

current_user_id = None

# Update Twilio configuration with your credentials
TWILIO_ACCOUNT_SID = 'AC9d11ee651425855d4c4c3621d4c565e7'
TWILIO_AUTH_TOKEN = '63d5382999d0e7660f0e44b047138318'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'
WHATSAPP_FROM_NUMBER = 'whatsapp:+14155238886'

# Drowsiness alert threshold
DROWSINESS_ALERT_THRESHOLD = 2
drowsiness_count = 0

print("-> Loading face detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Loading YOLO model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)
model.eval()

# Detection thresholds
PHONE_CLASS_IDS = [67]  # YOLO class ID for cell phone
FOOD_CLASS_ID = [52, 53, 54, 55]
EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 2
YAWN_THRESH = 20
REDNESS_THRESH = 0.08
SEATBELT_CLASS_ID = 73
PHONE_NEAR_EAR_THRESHOLD = 0.5  # Lowered threshold for better detection
PHONE_IN_VIEW_THRESHOLD = 0.3
PHONE_ALERT_COOLDOWN = 5  # seconds between phone alerts

pygame.mixer.init()

status_data = {
    "ear": None,
    "yawn_distance": None,
    "phone_detected": False,
    "phone_near_ear": False,
    "eating_detected": False,
    "seatbelt_detected": False,
    "drowsiness": False,
    "red_eyes": False,
    "yawning": False,
    "current_time": "",
    "alerts": [],
    "whatsapp_status": "",
    "phone_confidence": 0
}

vs = None
monitoring = False

def sound_alarm(path, is_drowsy=False):
    try:
        if is_drowsy:
            pygame.mixer.music.load(path)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
    except Exception as e:
        logging.error(f"Error playing sound alarm: {str(e)}")

def speak_alarm(message):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.say(message)
        engine.runAndWait()
        logging.debug(f"Voice alert played: {message}")
    except Exception as e:
        logging.error(f"Error in text-to-speech: {str(e)}")

def eye_aspect_ratio(eye):
    try:
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    except:
        return 0.15

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    try:
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        distance = abs(top_mean[1] - low_mean[1])
        return distance
    except:
        return 0

def detect_redness_improved(frame, eye):
    try:
        x_min, y_min = np.min(eye, axis=0)
        x_max, y_max = np.max(eye, axis=0)
        padding = 20
        x_min = max(x_min - padding, 0)
        y_min = max(y_min - padding, 0)
        x_max = min(x_max + padding, frame.shape[1])
        y_max = min(y_max + padding, frame.shape[0])
        eye_region = frame[y_min:y_max, x_min:x_max]
        if eye_region.size == 0:
            return 0, None
        eye_region = cv2.GaussianBlur(eye_region, (3,3), 0)
        hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        red_ranges = [
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([170, 70, 50]), np.array([180, 255, 255])),
            (np.array([0, 50, 50]), np.array([5, 255, 255]))
        ]
        combined_mask = np.zeros(eye_region.shape[:2], dtype=np.uint8)
        for lower, upper in red_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        total_pixels = eye_region.shape[0] * eye_region.shape[1]
        red_pixels = np.sum(combined_mask > 0)
        redness_ratio = red_pixels / total_pixels
        vis_image = eye_region.copy()
        vis_image[combined_mask > 0] = [0, 0, 255]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        return redness_ratio, vis_image
    except Exception as e:
        logging.error(f"Error in redness detection: {str(e)}")
        return 0, None

def detect_distractions(frame):
    try:
        if frame is None or frame.size == 0:
            logging.error("Invalid frame received in detect_distractions")
            return False, False, False, 0, 0, False

        # Convert to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame, size=640)
        
        phone_detected = False
        phone_confidence = 0

        # Process YOLO detections
        for det in results.xyxy[0]:
            class_id = int(det[5])
            confidence = float(det[4])
            
            if confidence > 0.3:  # Reduced threshold for better detection
                x1, y1, x2, y2 = map(int, det[:4])
                
                # Phone detection
                if class_id in PHONE_CLASS_IDS:
                    phone_detected = True
                    phone_confidence = max(phone_confidence, confidence)
                    
                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Phone: {confidence:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return phone_detected, False, False, phone_confidence, 0, False
        
    except Exception as e:
        logging.error(f"Error in distraction detection: {str(e)}")
        return False, False, False, 0, 0, False

# Fix the function definition to match how it's called
def handle_phone_detection(frame, phone_detected, phone_near_ear, phone_conf, last_alert_time):
    current_time = time.time()
    alert_triggered = False
    
    if phone_detected and (current_time - last_alert_time) > PHONE_ALERT_COOLDOWN:
        cv2.putText(frame, "WARNING: PHONE DETECTED!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        speak_alarm("Warning! Phone usage detected!")
        alert_triggered = True
        last_alert_time = current_time
        status_data["alerts"].append("Phone Detected!")
        if current_user_id:
            # Log phone detection with confidence score
            log_detection('phone', phone_conf, {
                "timestamp": datetime.datetime.now().isoformat(),
                "confidence": phone_conf,
                "alert_type": "phone_detected"
            })
    
    return alert_triggered, last_alert_time

def create_dashboard(frame, ear, yawn_distance, phone_conf, phone_detected, phone_near_ear):
    dashboard_height = 80  # Increased height for more info
    dashboard = np.zeros((dashboard_height, frame.shape[1], 3), dtype=np.uint8)
    current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
    
    # Time display
    cv2.putText(dashboard, f"Time: {current_time}", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Metrics display
    metrics = []
    colors = []
    
    if ear is not None:
        metrics.append(f"EAR: {ear:.2f}")
        colors.append((0, 255, 0) if ear > EYE_AR_THRESH else (0, 0, 255))
    
    if yawn_distance is not None:
        metrics.append(f"YAWN: {yawn_distance:.2f}")
        colors.append((0, 255, 0) if yawn_distance < YAWN_THRESH else (0, 0, 255))
    
    if phone_detected:
        if phone_near_ear:
            metrics.append(f"PHONE: NEAR EAR! ({phone_conf:.2f})")
            colors.append((0, 0, 255))  # Red for danger
        else:
            metrics.append(f"PHONE: DETECTED ({phone_conf:.2f})")
            colors.append((0, 165, 255))  # Orange for warning
    
    # Display metrics with appropriate colors
    x_offset = 10
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        cv2.putText(dashboard, metric, (x_offset, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        x_offset += 250  # Increased spacing
    
    return dashboard

def gen_frames():
    global vs, monitoring, current_user_id, drowsiness_count

    if not monitoring:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    try:
        if vs is None:
            logging.info("Initializing laptop camera...")
            vs = VideoStream(src=0).start()
            time.sleep(1.0)
    except Exception as e:
        logging.error(f"Failed to initialize camera: {str(e)}")
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Camera Error!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    COUNTER = 0
    REDNESS_COUNTER = 0
    YAWN_COUNTER = 0
    last_phone_alert_time = time.time()
    last_drowsy_alert_time = time.time()
    last_yawn_alert_time = time.time()
    ear = None
    yawn_distance = None

    while monitoring:
        frame = vs.read()
        if frame is None:
            logging.warning("Failed to read frame, reinitializing...")
            vs.stop()
            vs = VideoStream(src=0).start()
            time.sleep(3.0)
            continue

        logging.debug("Frame read successfully")
        frame = imutils.resize(frame, width=480)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect distractions (phones, etc.)
        phone_detected, phone_near_ear, eating_detected, phone_conf, food_conf, seatbelt_detected = detect_distractions(frame)
        
        # Handle phone alerts
        phone_alert_triggered, last_phone_alert_time = handle_phone_detection(
            frame, phone_detected, phone_near_ear, phone_conf, last_phone_alert_time
        )

        # Face detection and drowsiness analysis
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            ear, leftEye, rightEye = final_ear(shape)
            yawn_distance = lip_distance(shape)
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 2)
            mouth = shape[48:60]
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 2)

            if yawn_distance > YAWN_THRESH:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= 3:
                    cv2.putText(frame, "YAWNING DETECTED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    current_time = time.time()
                    if (current_time - last_yawn_alert_time) > 5:
                        speak_alarm("You are frequently yawning! Please take rest!")
                        last_yawn_alert_time = current_time
                        status_data["alerts"].append("Yawning Detected!")
                        if current_user_id:
                            log_detection('yawning', None, {"timestamp": datetime.datetime.now().isoformat()})
            else:
                YAWN_COUNTER = max(0, YAWN_COUNTER - 1)

            left_redness, _ = detect_redness_improved(frame, leftEye)
            right_redness, _ = detect_redness_improved(frame, rightEye)
            if left_redness > REDNESS_THRESH or right_redness > REDNESS_THRESH:
                REDNESS_COUNTER += 1
                if REDNESS_COUNTER > 10:
                    cv2.putText(frame, "RED EYES DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    speak_alarm("Red eyes detected. Please take rest!")
                    REDNESS_COUNTER = 0
                    status_data["alerts"].append("Red Eyes Detected!")
                    if current_user_id:
                        log_detection('red_eyes', None, {"timestamp": datetime.datetime.now().isoformat()})
            else:
                REDNESS_COUNTER = max(0, REDNESS_COUNTER - 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    current_time = time.time()
                    if (current_time - last_drowsy_alert_time) > 0.5:
                        Thread(target=sound_alarm, args=("Alert.WAV", True), daemon=True).start()
                        speak_alarm("Drowsiness Detected! Please take a break!")
                        last_drowsy_alert_time = current_time
                        status_data["alerts"].append("Drowsiness Detected!")
                        if current_user_id:
                            global drowsiness_count
                            drowsiness_count += 1
                            log_detection('drowsiness', ear, {"timestamp": datetime.datetime.now().isoformat()})
                            if drowsiness_count >= DROWSINESS_ALERT_THRESHOLD:
                                with app.app_context():  # Push application context
                                    send_whatsapp_alert(current_user_id)
                                drowsiness_count = 0  # Reset after sending alert
            else:
                COUNTER = 0

        if not seatbelt_detected:
            cv2.putText(frame, "WEAR SEATBELT!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            if current_user_id:
                log_detection('seatbelt', None, {"timestamp": datetime.datetime.now().isoformat()})

        # Create dashboard and combine with frame
        dashboard = create_dashboard(frame, ear, yawn_distance, phone_conf, phone_detected, phone_near_ear)
        combined_frame = np.vstack([frame, dashboard])

        # Update status data
        status_data.update({
            "ear": ear,
            "yawn_distance": yawn_distance,
            "phone_detected": phone_detected,
            "phone_near_ear": phone_near_ear,
            "eating_detected": eating_detected,
            "seatbelt_detected": seatbelt_detected,
            "drowsiness": COUNTER >= EYE_AR_CONSEC_FRAMES,
            "red_eyes": REDNESS_COUNTER > 10,
            "yawning": YAWN_COUNTER >= 3,
            "phone_conf": phone_conf,
            "current_time": datetime.datetime.now().strftime("%I:%M:%S %p"),
            "alerts": status_data["alerts"]
        })

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', combined_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/phone')
def phone():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('phone.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        phone_number = request.form['phone_number']  # New field
        user_type = request.form['user_type']
        
        if user_type == 'admin':
            return render_template('register.html', error="Admin registration is not allowed")
        
        cur = mysql.connection.cursor()
        try:
            # Modified query to include phone_number
            cur.execute("INSERT INTO users (name, email, password, phone_number) VALUES (%s, %s, %s, %s)",
                       (name, email, password, phone_number))
            mysql.connection.commit()
            cur.close()
            return redirect(url_for('login'))
        except Exception as e:
            cur.close()
            return render_template('register.html', error="Registration failed: " + str(e))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    global current_user_id
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']
        
        if user_type == 'admin':
            if email == 'khushalmali7524@gmail.com' and password == '1234':
                session['logged_in'] = True
                session['user_type'] = 'admin'
                current_user_id = 1
                session['user_id'] = current_user_id
                session['name'] = 'Admin'
                return redirect(url_for('admin'))
            else:
                return render_template('login.html', error="Invalid admin credentials")
        
        cur = mysql.connection.cursor()
        try:
            cur.execute("SELECT * FROM users WHERE email = %s AND password = %s", 
                       (email, password))
            user = cur.fetchone()
            
            if user:
                session['logged_in'] = True
                session['user_type'] = 'user'
                current_user_id = user[0]
                session['user_id'] = user[0]
                session['name'] = user[1]
                return redirect(url_for('phone'))
            else:
                return render_template('login.html', error="Invalid credentials")
                
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            return render_template('login.html', error=f"Login error: {str(e)}")
        finally:
            cur.close()
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    global current_user_id
    session.clear()
    current_user_id = None
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('detection.html')

@app.route('/whatsapp_setup')
def whatsapp_setup():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('whatsapp_setup.html')

@app.route('/video_feed')
def video_feed():
    response = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/status')
def status():
    return jsonify(status_data)

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    try:
        global monitoring, vs
        if not monitoring:
            logging.info("Setting monitoring to True")
            monitoring = True
            if vs is None:
                logging.info("Initializing laptop camera...")
                vs = VideoStream(src=0).start()
                time.sleep(1.0)
            logging.info(f"Monitoring status after start: {monitoring}")
        return jsonify({'status': 'Monitoring started successfully'})
    except Exception as e:
        logging.error(f"Error in start_monitoring: {str(e)}")
        return jsonify({'status': f'Error: {str(e)}'}), 500

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    try:
        global monitoring, vs
        logging.info("Stopping monitoring")
        monitoring = False
        if vs is not None:
            logging.info("Stopping video stream...")
            vs.stop()
            vs = None
        status_data["alerts"] = []
        status_data["whatsapp_status"] = ""
        return jsonify({'status': 'Monitoring stopped successfully'})
    except Exception as e:
        logging.error(f"Error in stop_monitoring: {str(e)}")
        return jsonify({'status': f'Error: {str(e)}'}), 500

@app.route('/admin')
def admin():
    if not session.get('logged_in') or session.get('user_type') != 'admin':
        return redirect(url_for('login'))
    
    cur = mysql.connection.cursor()
    try:
        cur.execute("""
            SELECT 
                u.name, 
                u.email,
                u.phone_number,
                COUNT(CASE WHEN d.detection_type IN ('drowsiness', 'phone', 'red_eyes') THEN 1 END) as total_detections,
                COUNT(CASE WHEN d.detection_type = 'drowsiness' THEN 1 END) as drowsiness_count,
                COUNT(CASE WHEN d.detection_type = 'phone' THEN 1 END) as phone_count,
                COUNT(CASE WHEN d.detection_type = 'red_eyes' THEN 1 END) as red_eyes_count,
                COUNT(CASE WHEN d.detection_type = 'seatbelt' THEN 1 END) as seatbelt_count,
                MAX(d.detection_time) as last_detection,
                MAX(CASE WHEN d.detection_type = 'phone' 
                    THEN d.confidence_score END) as max_phone_confidence
            FROM users u
            LEFT JOIN detection_logs d ON u.id = d.user_id
            WHERE u.email != 'khushalmali7524@gmail.com'
            GROUP BY u.id, u.name, u.email, u.phone_number
        """)
        users = [dict(zip([column[0] for column in cur.description], row)) for row in cur.fetchall()]
        
        # Get detailed detection logs
        cur.execute("""
            SELECT 
                u.name as user_name,
                d.detection_type,
                d.detection_time,
                d.confidence_score,
                d.additional_info
            FROM detection_logs d
            JOIN users u ON d.user_id = u.id
            WHERE d.detection_type IN ('phone', 'drowsiness', 'red_eyes', 'seatbelt')
            ORDER BY d.detection_time DESC
            LIMIT 50
        """)
        detection_logs = [dict(zip([column[0] for column in cur.description], row)) for row in cur.fetchall()]
        
        return render_template('admin.html', users=users, detection_logs=detection_logs)
    except Exception as e:
        logging.error(f"Admin dashboard error: {str(e)}")
        return "Error loading dashboard", 500
    finally:
        cur.close()

def log_detection(detection_type, confidence_score=None, additional_info=None):
    global current_user_id
    
    if not current_user_id:
        logging.debug("Skipping detection logging: No user logged in")
        return
    
    with app.app_context():
        cur = mysql.connection.cursor()
        try:
            cur.execute("SELECT id FROM users WHERE id = %s", (current_user_id,))
            user_exists = cur.fetchone()
            
            if not user_exists:
                logging.warning(f"User {current_user_id} not found in database")
                return
                
            cur.execute("""
                INSERT INTO detection_logs 
                (user_id, detection_type, confidence_score, additional_info, detection_time)
                VALUES (%s, %s, %s, %s, NOW())
            """, (current_user_id, detection_type, confidence_score, 
                  json.dumps(additional_info) if additional_info else None))
            mysql.connection.commit()
            logging.debug(f"Successfully logged {detection_type} for user {current_user_id}")
            
        except Exception as e:
            logging.error(f"Detection logging error: {str(e)}")
            mysql.connection.rollback()
        finally:
            cur.close()

@app.route('/save_whatsapp', methods=['POST'])
def save_whatsapp():
    if not session.get('logged_in'):
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    try:
        whatsapp_number = request.form.get('whatsapp_number')
        if not whatsapp_number:
            return jsonify({'status': 'error', 'message': 'No number provided'}), 400

        if not whatsapp_number.startswith('+') or not len(whatsapp_number) >= 10 or not whatsapp_number[1:].isdigit():
            return jsonify({'status': 'error', 'message': 'Invalid WhatsApp number format (e.g., +1234567890)'}), 400

        cur = mysql.connection.cursor()
        try:
            # Check if number already exists for this user
            cur.execute("SELECT id FROM whatsapp_numbers WHERE user_id = %s AND whatsapp_number = %s", 
                       (session.get('user_id'), whatsapp_number))
            if cur.fetchone():
                return jsonify({'status': 'error', 'message': 'This number is already registered'}), 400
            
            # Add new number
            cur.execute("INSERT INTO whatsapp_numbers (user_id, whatsapp_number) VALUES (%s, %s)", 
                       (session.get('user_id'), whatsapp_number))
            mysql.connection.commit()
            return jsonify({'status': 'success', 'message': 'WhatsApp number saved successfully'})
        finally:
            cur.close()
    except Exception as e:
        logging.error(f"Error saving WhatsApp number: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error saving number: {str(e)}'}), 500

def send_whatsapp_alert(user_id):
    with app.app_context():
        try:
            # Check if alert was already sent in last 30 minutes
            cur = mysql.connection.cursor()
            cur.execute("""
                SELECT detection_time 
                FROM detection_logs 
                WHERE user_id = %s 
                AND detection_type = 'whatsapp_alert'
                AND detection_time > NOW() - INTERVAL 30 MINUTE
                ORDER BY detection_time DESC LIMIT 1
            """, (user_id,))
            last_alert = cur.fetchone()
            
            if last_alert:
                status_data["whatsapp_status"] = "Alert already sent in last 30 minutes"
                cur.close()
                return

            # Get all WhatsApp numbers for the user
            cur.execute("SELECT whatsapp_number FROM whatsapp_numbers WHERE user_id = %s", (user_id,))
            numbers = cur.fetchall()
            
            if not numbers:
                status_data["whatsapp_status"] = "Error: No WhatsApp numbers saved"
                cur.close()
                return

            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            # Get location info
            geolocator = Nominatim(user_agent="driver_safety_app")
            g = geocoder.ip('me')
            if g and g.latlng:
                lat, lng = g.latlng
                location = geolocator.reverse(f"{lat}, {lng}", language='en')
                address = location.address if location else "Unknown location"
            else:
                lat, lng = None, None
                address = "Unknown location"
            
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            
            message_text = (
                "🚨 URGENT ALERT! 🚨\n"
                "Your friend seems to be in drowsy condition while driving!\n"
                f"Time: {current_time}\n\n"
                f"📍 Last Known Location:\n{address}\n\n"
                "🗺️ Track Location:\n"
                f"https://www.google.com/maps?q={lat},{lng if lat else ''}\n\n"
                "This is an automated safety alert."
            )
            
            # Send to all registered numbers
            for number in numbers:
                whatsapp_number = number[0]
                if not whatsapp_number.startswith('whatsapp:'):
                    whatsapp_number = f'whatsapp:{whatsapp_number}'
                
                message = client.messages.create(
                    body=message_text,
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=whatsapp_number
                )
            
            # Log the alert
            cur.execute("""
                INSERT INTO detection_logs 
                (user_id, detection_type, additional_info, detection_time)
                VALUES (%s, 'whatsapp_alert', %s, NOW())
            """, (user_id, json.dumps({'sent_to': len(numbers)})))
            mysql.connection.commit()
            
            status_data["whatsapp_status"] = f"WhatsApp alert sent to {len(numbers)} numbers"
            logging.info(f"WhatsApp alert sent to {len(numbers)} numbers")
            
        except Exception as e:
            error_msg = str(e)
            status_data["whatsapp_status"] = f"Error: {error_msg}"
            logging.error(f"WhatsApp alert error: {error_msg}")
        finally:
            cur.close()

if __name__ == '__main__':
    try:
        print("Starting server on http://127.0.0.1:5000")
        serve(app, host='127.0.0.1', port=5000, threads=6)
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        if vs is not None:
            vs.stop()

