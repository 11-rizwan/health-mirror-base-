
import cv2, mediapipe as mp, numpy as np, tensorflow as tf, time, base64, io, json
from scipy.spatial import distance as dist
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_change_this' # Change this to something random
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
socketio = SocketIO(app)

# --- Login Manager Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to /login if user is not authenticated

# --- Database Models ---
# UserMixin provides default implementations for Flask-Login user methods
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    logs = db.relationship('HealthLog', backref='author', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class HealthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    session_data = db.Column(db.String, nullable=False)
    # Foreign Key to link logs to a user
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# This function is required by Flask-Login to load a user from the DB
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Constants ---
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 10  # Number of frames for eye closure to count as a blink/fatigue
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
MODEL_PATH = 'emotion_model_finetuned.h5'
# --- Helper Functions ---
def calculate_ear(eye_landmarks):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    h1 = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

    # Avoid division by zero
    if h1 == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h1)
    return ear

def get_health_score_and_recommendations(emotion, fatigue_level):
    """Calculates a health score and provides recommendations."""
    score = 100
    recommendations = []

    if emotion in ['Angry', 'Sad', 'Fear']:
        score -= 30
        recommendations.append("You seem stressed. Try a 2-minute breathing exercise.")
    elif emotion == 'Neutral':
        score -= 10
        recommendations.append("A quick smile can boost your mood!")

    if fatigue_level > 0.5:
        fatigue_deduction = int(fatigue_level * 40)
        score -= fatigue_deduction
        recommendations.append("You look tired. Remember to take short breaks and stretch.")

    # Placeholder for future hydration logic
    hydration_level = 0.8
    if hydration_level < 0.6:
        score -= 15
        recommendations.append("Don't forget to stay hydrated! Drink some water.")

    score = max(0, score)

    if not recommendations:
        recommendations.append("You're looking great! Keep it up.")

    return score, recommendations

# --- AI Health Analyzer Class ---
class AIHealthAnalyzer:
    def __init__(self):
        # State variables
        self.emotion_text = "Analyzing..."
        self.fatigue_alert = False
        self.fatigue_score = 0.0
        self.ear_counter = 0
        self.last_emotion_check_time = time.time()

        # Load the pre-trained emotion detection model
        try:
            self.emotion_model = tf.keras.models.load_model(MODEL_PATH)
            print(f"--- SUCCESS: The model loaded is: {MODEL_PATH} ---")
        except Exception as e:
            print(f"--- ERROR: FAILED to load model from '{MODEL_PATH}'. ---")
            print(f"REASON: {e}")
            self.emotion_model = None # Handle gracefully if model fails to load

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Eye landmark indices from MediaPipe
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.EAR_LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.EAR_RIGHT_EYE_INDICES = [33, 158, 159, 133, 144, 153]

    def process_frame(self, frame):
        """
        Processes a single video frame to analyze emotion and fatigue.
        This method contains all the logic previously in the `while` loop.
        """
        # If model failed to load, return an error state
        if not self.emotion_model:
            return {"emotion": "Model Error", "fatigueAlert": False, "healthScore": 0, "recommendations": ["Emotion model failed to load."], "ear": "N/A"}

        # Flip for selfie view and get dimensions
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Calculate pixel landmarks once and reuse
            landmarks = np.array(
                [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
            )

            # Use NumPy for faster bounding box calculation
            x_min, y_min = np.min(landmarks, axis=0)
            x_max, y_max = np.max(landmarks, axis=0)
            
            padding = 30
            x_min_padded = max(0, x_min - padding)
            y_min_padded = max(0, y_min - padding)
            x_max_padded = min(w, x_max + padding)
            y_max_padded = min(h, y_max + padding)

            # --- Emotion Detection (runs once per second) ---
            if time.time() - self.last_emotion_check_time > 1.0:
                face_roi = frame[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

                if face_roi.size > 0:
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
                    reshaped_face = np.reshape(resized_face, (1, 48, 48, 1))

                    emotion_prediction = self.emotion_model.predict(reshaped_face)[0]
                    confidence = np.max(emotion_prediction)
                    emotion_index = np.argmax(emotion_prediction)
                    predicted_emotion = EMOTION_LABELS[emotion_index]

                    # Confidence Threshold Logic
                    if predicted_emotion in ['Angry', 'Sad', 'Fear'] and confidence < 0.60:
                        self.emotion_text = 'Neutral'
                    else:
                        self.emotion_text = predicted_emotion

                    self.last_emotion_check_time = time.time()

            # --- Fatigue Detection (EAR calculation) ---
            left_ear_pts = np.array([landmarks[i] for i in self.EAR_LEFT_EYE_INDICES], dtype=np.int32)
            right_ear_pts = np.array([landmarks[i] for i in self.EAR_RIGHT_EYE_INDICES], dtype=np.int32)

            left_ear = calculate_ear(left_ear_pts)
            right_ear = calculate_ear(right_ear_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                self.ear_counter += 1
                if self.ear_counter >= EAR_CONSEC_FRAMES:
                    self.fatigue_alert = True
                    self.fatigue_score = min(1.0, self.fatigue_score + 0.05)
            else:
                self.ear_counter = 0
                self.fatigue_alert = False
                self.fatigue_score = max(0.0, self.fatigue_score - 0.01)

            # --- Health Score & Recommendations ---
            health_score, recommendations = get_health_score_and_recommendations(self.emotion_text, self.fatigue_score)

            return {
                "emotion": self.emotion_text,
                "fatigueAlert": self.fatigue_alert,
                "healthScore": int(health_score),
                "recommendations": recommendations,
                "ear": f"{avg_ear:.2f}"
            }

        # Return default values if no face is detected
        return {"emotion": "Looking for user...", "fatigueAlert": False, "healthScore": "N/A", "recommendations": [], "ear": "N/A"}

# --- Create a single instance of our analyzer ---
analyzer = AIHealthAnalyzer()

# --- Authentication Routes (Login, Register, Logout) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('monitoring'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user is None or not user.check_password(request.form['password']):
            flash('Invalid username or password.', 'error')
            return redirect(url_for('login'))
        login_user(user, remember=True)
        return redirect(url_for('monitoring'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('monitoring'))
    if request.method == 'POST':
        if User.query.filter_by(username=request.form['username']).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
        user = User(username=request.form['username'])
        user.set_password(request.form['password'])
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

# --- Main Application Routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/monitoring')
@login_required # Protect this page
def monitoring():
    return render_template('mirror.html')

@app.route('/history')
@login_required # Protect this page
def history():
    return render_template('history.html')

@app.route('/about')
def about():
    return render_template('about.html')

# --- API Endpoint (Updated for current user) ---
# In app.py

# --- API Endpoint (Corrected for proper data formatting) ---
@app.route('/api/get_history')
@login_required # Protect this endpoint
def get_history():
    # Query the database for logs belonging only to the currently logged-in user
    logs = HealthLog.query.filter_by(user_id=current_user.id).order_by(HealthLog.timestamp.desc()).limit(20).all()
    
    history_data = []
    for log in logs:
        # Load the JSON data from the database record
        data = json.loads(log.session_data)
        
        # Create a dictionary for this log entry
        log_entry = {
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M'),
            'dominantEmotion': data.get('dominantEmotion', 'N/A'), # Use .get() for safety
            'avgScore': data.get('avgScore', 'N/A'),
            'fatigueEvents': data.get('fatigueEvents', 0)
        }
        history_data.append(log_entry)
        
    return jsonify(history_data)

# --- SocketIO Events (Updated for current user) ---
@socketio.on('frame')
def handle_frame(data_url):
    if not current_user.is_authenticated: return
    img_data = base64.b64decode(data_url.split(',')[1])
    frame = cv2.cvtColor(np.array(Image.open(io.BytesIO(img_data))), cv2.COLOR_RGB2BGR)
    results = analyzer.process_frame(frame)
    emit('analysis_results', results)



@socketio.on('save_session')
def handle_save_session(session_data):
    if not current_user.is_authenticated:
        return

    try:
        new_log = HealthLog(session_data=json.dumps(session_data), author=current_user)
        db.session.add(new_log)
        db.session.commit()
        print(f"Session saved successfully for user {current_user.username}")
        # Send a success message back to the client
        emit('session_saved', {'status': 'success'})
    except Exception as e:
        db.session.rollback() # Roll back the transaction on error
        print(f"DATABASE ERROR for user {current_user.username}: {e}")
        # Send a failure message back to the client
        emit('session_saved', {'status': 'failure', 'message': str(e)})
# --- Main Entry Point ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Create database tables if they don't exist
    socketio.run(app, port=5000, debug=True)