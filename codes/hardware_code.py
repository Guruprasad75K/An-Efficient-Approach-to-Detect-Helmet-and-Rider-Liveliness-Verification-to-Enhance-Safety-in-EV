import time
import os
import requests
import cv2
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
from ultralytics import YOLO
from picamera2 import Picamera2
from collections import deque
import numpy as np

# ================= CONFIG =================
API_URL = "https://serverless.roboflow.com"
MODEL_ID = "face-anti-spoofing-icbck/1"
API_KEY = "#################"

# GPIO pins (BCM)
MOTOR_IN1 = 27
MOTOR_IN2 = 22
MOTOR_ENA = 18
BUZZER_PIN = 17

# motor speeds
NORMAL_SPEED = 75
STAGE2_SPEED = 40

# thresholds / timers
HELMET_WINDOW_SEC = 3.0
STAGE_COUNTDOWN_SEC = 6
SPOOF_CONF_THRESHOLD = 0.92   # if fake_conf >= this -> immediate Stage-1
SPOOF_REQUIRED_STRIKES = 2    # also keep strikes for robustness
API_MIN_INTERVAL = 1.0
CAM_SIZE = (640, 480)

# startup wait: how long to try to get initial detections before falling back (seconds)
STARTUP_TIMEOUT = 8.0

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)
GPIO.setup(MOTOR_ENA, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

motor_pwm = GPIO.PWM(MOTOR_ENA, 1000)
motor_pwm.start(0)
GPIO.output(BUZZER_PIN, GPIO.LOW)

# ================= LCD SETUP =================
try:
    lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)
except Exception:
    lcd = CharLCD('PCF8574', 0x3F, cols=16, rows=2)

def display_message(line1, line2=""):
    lcd.clear()
    lcd.write_string(line1[:16])
    if line2:
        lcd.cursor_pos = (1, 0)
        lcd.write_string(line2[:16])

# ================= MOTOR / BUZZER =================
def motor_forward(speed=NORMAL_SPEED):
    GPIO.output(MOTOR_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(speed)

def motor_stop():
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(0)

def buzzer_on(): GPIO.output(BUZZER_PIN, GPIO.HIGH)
def buzzer_off(): GPIO.output(BUZZER_PIN, GPIO.LOW)

# ================= HELMET VOTING BUFFER =================
helmet_votes = deque()
def add_helmet_vote(detected_bool):
    now = time.time()
    helmet_votes.append((now, bool(detected_bool)))
    while helmet_votes and (now - helmet_votes[0][0]) > HELMET_WINDOW_SEC:
        helmet_votes.popleft()

def helmet_status_from_votes():
    now = time.time()
    while helmet_votes and (now - helmet_votes[0][0]) > HELMET_WINDOW_SEC:
        helmet_votes.popleft()
    if not helmet_votes:
        return False, "No Detection"
    trues = sum(1 for (_, v) in helmet_votes if v)
    falses = len(helmet_votes) - trues
    return (True, "withHelmet") if trues >= falses else (False, "withOUTHelmet")

# ================= SPOOF STRIKES / API (in-memory) =================
spoof_strikes = 0
last_api_time = 0.0

def call_spoof_api_and_update_inmemory(rgb_frame, force_call=False):
    """
    rgb_frame: numpy array in RGB
    returns (real_detected_or_None, fake_detected_or_None,
             conf_real (0-1) or 0.0, conf_fake (0-1) or 0.0, api_resp_or_None)
    """
    global last_api_time, spoof_strikes
    now = time.time()
    if (not force_call) and (now - last_api_time < API_MIN_INTERVAL):
        return None, None, None, None, None
    last_api_time = now

    # encode in memory as jpeg
    try:
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    except Exception:
        bgr = rgb_frame
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None, None, None, None, None
    files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}

    api_resp = None
    real_detected = False
    fake_detected = False
    conf_real = 0.0
    conf_fake = 0.0

    try:
        r = requests.post(f"{API_URL}/{MODEL_ID}?api_key={API_KEY}", files=files, timeout=6)
        api_resp = r.json()
    except Exception as e:
        print("API error:", e)
        api_resp = None

    if api_resp and "predictions" in api_resp:
        for p in api_resp.get("predictions", []):
            cls = p.get("class", "").lower()
            conf = float(p.get("confidence", 0.0))
            if cls == "real":
                real_detected = True
                conf_real = max(conf_real, conf)
            elif cls == "fake":
                fake_detected = True
                conf_fake = max(conf_fake, conf)

    # update strikes and immediate violation
    if fake_detected and conf_fake >= SPOOF_CONF_THRESHOLD:
        spoof_strikes += 1
    elif real_detected:
        spoof_strikes = 0
    else:
        # ambiguous -> keep strikes unchanged
        pass

    return real_detected, fake_detected, conf_real, conf_fake, api_resp

def spoof_violation_now():
    """Return boolean: immediate if conf_fake >= threshold, OR strikes exceeded."""
    return spoof_strikes >= SPOOF_REQUIRED_STRIKES

# ================= STATE MACHINE =================
STAGE_NORMAL = "NORMAL"
STAGE_1 = "STAGE1"
STAGE_2 = "STAGE2"
STAGE_SHUTDOWN = "SHUTDOWN"

state = STAGE_NORMAL
stage_timer_start = None
last_buzzer_beep = 0.0

def enter_stage(new_stage):
    global state, stage_timer_start, last_buzzer_beep
    state = new_stage
    stage_timer_start = time.time()
    last_buzzer_beep = 0.0
    print("STATE ->", state)

def get_stage_elapsed():
    return 0.0 if not stage_timer_start else (time.time() - stage_timer_start)

# ========= NOT NECCESSAY, DEPENDS ON CAMERA ============
# ================= COLOR FIX FUNCTIONS =================
def simple_white_balance_rgb(img_rgb):
    try:
        arr = img_rgb.astype(np.float32)
        r_mean = arr[:,:,0].mean(); g_mean = arr[:,:,1].mean(); b_mean = arr[:,:,2].mean()
        if r_mean <= 0 or g_mean <= 0 or b_mean <= 0:
            return img_rgb
        mean_gray = (r_mean + g_mean + b_mean) / 3.0
        kr = mean_gray / r_mean; kg = mean_gray / g_mean; kb = mean_gray / b_mean
        arr[:,:,0] = np.clip(arr[:,:,0] * kr, 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1] * kg, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] * kb, 0, 255)
        return arr.astype(np.uint8)
    except Exception:
        return img_rgb

def enhance_color_rgb(img_rgb):
    # white-balance + CLAHE on V channel
    try:
        wb = simple_white_balance_rgb(img_rgb)
        hsv = cv2.cvtColor(wb, cv2.COLOR_RGB2HSV)
        h,s,v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v2 = clahe.apply(v)
        hsv2 = cv2.merge((h,s,v2))
        out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)
        return out
    except Exception:
        return img_rgb

# ================= LOAD HELMET MODEL & CAMERA =================
helmet_model = YOLO('/home/pi/models/Helmet_Detection.pt')
print("Helmet model loaded. Classes:", helmet_model.names)

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": CAM_SIZE})
picam2.configure(config)
# try enabling AWB/AE controls - driver dependent
try:
    picam2.set_controls({"AwbEnable": True, "AeEnable": True})
except Exception:
    pass
picam2.start()
time.sleep(0.4)

# Initial UI & motor OFF until startup done
display_message("System Ready", "Waiting Start")
motor_stop()

# ================= STARTUP FLAGS =================
startup = True
startup_start_time = time.time()
seen_helmet_once = False
seen_spoof_once = False

# overlay colors
COLOR_HELMET = (0,255,0)
COLOR_NOHELM = (0,0,255)
COLOR_SPOOF = (0,0,255)
COLOR_REAL = (0,255,0)

try:
    print("System running - press Q or Ctrl+C to quit")
    while True:
        frame_src = picam2.capture_array()  # often RGB or BGRA depending on driver

        # Robust channel handling:
        # If 4 channels (BGRA), convert to BGR
        # If 3 channels we assume frame_src is RGB (Picamera2 configured RGB888)
        if frame_src.ndim == 3 and frame_src.shape[2] == 4:
            # BGRA -> BGR
            bgr = cv2.cvtColor(frame_src, cv2.COLOR_BGRA2BGR)
        elif frame_src.ndim == 3 and frame_src.shape[2] == 3:
            # frame likely in RGB from Picamera2; convert to BGR for display
            # but keep a correct RGB copy for models
            bgr = cv2.cvtColor(frame_src, cv2.COLOR_RGB2BGR)
        else:
            # fallback: convert as BGR
            bgr = frame_src.copy()

        # Prepare a clean RGB for model usage (convert from BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Enhance color on the RGB for both display and inference
        rgb_enh = enhance_color_rgb(rgb)

        # Use enhanced RGB for model inputs; create BGR display from it
        bgr_display = cv2.cvtColor(rgb_enh, cv2.COLOR_RGB2BGR)

        # ---------- HELMET DETECTION (YOLO) ----------
        # Ultralytics expects RGB arrays
        result1 = helmet_model(rgb_enh, conf=0.5)[0]
        helmet_detected_now = False
        helmet_conf = 0.0
        if result1.boxes is not None and len(result1.boxes) > 0:
            for box in result1.boxes:
                try:
                    cls = int(box.cls[0])
                    conf_box = float(box.conf[0])
                except Exception:
                    cls = None
                    conf_box = 0.0
                if cls == 0:  # withHelmet
                    helmet_detected_now = True
                    helmet_conf = max(helmet_conf, conf_box)

        # record instant seen for startup condition
        if helmet_detected_now:
            seen_helmet_once = True

        # push vote for rolling window (used after startup)
        add_helmet_vote(helmet_detected_now)
        helmet_status, helmet_reason = helmet_status_from_votes()

        # ---------- SPOOF API (in-memory) ----------
        api_force = startup and ((time.time() - startup_start_time) <= STARTUP_TIMEOUT)
        api_real, api_fake, conf_real, conf_fake, api_raw = call_spoof_api_and_update_inmemory(rgb_enh, force_call=api_force)

        # mark seen_spoof_once when any valid API prediction appears during startup
        if api_real is True or api_fake is True:
            seen_spoof_once = True

        # If we got real immediately during startup, we prefer that for auto-start
        if api_real:
            seen_spoof_once = True

        # If we have a strong fake this frame, we consider spoof violation immediate
        immediate_spoof_flag = False
        if (conf_fake is not None) and (conf_fake >= SPOOF_CONF_THRESHOLD):
            immediate_spoof_flag = True

        # spoof violation check (strikes or immediate)
        spoof_now = (spoof_strikes >= SPOOF_REQUIRED_STRIKES) or immediate_spoof_flag

        # ---------- STARTUP HANDLING ----------
        if startup:
            # require at least one helmet instant detection and one API detection of face (real or fake)
            if seen_helmet_once and seen_spoof_once and api_real:
                startup = False
                display_message("Status: Running", "")
                motor_forward(NORMAL_SPEED)
                print("Startup instant-safe detected -> motor started, startup flag cleared.")
            else:
                if not seen_helmet_once:
                    display_message("No Detection", "Wear Helmet")
                elif not seen_spoof_once:
                    display_message("Helmet Detected", "Waiting Spoof")
                else:
                    display_message("Waiting Results", "")
                if (time.time() - startup_start_time) > STARTUP_TIMEOUT:
                    startup = False
                    print("Startup timeout expired; proceeding to normal operation (motor remains off until safe).")

        # ---------- NORMAL OPERATION ----------
        if not startup:
            # Spoof priority: if spoof_now true -> spoof_violation
            spoof_violation = spoof_now
            # Helmet violation only matters if spoof not currently violating
            helmet_violation = (not helmet_status) and (not spoof_violation)

            # STATE MACHINE TRANSITIONS
            if state == STAGE_NORMAL:
                if helmet_violation or spoof_violation:
                    enter_stage(STAGE_1)
            elif state == STAGE_1:
                elapsed = get_stage_elapsed()
                if (not helmet_violation) and (not spoof_violation):
                    enter_stage(STAGE_NORMAL)
                    motor_forward(NORMAL_SPEED)
                    display_message("Status: Running", "")
                else:
                    remaining = max(0, STAGE_COUNTDOWN_SEC - int(elapsed))
                    top = "Wear Helmet!" if helmet_violation else ("Spoof Detected!" if spoof_violation else "Warning!")
                    # bottom compact shows confidence
                    if spoof_violation:
                        bottom = f"Fake:{int(conf_fake*100) if conf_fake else 0}%"
                    else:
                        bottom = f"H:{int(helmet_conf*100)}%"
                    display_message(top, f"Slow {remaining}s {bottom}")
                    now = time.time()
                    if now - last_buzzer_beep >= 1.0:
                        buzzer_on(); time.sleep(0.12); buzzer_off(); last_buzzer_beep = now
                    if elapsed >= STAGE_COUNTDOWN_SEC:
                        enter_stage(STAGE_2); motor_forward(STAGE2_SPEED)
            elif state == STAGE_2:
                elapsed = get_stage_elapsed()
                if (not helmet_violation) and (not spoof_violation):
                    enter_stage(STAGE_NORMAL)
                    motor_forward(NORMAL_SPEED)
                    display_message("Status: Running", "")
                else:
                    remaining = max(0, STAGE_COUNTDOWN_SEC - int(elapsed))
                    top = "Wear Helmet!" if helmet_violation else ("Spoof Detected!" if spoof_violation else "Warning!")
                    if spoof_violation:
                        bottom = f"Fake:{int(conf_fake*100) if conf_fake else 0}%"
                    else:
                        bottom = f"H:{int(helmet_conf*100)}%"
                    display_message(top, f"Shut {remaining}s {bottom}")
                    now = time.time()
                    if now - last_buzzer_beep >= 1.0:
                        buzzer_on(); time.sleep(0.12); buzzer_off(); last_buzzer_beep = now
                    if elapsed >= STAGE_COUNTDOWN_SEC:
                        enter_stage(STAGE_SHUTDOWN); motor_stop(); buzzer_on(); display_message(top, "Motor Stopped")
            elif state == STAGE_SHUTDOWN:
                if helmet_status and (spoof_strikes == 0):
                    buzzer_off(); motor_forward(NORMAL_SPEED); display_message("Status: Running", ""); enter_stage(STAGE_NORMAL)
                else:
                    top = "Wear Helmet!" if helmet_violation else ("Spoof Detected!" if spoof_violation else "Motor Stopped")
                    display_message(top, "Motor Stopped")
                    buzzer_on()

        # ---------- OVERLAY & DISPLAY ----------
        try:
            annotated_rgb = result1.plot()  # returns RGB annotated
            display_img = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            display_img = bgr_display.copy()

        # overlay texts (left column)
        y0 = 20
        line_h = 26
        cv2.putText(display_img, f"State:{state}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y = y0 + line_h
        cv2.putText(display_img, f"HelmetVote:{'OK' if helmet_status else 'NO'}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HELMET if helmet_status else COLOR_NOHELM, 2)
        y += line_h
        cv2.putText(display_img, f"H_conf:{int(helmet_conf*100)}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
        y += line_h
        cv2.putText(display_img, f"Fake_conf:{int(conf_fake*100) if conf_fake else 0}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SPOOF, 2)
        y += line_h
        cv2.putText(display_img, f"Real_conf:{int(conf_real*100) if conf_real else 0}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_REAL, 2)

        cv2.imshow("Helmet Detection", display_img)

        # Debug line
        print(f"[DEBUG] state:{state} startup:{startup} seen_helmet:{seen_helmet_once} seen_spoof:{seen_spoof_once} "
              f"votes:{len(helmet_votes)} trues:{sum(1 for _,v in helmet_votes if v)} helmet_status:{helmet_status} helmet_conf={helmet_conf:.2f} spoof_strikes:{spoof_strikes} conf_fake={conf_fake:.2f} conf_real={conf_real:.2f}")

        # key handling
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

        # small sleep
        time.sleep(0.03)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    print("Cleanup...")
    motor_stop()
    buzzer_off()
    motor_pwm.stop()
    display_message("System Off", "")
    time.sleep(1)
    lcd.clear()
    GPIO.cleanup()
    try:
        picam2.close()
    except:
        pass
    cv2.destroyAllWindows()
    print("Done.")
