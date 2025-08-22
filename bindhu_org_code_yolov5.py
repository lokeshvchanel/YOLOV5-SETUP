from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import asyncio
import gc
import cv2
from ultralytics import YOLO
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import os
import traceback
from screeninfo import get_monitors
import torch

import sys
import torch
import cv2
import numpy as np
import time
import os
import pathlib
# Initialize FastAPI
app = FastAPI()

# Ensure necessary directories exist
log_dir = "application_logs"
confidence_log_dir = "confidence_logs"
video_dir = "detection_videos"

for directory in [log_dir, confidence_log_dir, video_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Setup application error logging (ONLY application errors)
app_error_log_filename = os.path.join(log_dir, f"application_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
app_error_logger = logging.getLogger("app_error_logger")
app_error_handler = logging.FileHandler(app_error_log_filename)
app_error_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
app_error_handler.setFormatter(app_error_formatter)
app_error_logger.addHandler(app_error_handler)
app_error_logger.setLevel(logging.ERROR)

# Setup confidence logging (ONLY confidence > threshold)
confidence_log_filename = os.path.join(confidence_log_dir, f"confidence_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
confidence_logger = logging.getLogger("confidence_logger")
confidence_handler = logging.FileHandler(confidence_log_filename)
confidence_formatter = logging.Formatter("%(asctime)s - Class: %(message)s")
confidence_handler.setFormatter(confidence_formatter)
confidence_logger.addHandler(confidence_handler)
confidence_logger.setLevel(logging.INFO)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)
YOLOV5_PATH = os.path.join(ROOT, 'yolov5')
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Patch PosixPath issue for Windows when loading torch models
class PosixPathPatch(pathlib.PosixPath):
    def __new__(cls, *args, **kwargs):
        return pathlib.WindowsPath(*args, **kwargs)
pathlib.PosixPath = PosixPathPatch

# Load YOLOv5 model

# Load YOLO model with error handling
try:
    model = torch.hub.load(YOLOV5_PATH, 'custom', path=r'fastback\tbale_test1.pt', source='local')
except Exception as e:
    app_error_logger.error(f"Failed to load YOLOv5 model: {traceback.format_exc()}")
    raise RuntimeError("Failed to load YOLOv5 model.")

# Global session management
sessions = {}

class StopRequest(BaseModel):
    session_id: str

class DetectionRequest(BaseModel):
    video_url: str
    supervisor_name: str
    vehicle_number: str
    session_id: str

def cleanup_session(session_id):
    """Release OpenCV resources and remove session."""
    if session_id in sessions:
        session = sessions.pop(session_id, None)
        if session:
            if session.get("cap"):
                session["cap"].release()
            if session.get("video_writer"):
                session["video_writer"].release()
        gc.collect()



async def detect_objects(video_url, session_id, org_url):
    conf_threshold = 0.5

    try:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            app_error_logger.error(f"Failed to open video: {video_url}")
            cleanup_session(session_id)
            return

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_filename = os.path.join(video_dir, f"detection_{session_id}.avi")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

        sessions[session_id]["cap"] = cap
        sessions[session_id]["video_writer"] = video_writer
        sessions[session_id]["count_cooldown"] = 0

        frame_counter = 0

        while cap.isOpened() and session_id in sessions:
            ret, frame = cap.read()
            if not ret:
                app_error_logger.error(f"Video stream failed while processing frame {session_id}")
                break

            try:
                # Run YOLOv5 inference (no tracking IDs)
                results = model(frame)
                detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
                cv2.line(frame, (600, 0), (600, frame.shape[0]), (0, 0, 255), 2)  # left line
                cv2.line(frame, (680, 0), (680, frame.shape[0]), (0, 0, 255), 2)
                for *box, conf, cls_id in detections:
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Create a simple ID for this detection (using box coords rounded)
                    # det_id = (int(x1/10), int(y1/10), int(x2/10), int(y2/10))

                    # Check if object already counted
                    if 600 < cx < 680 and  sessions[session_id]["count_cooldown"] == 0:
                        sessions[session_id]["object_count"] += 1
                        batch_id = list(sessions[session_id]["batches"].keys())[-1]
                        sessions[session_id]["batches"][batch_id]["batch_count"] += 1
                        sessions[session_id]["count_cooldown"] = 20  # mark as counted

                    if conf > conf_threshold:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        confidence_logger.info(f"{timestamp} - Class: {int(cls_id)}, Confidence: {conf:.2f}")

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID:{int(cls_id)} {conf:.2f} ",
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                if sessions[session_id]["count_cooldown"] > 0:
                    sessions[session_id]["count_cooldown"] -= 1
            except Exception as e:
                app_error_logger.error(f"YOLOv5 detection failed for session {session_id}: {traceback.format_exc()}")
                continue

            try:
                cv2.putText(frame, f"{sessions[session_id]['object_count']}", (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (57, 255, 0), 4)
                batch_id = list(sessions[session_id]["batches"].keys())[-1]
                cv2.putText(
                    frame,
                    f"Batch Count: {sessions[session_id]['batches'][batch_id]['batch_count']}",
                    (170, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (57, 255, 0),
                    5
                )

                cv2.putText(frame, f"Camera: {org_url}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (57, 255, 0), 1)

                video_writer.write(frame)
                re = cv2.resize(frame, (1200, 1200))
                cv2.imshow("Detection", re)
            except cv2.error as e:
                logging.error(f"OpenCV error during video processing for session {session_id}: {e}")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        cleanup_session(session_id)

    except Exception as e:
        app_error_logger.error(f"Unexpected error in detection for session {session_id}: {traceback.format_exc()}")
        cleanup_session(session_id)


async def send_email(session_id, supervisor_name, start_time, stop_time, vehicle_number, object_count):
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        username = "support@vchanel.com"
        password = "buqd muuc ylzz ljgr"
        to_emails = ["bindhupumpspacking@gmail.com"]

        subject = "Detection Report - Bindhu"
        body = f"""
        Session ID: {session_id}
        Supervisor: {supervisor_name}
        Vehicle Number: {vehicle_number}
        Start Time: {start_time}
        Stop Time: {stop_time}
        Object Count: {object_count}
        """

        msg = MIMEMultipart()
        msg["From"] = username
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        msg["Bcc"] = ", ".join(to_emails)

        async with asyncio.Lock():
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.sendmail(username, to_emails, msg.as_string())

        logging.info(f"Email sent successfully for session {session_id}")

    except Exception as e:
        logging.error(f"Failed to send email for session {session_id}: {traceback.format_exc()}")
@app.post("/reset")
async def reset_batch(session_id: str):
    """Resets the batch by generating a new batch ID and setting batch count to 0."""

    if session_id not in sessions:
        return {"error": f"Session {session_id} not found."}

    # Generate a new batch ID
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Reset batch count but keep object count
    sessions[session_id]["batches"][batch_id] = {
    "batch_count": 0}
    sessions[session_id]["batches"][batch_id]["batch_count"] = 0
    logging.info(f"Batch reset for session {session_id}, new batch_id: {batch_id}")

    return {
        "message": "Batch reset successful.",
        "session_id": session_id,
        "new_batch_id": batch_id,
        "batch_count": sessions[session_id]["batch_id"]["batch_count"],
        "object_count": sessions[session_id]["object_count"]
    }
@app.post("/start")
async def start_detection(data: DetectionRequest):
    org_url=data.video_url
    full_url =  data.video_url
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("batch _id ------------->",batch_id)
    sessions[data.session_id] = {
    "object_count": 0,
    "supervisor_name": data.supervisor_name,
    "vehicle_number": data.vehicle_number,
    "start_time": start_time,
    "batches": {}
}

    sessions[data.session_id]["batches"][batch_id] = {
    "batch_count": 0
}
    print(sessions[data.session_id]["batches"][batch_id])
    asyncio.create_task(detect_objects(full_url, data.session_id,org_url))
    logging.info(f"Detection started for session {data.session_id}")  # log detection started for particular session

    return {"message": "Detection started.", "session_id": data.session_id}

@app.post("/stop")
async def stop_detection(data: StopRequest):
    session_id = data.session_id
    stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found.")

    session_data = sessions.pop(session_id, None)
    if session_data:
        await send_email(session_id, session_data["supervisor_name"], session_data["start_time"], stop_time, session_data["vehicle_number"], session_data["object_count"])
    logging.info(f"Detection stopped and email sent for session {session_id}")  # session id data log

    return {"message": "Detection stopped and email sent.", "session_id": session_id}
@app.get("/count/{session_id}")
async def get_detection_count(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "detection_count": sessions[session_id]["object_count"]}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)