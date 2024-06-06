import os

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
import shutil
from pydantic import BaseModel
import torch
import os
import asyncio
from vision_codes.frame_extraction import extract_and_organize_frames
from vision_codes.yolov8_train import train_yolov8_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/upload_video/")
async def upload_video(background_tasks: BackgroundTasks, class_name: str= Form(...), file: UploadFile = File(...)):
    logger.info("Received request to upload video")

    # Check the file type
    allowed_extensions = {".mp4", ".avi", ".mkv", ".mov"}
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension not in allowed_extensions:
        logger.error(f"Invalid file type: {file_extension}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only video files are allowed.")

    # Check the file size (limit to 100MB for this example)
    MAX_FILE_SIZE = 1000 * 1024 * 1024  # 1000MB
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0, os.SEEK_SET)
    if file_size > MAX_FILE_SIZE:
        logger.error(f"File size exceeds limit: {file_size} bytes")
        raise HTTPException(status_code=400, detail="File size exceeds the 100MB limit.")

    videos_directory = f"temp_videos/{file.filename}"
    video_path = f"{videos_directory}/{file.filename}"
    if not os.path.exists(videos_directory):
        os.makedirs(videos_directory)

    # Save the file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logger.info(f"Saved video to {video_path}")

    dataset_folder = f"dataset/"
    background_tasks.add_task(extract_and_organize_frames,
                              video_path, dataset_folder,
                                    class_name, save_every_n_frames=5)

    # shutil.rmtree(video_path)

    # Trigger training in the background
    background_tasks.add_task(train_yolov8_model, dataset_folder, epochs=50)

    return {"filename": file.filename, "class_name": class_name}


class ImageRequest(BaseModel):
    image_path: str

@app.get("/train/")
async def train(background_tasks: BackgroundTasks,):
    logger.info(f"Train is started...")
    dataset_folder = f"dataset/"
    # Trigger training in the background
    background_tasks.add_task(train_yolov8_model, dataset_folder, epochs=50)

    return {"Training..."}
