import shutil
from pathlib import Path
import cv2
import os
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_dataset(all_files, train_ratio=0.7, val_ratio=0.2):
    logger.info("Shuffling and splitting dataset")
    random.shuffle(all_files)
    total_files = len(all_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    return train_files, val_files, test_files


def save_frames_to_directory(files, directory):
    logger.info(f"Saving {len(files)} frames to {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file_path in files:
        filename = os.path.basename(file_path)
        destination_path = os.path.join(directory, filename)
        shutil.move(file_path, destination_path)
    logger.info(f"Finished saving frames to {directory}")


def extract_and_organize_frames(video_path, output_folder, class_name, save_every_n_frames=5):
    logger.info(f"Extracting frames from {video_path} for class {class_name}")
    temp_output_folder = f"temp_frames/{class_name}"
    if not os.path.exists(temp_output_folder):
        os.makedirs(temp_output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0
    success = True

    while success:
        success, image = cap.read()
        if not success:
            break
        if frame_count % save_every_n_frames == 0:
            frame_path = os.path.join(temp_output_folder, f"{class_name}_{frame_count}.jpg")
            cv2.imwrite(frame_path, image)
            count += 1
        frame_count += 1

    cap.release()
    logger.info(f"Extracted {count} frames for class {class_name}")

    all_files = list(Path(temp_output_folder).glob("*.jpg"))
    train_files, val_files, test_files = split_dataset(all_files)

    save_frames_to_directory(train_files, os.path.join(output_folder, 'train', class_name))
    save_frames_to_directory(val_files, os.path.join(output_folder, 'val', class_name))
    save_frames_to_directory(test_files, os.path.join(output_folder, 'test', class_name))

    shutil.rmtree(temp_output_folder)
    logger.info(f"Organized frames into train, val, and test folders for class {class_name}")
