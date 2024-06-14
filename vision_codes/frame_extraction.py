import shutil
from pathlib import Path
import cv2
import os
import random
import logging
from ruamel.yaml import YAML

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_dataset(all_files, train_ratio=0.7, val_ratio=0.2):
    logger.info("Shuffling and splitting dataset")
    random.shuffle(all_files)
    total_files = len(all_files)
    print(total_files)
    train_end = int(total_files * train_ratio)
    print(train_end)
    val_end = train_end + int(total_files * val_ratio)
    print(val_end)
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    print(test_files)
    print(train_files)
    print(val_files)
    return train_files, val_files, test_files


def save_frames_to_directory(temp_folder, files, directory, split_name):
    logger.info(f"Saving {len(files)} frames to {directory}")
    print(files)
    os.makedirs(f"{directory}images/{split_name}", exist_ok=True)
    os.makedirs(f"{directory}labels/{split_name}", exist_ok=True)

    for file_path in files:
        print(file_path)
        base_name = file_path.split(".txt")[0]
        destination_path_jpg = f"{directory}images\{split_name}\{base_name}.jpg"
        destination_path_txt = f"{directory}labels\{split_name}\{base_name}.txt"
        print(destination_path_txt)
        print(destination_path_jpg)
        print(f"{temp_folder}{file_path.split('.txt')[0]}.jpg")
        print(f"{temp_folder}{file_path}")
        try:
            os.path.isfile(f"./{temp_folder}{file_path.split('.txt')[0]}.jpg")
            shutil.move(f"./{temp_folder}{file_path.split('.txt')[0]}.jpg", destination_path_jpg)
            shutil.move(f"./{temp_folder}{file_path}", destination_path_txt)
            logger.info(f"Moved {file_path} to {directory}")
        except Exception as e:
            logger.error(f"Error moving {temp_folder}{file_path.split('.txt')[0]}.jpg: {e}")
    logger.info(f"Finished saving frames to {directory}")


def update_classes_file(classes_file_path, new_classes, config_file_path):
    # Read existing classes from the file
    try:
        with open(classes_file_path, 'r') as file:
            existing_classes = file.read().splitlines()
    except FileNotFoundError:
        existing_classes = []

    # Find classes that are not already in the existing classes
    classes_to_add = [cls for cls in new_classes if cls.lower() not in existing_classes and cls]

    # If there are classes to add, append them to the file
    if classes_to_add:
        with open(classes_file_path, 'a') as file:
            for cls in classes_to_add:
                file.write(f"{cls}\n")

    # Update the config.yaml file
    yaml = YAML()
    with open(config_file_path, 'r') as file:
        config = yaml.load(file)

    # Get current classes in the config file
    config_classes = config['names']
    config_classes_lower = {v.lower(): k for k, v in config_classes.items()}

    # Add new classes to the config file
    max_index = max(config_classes.keys()) if config_classes else -1
    for cls in new_classes:
        if cls.lower() not in config_classes_lower and cls:
            max_index += 1
            config_classes[max_index] = cls

    # Write the updated config back to the file
    with open(config_file_path, 'w') as file:
        yaml.dump(config, file)


def list_txt_files(src_folder):

    moved_txt_files = []

    # Iterate over all files in the source folder
    for file_name in os.listdir(src_folder):
        # Check if the file is a .txt file
        if file_name.endswith('.txt'):
            # Add the moved .txt file name to the list
            moved_txt_files.append(file_name)

    return moved_txt_files


def extract_and_organize_frames(video_path, raw_output_folder, class_names=None, save_every_n_frames=15):
    logger.info(f"Extracting frames from {video_path}")

    temp_output_folder = f"temp_frames/"
    classes_file_path = r'datasets/classes.txt'
    config_file_path = r'datasets/langlearn_dataset_config.yaml'

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
            frame_path = os.path.join(temp_output_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, image)
            count += 1
        frame_count += 1

    cap.release()
    logger.info(f"Extracted {count} frames.")

    update_classes_file(classes_file_path, class_names, config_file_path)

    os.system(f"python ../labelImg/labelimg.py {temp_output_folder} {classes_file_path}")

    logger.info(f"Finished annotating frames from {video_path}")
    logger.info(f"Start moving frames to {raw_output_folder}")

    all_files = list_txt_files(temp_output_folder)
    print(all_files)
    train_files, val_files, test_files = split_dataset(all_files)
    #
    save_frames_to_directory(temp_output_folder, train_files, raw_output_folder, 'train')
    save_frames_to_directory(temp_output_folder, val_files, raw_output_folder, 'val')
    save_frames_to_directory(temp_output_folder, test_files, raw_output_folder, 'test')
    #
    # shutil.rmtree(temp_output_folder)
    logger.info(f"Organized frames into train, val, and test folders")
