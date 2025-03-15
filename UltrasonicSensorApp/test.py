import os
import sys
import shutil
import random

def rename_files_in_folder(folder_path, base_name="fft_occupant"):
    """
    Renames all files in the given folder sequentially to:
    fft_nooccupant_1.ext, fft_nooccupant_2.ext, etc.
    
    Each time the function is called, it starts the counter at 1 and renames every file.
    The folder_path should be the complete folder path.
    """
    # Get list of files (ignore directories)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  # sort to ensure a consistent order

    counter = 1
    for filename in files:
        name, ext = os.path.splitext(filename)
        new_name = f"{base_name}_{counter}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        try:
            os.rename(src, dst)
            print(f"Renamed {filename} to {new_name}")
        except Exception as e:
            print(f"Error renaming {filename}: {e}")
        counter += 1


def copy_one_per_size(source_folder, dest_folder):
    """
    Iterates through all files in the source folder and groups them by file size.
    For each group (even if there are duplicates), copies exactly one file to the destination folder.
    """
    # Ensure destination folder exists.
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Build a dictionary: file size -> list of filenames with that size.
    size_dict = {}
    for filename in os.listdir(source_folder):
        src_path = os.path.join(source_folder, filename)
        if os.path.isfile(src_path):
            file_size = os.path.getsize(src_path)
            size_dict.setdefault(file_size, []).append(filename)

    # Iterate over the groups, copying one file per size group.
    for file_size, file_list in size_dict.items():
        # Copy the first file in the group.
        filename = file_list[0]
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(dest_folder, filename)
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename} (size: {file_size} bytes) from group of {len(file_list)}")
        except Exception as e:
            print(f"Error copying {filename}: {e}")

def duplicate_files_randomly(source_folder, dest_folder, total_files):
    """
    Duplicates files from the source folder into the destination folder until the total number of files
    in the destination folder reaches the specified total_files count. The duplicated files are renamed 
    uniquely to ensure no filename conflicts.

    :param source_folder: The path to the source directory containing the original files.
    :param dest_folder: The path to the destination directory where files will be duplicated.
    :param total_files: The total number of files desired in the destination directory.
    """
    # Ensure the destination folder exists.
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # List all files in the source folder.
    source_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    if not source_files:
        print("No files found in the source folder.")
        return

    current_files = len(source_files)
    copies_needed = total_files - current_files

    if copies_needed <= 0:
        print("No need for additional files, the current number is already sufficient.")
        return

    # Loop to create duplicates until the total_files count is reached.
    while copies_needed > 0:
        # Randomly select a file to duplicate.
        file_to_copy = random.choice(source_files)
        name, ext = os.path.splitext(file_to_copy)

        # Generate a unique new filename.
        unique_suffix = 0
        new_filename = f"{name}_{unique_suffix}{ext}"
        while new_filename in source_files or os.path.exists(os.path.join(dest_folder, new_filename)):
            unique_suffix += 1
            new_filename = f"{name}_{unique_suffix}{ext}"

        # Copy and rename the file.
        src_path = os.path.join(source_folder, file_to_copy)
        dst_path = os.path.join(dest_folder, new_filename)
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Copied and renamed {file_to_copy} to {new_filename}")
            copies_needed -= 1
            source_files.append(new_filename)  # Add to the list to potentially use for further copying.
        except Exception as e:
            print(f"Error copying {file_to_copy}: {e}")

def count_lines_and_files(folder_path):
    """
    Recursively counts the lines in all .txt files and the number of .txt files starting from the given folder path.

    :param folder_path: The root directory from which to start searching for .txt files.
    :return: A tuple containing the total number of lines across all .txt files and the total number of .txt files read.
    """
    total_lines = 0
    total_files = 0

    # Walk through all directories and files in the folder.
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is a .txt file.
            if file.endswith('.txt'):
                total_files += 1  # Increment the .txt file counter.
                file_path = os.path.join(root, file)
                try:
                    # Open the .txt file and count its lines.
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        total_lines += line_count
                        # print(f"Found {line_count} lines in {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return total_lines, total_files



if __name__ == '__main__':

    #folder_path = 'E:/Frankfurt University of Applied Sciences/Master Thesis/GitHub/Coding/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/15.03.2025/Soft'

    #rename_files_in_folder(folder_path)

    source_folder = 'E:/Frankfurt University of Applied Sciences/Master Thesis/GitHub/Coding/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/Data-Set 5/Soft'
    dest_folder = 'E:/Frankfurt University of Applied Sciences/Master Thesis/GitHub/Coding/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/Data-Set 3/New folder'
    #copy_one_per_size(source_folder, dest_folder)

    # rename_files_in_folder(source_folder)

    #total_files = 130  # Set the total desired files in the destination.
    
    #duplicate_files_randomly(source_folder, dest_folder, total_files)

    source = 'E:/Frankfurt University of Applied Sciences/Master Thesis/GitHub/Coding/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data'
    total_lines, total_files = count_lines_and_files(source)
    print(f"Total number of lines across all .txt files: {total_lines}")
    print(f"Total number of .txt files read: {total_files}")

