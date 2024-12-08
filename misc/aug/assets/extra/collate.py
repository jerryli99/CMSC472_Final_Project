import os
import shutil

# Define the source and destination directories
source_dir = './'
destination_dir = os.path.join(source_dir, 'total')
print("Target directory: ", destination_dir)

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate through all folders in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)

    # Check if it is a directory
    if os.path.isdir(folder_path):
        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Copy each file to the destination directory if the file is not already there
            if os.path.isfile(file_path) and not os.path.exists(os.path.join(destination_dir, file_name)):
                shutil.copy(file_path, destination_dir)