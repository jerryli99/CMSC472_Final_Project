"""
File structure of total:
total
-> {name}_{rest}

for each file

We want to restructure it so that its 

test
-> {name}
    -> {rest}
"""
import os
import shutil
import re
import random
from tqdm import tqdm

# List of names to match
name_list = {
    "art_sociology": "art_sociology_building_test",
    "atlantic_building": "atlantic_building_test",
    "brendan_iribe_center": "brendan_iribe_center_test",
    "esj": "esj_building_test",
    "farm_building": "farm_building_test",
    "mckeldinlib": "mckeldinlib_building_test",
    "physics_building": "physics_building_test",
    "prince_frederick": "prince_frederick_building_test",
    "reckord_armory": "reckord_armory_test",
    "regents_drive": "regents_drive_parking_garage_test",
    "yahentamitsi": "yahentamitsi_dinning_hall_test",
    "denton": "denton_test",
    "elkton": "elkton_test",
    "ellicott": "ellicott_test",
    "hagerstown": "hagerstown_test",
    "james_clark": "james_clark_test",
    "laplata": "laplata_test",
    "manufacture": "manufacture_test",
    "oakland": "oakland_test",
    "recreation": "recreation_test",
}
# Directory containing all folders to be reorganized
base_dir = 'total'
new_base_dir = "."

# Iterate over all directories in the base directory
for folder in tqdm(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder)
    
    # Check if it's a folder
    if os.path.isdir(folder_path):
        # Use regex to match the folder name with any of the names in the list
        match = None
        for name in name_list:
            if re.match(f"^{name}_.+$", folder):
                match = name
                break
        
        # If a match is found, restructure the folder
        if match:
            dest_name = name_list[match]
            # Extract the "rest" part of the folder name
            rest = folder # [len(match) + 1:]  # Skip the name and underscore

            # Create the new structure: 'test/{name}'
            new_dir = f'{new_base_dir}/test/{dest_name}'
            
            # Create the necessary directories
            os.makedirs(new_dir, exist_ok=True)

            # choose 2 random numbers from 0,1,2,3,4
            matches = [f'rain_{str(x)}' for x in random.sample(range(5), 2)]
            matches += ["Clear"]

            # Move contents from old folder to the new folder
            for item in os.listdir(folder_path):
                for match in matches:
                    if match in str(item):
                        old_item_path = f'{folder_path}/{item}'
                        new_item_path = f'{new_dir}/{rest}_{item}'
                        print(f"Moving {old_item_path} to {new_item_path}")
                        shutil.move(old_item_path, new_item_path)
