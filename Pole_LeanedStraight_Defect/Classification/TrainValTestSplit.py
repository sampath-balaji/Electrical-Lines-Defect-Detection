import glob
import random
import os
import shutil

# Set the random seed for reproducibility
random.seed(0)

# List all class directories in your original dataset folder
class_dirs = glob.glob('ElectricPoles_Classification_StraightLeaned/*')
print("Found class directories:", class_dirs)  # Debug: list of found directories

if not class_dirs:
    print("No directories found. Check the folder name and path.")

for class_dir in class_dirs:
    # Get all jpg images for the current class
    fnames = glob.glob(os.path.join(class_dir, '*.jpg'))
    print(f"Processing directory: {class_dir}, found {len(fnames)} JPG files.")  # Debug
    
    if not fnames:
        print(f"No JPG images found in {class_dir}.")
        continue
    
    random.shuffle(fnames)
    
    # Define num_files as the number of images in the current class
    num_files = len(fnames)
    
    # Calculate split sizes: 75% train, 15% val, 10% test
    num_train = round(0.75 * num_files)
    num_test = round(0.10 * num_files)
    num_val = num_files - num_train - num_test
    print(f"Splitting {num_files} images into {num_train} train, {num_val} val, {num_test} test.")  # Debug
    
    # Split the file names
    train_fnames = fnames[:num_train]
    test_fnames = fnames[num_train:num_train+num_test]
    val_fnames = fnames[num_train+num_test:]
    
    # Get the class name from the directory name
    class_name = os.path.basename(class_dir)
    
    # Define target directories for split data
    target_train_dir = os.path.join('final_dataset_split', 'train', class_name)
    target_val_dir = os.path.join('final_dataset_split', 'val', class_name)
    target_test_dir = os.path.join('final_dataset_split', 'test', class_name)
    
    # Create target directories if they don't exist
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_val_dir, exist_ok=True)
    os.makedirs(target_test_dir, exist_ok=True)
    
    # Copy images to their respective folders
    for fname in train_fnames:
        shutil.copy2(fname, target_train_dir)
    for fname in val_fnames:
        shutil.copy2(fname, target_val_dir)
    for fname in test_fnames:
        shutil.copy2(fname, target_test_dir)
    print(f"Copied images for class '{class_name}' to train, val, and test directories.\n")
