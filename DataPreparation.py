import os
import argparse
import random
import shutil
import zipfile
import requests
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance, ImageOps


def download_sample_dataset(download_dir='datasets'):
    """
    Download a sample space object dataset

    Args:
        download_dir (str): Directory to download the dataset to
    """
    # You can replace this URL with a real dataset URL
    # For demonstration, we're using a placeholder
    url = "https://example.com/space_object_dataset.zip"  # Replace with actual dataset URL

    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Define the local file path
    local_file = os.path.join(download_dir, "space_object_dataset.zip")

    print(f"Downloading space object dataset to {local_file}...")

    try:
        # Download the file
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(local_file, 'wb') as f:
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
            progress_bar.close()

        # Extract the zip file
        with zipfile.ZipFile(local_file, 'r') as zip_ref:
            zip_ref.extractall(download_dir)

        print(f"Dataset downloaded and extracted to {download_dir}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download a dataset manually and place it in the appropriate directory.")


def augment_image(image, save_path):
    """
    Apply various augmentations to an image and save it

    Args:
        image (PIL.Image): Image to augment
        save_path (str): Path to save the augmented image
    """
    # Randomly select an augmentation technique
    augmentation = random.choice(['rotate', 'flip', 'brightness', 'contrast', 'crop'])

    if augmentation == 'rotate':
        # Rotate by a random angle
        angle = random.uniform(-30, 30)
        augmented = image.rotate(angle, resample=Image.BICUBIC, expand=False)

    elif augmentation == 'flip':
        # Horizontal flip
        augmented = ImageOps.mirror(image)

    elif augmentation == 'brightness':
        # Adjust brightness
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        augmented = enhancer.enhance(factor)

    elif augmentation == 'contrast':
        # Adjust contrast
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        augmented = enhancer.enhance(factor)

    elif augmentation == 'crop':
        # Random crop and resize
        width, height = image.size
        crop_size = min(width, height) * random.uniform(0.8, 0.9)
        left = random.uniform(0, width - crop_size)
        top = random.uniform(0, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size

        augmented = image.crop((left, top, right, bottom))
        augmented = augmented.resize((width, height), Image.BICUBIC)

    # Save the augmented image
    augmented.save(save_path)


def augment_dataset(data_dir, augmentation_factor=2):
    """
    Augment a dataset by creating new images through transformations

    Args:
        data_dir (str): Directory containing the dataset
        augmentation_factor (int): Number of augmented images to create per original image
    """
    print(f"Augmenting dataset in {data_dir} with factor {augmentation_factor}...")

    # Loop through each class directory
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue

        # Get list of image files
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Augmenting {len(image_files)} images for class '{class_name}'...")

        # Create augmented images
        for img_file in tqdm(image_files):
            img_path = os.path.join(class_dir, img_file)

            try:
                # Open the image
                image = Image.open(img_path).convert('RGB')

                # Create augmented versions
                for i in range(augmentation_factor):
                    # Generate new filename
                    base_name, ext = os.path.splitext(img_file)
                    new_filename = f"{base_name}_aug_{i + 1}{ext}"
                    save_path = os.path.join(class_dir, new_filename)

                    # Augment and save
                    augment_image(image, save_path)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")


def organize_dataset(src_dir, dest_dir, test_split=0.2):
    """
    Organize a dataset into training and testing sets

    Args:
        src_dir (str): Source directory containing the dataset
        dest_dir (str): Destination directory for the organized dataset
        test_split (float): Fraction of data to use for testing
    """
    print(f"Organizing dataset from {src_dir} to {dest_dir} with test split {test_split}...")

    # Create destination directories
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Loop through each class directory
    for class_name in os.listdir(src_dir):
        class_dir = os.path.join(src_dir, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue

        # Create class directories in train and test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Get list of image files
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Split into train and test
        train_files, test_files = train_test_split(image_files, test_size=test_split, random_state=42)

        print(f"Class '{class_name}': {len(train_files)} training, {len(test_files)} testing")

        # Copy training files
        for img_file in tqdm(train_files, desc=f"Copying {class_name} training files"):
            src_path = os.path.join(class_dir, img_file)
            dest_path = os.path.join(train_class_dir, img_file)
            shutil.copy2(src_path, dest_path)

        # Copy testing files
        for img_file in tqdm(test_files, desc=f"Copying {class_name} testing files"):
            src_path = os.path.join(class_dir, img_file)
            dest_path = os.path.join(test_class_dir, img_file)
            shutil.copy2(src_path, dest_path)

    print(f"Dataset organized into {train_dir} and {test_dir}")


def check_image_stats(data_dir):
    """
    Check and print statistics about the images in the dataset

    Args:
        data_dir (str): Directory containing the dataset
    """
    print(f"Checking image statistics in {data_dir}...")

    class_counts = {}
    image_sizes = []
    corrupt_images = []

    # Loop through each class directory
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue

        # Get list of image files
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[class_name] = len(image_files)

        # Check image sizes and validity
        for img_file in tqdm(image_files, desc=f"Checking {class_name} images"):
            img_path = os.path.join(class_dir, img_file)

            try:
                # Open the image
                with Image.open(img_path) as img:
                    width, height = img.size
                    image_sizes.append((width, height))
            except Exception as e:
                corrupt_images.append((img_path, str(e)))

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total classes: {len(class_counts)}")

    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")

    if image_sizes:
        widths, heights = zip(*image_sizes)
        print("\nImage size statistics:")
        print(f"  Width range: {min(widths)} to {max(widths)} pixels")
        print(f"  Height range: {min(heights)} to {max(heights)} pixels")
        print(f"  Most common dimensions: {max(set(image_sizes), key=image_sizes.count)}")

    if corrupt_images:
        print("\nCorrupt images found:")
        for path, error in corrupt_images[:10]:
            print(f"  {path}: {error}")

        if len(corrupt_images) > 10:
            print(f"  ... and {len(corrupt_images) - 10} more")


def normalize_image_sizes(data_dir, target_size=(224, 224)):
    """
    Normalize all images to a standard size

    Args:
        data_dir (str): Directory containing the dataset
        target_size (tuple): Target (width, height) for all images
    """
    print(f"Normalizing images in {data_dir} to size {target_size}...")

    # Loop through each class directory
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue

        # Get list of image files
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Resize each image
        for img_file in tqdm(image_files, desc=f"Resizing {class_name} images"):
            img_path = os.path.join(class_dir, img_file)

            try:
                # Open, resize, and save the image
                with Image.open(img_path) as img:
                    img = img.convert('RGB')  # Convert to RGB
                    resized_img = img.resize(target_size, Image.BICUBIC)
                    resized_img.save(img_path)
            except Exception as e:
                print(f"Error resizing {img_path}: {e}")


def parse_arguments():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Space Object Image Dataset Preparation')
    parser.add_argument('--download', action='store_true', help='Download a sample dataset')
    parser.add_argument('--augment', action='store_true', help='Augment the dataset')
    parser.add_argument('--organize', action='store_true', help='Organize dataset into train/test splits')
    parser.add_argument('--check', action='store_true', help='Check dataset statistics')
    parser.add_argument('--normalize', action='store_true', help='Normalize image sizes')
    parser.add_argument('--augmentation_factor', type=int, default=2, help='Number of augmented images per original')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--src_dir', type=str, default='datasets/raw', help='Source directory')
    parser.add_argument('--dest_dir', type=str, default='datasets/processed', help='Destination directory')

    return parser.parse_args()


def main():
    """
    Main function to prepare the dataset
    """
    args = parse_arguments()

    # Download a sample dataset
    if args.download:
        download_sample_dataset()

    # Check dataset statistics
    if args.check:
        check_image_stats(args.src_dir)

    # Normalize image sizes
    if args.normalize:
        normalize_image_sizes(args.src_dir)

    # Augment the dataset
    if args.augment:
        augment_dataset(args.src_dir, args.augmentation_factor)

    # Organize dataset into train/test splits
    if args.organize:
        organize_dataset(args.src_dir, args.dest_dir, args.test_split)

    # If no options selected, show help
    if not any([args.download, args.augment, args.organize, args.check, args.normalize]):
        print("No actions selected. Use --help to see available options.")
        print("\nExample commands:")
        print("1. Check the dataset: python data_preparation.py --check --src_dir path/to/dataset")
        print("2. Normalize images: python data_preparation.py --normalize --src_dir path/to/dataset")
        print(
            "3. Augment the dataset: python data_preparation.py --augment --src_dir path/to/dataset --augmentation_factor 3")
        print(
            "4. Organize into train/test: python data_preparation.py --organize --src_dir path/to/dataset --dest_dir path/to/organized --test_split 0.2")
        print("5. Complete pipeline: python data_preparation.py --check --normalize --augment