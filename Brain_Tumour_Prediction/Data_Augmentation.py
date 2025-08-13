import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import join, exists
import time

def hms_string(sec_elapsed):
    """Convert seconds to hours:minutes:seconds format"""
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"

def augment_data(file_dir, n_generated_samples, save_to_dir):
    """Augment data from the given directory and save to output directory"""
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=(0.3, 1.0),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    if not exists(save_to_dir):
        makedirs(save_to_dir)

    for filename in listdir(file_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = join(file_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read {image_path}")
                continue

            image = image.reshape((1,) + image.shape)
            save_prefix = 'aug_' + filename.split('.')[0]

            i = 0
            for batch in data_gen.flow(
                x=image, 
                batch_size=1,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format='jpg'
            ):
                i += 1
                if i > n_generated_samples:
                    break

def data_summary(main_path):
    """Print summary of the dataset in the given path"""
    yes_path = join(main_path, 'yes')
    no_path = join(main_path, 'no')

    m_pos = len(listdir(yes_path)) if exists(yes_path) else 0
    m_neg = len(listdir(no_path)) if exists(no_path) else 0
    m = m_pos + m_neg

    if m == 0:
        print("No data found in Augmented folder.")
        return

    pos_prec = (m_pos * 100.0) / m
    neg_prec = (m_neg * 100.0) / m

    print(f"Total examples: {m}")
    print(f"Tumorous (yes): {m_pos} ({pos_prec:.2f}%)")
    print(f"Non-tumorous (no): {m_neg} ({neg_prec:.2f}%)")

def main():
    start_time = time.time()

    # Input data paths
    yes_input_path = join("Data", "yes")
    no_input_path = join("Data", "no")

    # Output augmented data paths
    augmented_path = "Augmented"
    yes_augmented_path = join(augmented_path, "yes")
    no_augmented_path = join(augmented_path, "no")

    # Perform data augmentation
    print("Starting data augmentation...")
    augment_data(file_dir=yes_input_path, n_generated_samples=6, save_to_dir=yes_augmented_path)
    augment_data(file_dir=no_input_path, n_generated_samples=9, save_to_dir=no_augmented_path)

    end_time = time.time()
    print(f"\nElapsed time: {hms_string(end_time - start_time)}")

    # Display summary
    print("\nAugmented Data Summary:")
    data_summary(augmented_path)

if __name__ == "__main__":
    main()
