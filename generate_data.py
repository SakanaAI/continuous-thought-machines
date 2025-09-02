import os
import numpy as np
from PIL import Image, ImageDraw
import json
from tqdm import tqdm

# --- Configuration ---
IMG_SIZE = 64
MIN_RADIUS = 5
MAX_RADIUS = 10
NUM_CIRCLES_TRAIN = 2 # Let's start with the simplest case: always 2 circles
NUM_CIRCLES_VAL = 2
NUM_SAMPLES_TRAIN = 50000
NUM_SAMPLES_VAL = 5000

# Directory to save the generated data
DATA_DIR = 'visual_sorter_data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
# --- End Configuration ---


def check_overlap(c1_x, c1_y, c1_r, existing_circles):
    """
    Checks if a new circle overlaps with any existing circles.
    Adds a small buffer (2 pixels) to ensure they are visually distinct.
    """
    for (c2_x, c2_y, c2_r) in existing_circles:
        distance = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
        if distance < (c1_r + c2_r + 2): # +2 pixel buffer
            return True # Overlap detected
    return False

def generate_single_sample(num_circles, img_size):
    """
    Generates a single image with non-overlapping circles and its corresponding
    sorted coordinate label.
    """
    # Create a blank black image
    image = Image.new('L', (img_size, img_size), 0) # 'L' for 8-bit grayscale
    draw = ImageDraw.Draw(image)

    circles_data = []
    
    # Place circles one by one, ensuring no overlaps
    for _ in range(num_circles):
        attempts = 0
        while attempts < 1000: # Max attempts to prevent infinite loops
            # Generate random properties for a new circle
            radius = np.random.randint(MIN_RADIUS, MAX_RADIUS + 1)
            # Position must account for radius to stay within bounds
            pos_x = np.random.randint(radius, img_size - radius)
            pos_y = np.random.randint(radius, img_size - radius)

            if not check_overlap(pos_x, pos_y, radius, circles_data):
                # If no overlap, add the circle
                circles_data.append((pos_x, pos_y, radius))
                break
            attempts += 1
        
        if attempts >= 1000:
            # Could not place a circle without overlap, this is rare but possible.
            # We can either raise an error or try generating the sample again.
            # For simplicity, we'll just return None and the calling loop will skip it.
            print("Warning: Could not place a circle without overlap. Skipping this sample.")
            return None, None


    # Draw all the successfully placed circles
    for x, y, r in circles_data:
        # The bounding box for the circle is (left, top, right, bottom)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=255) # White circle

    # --- Create the target label ---
    # Sort the list of circles based on radius (smallest to largest)
    # The third element in our tuple is the radius (index 2)
    circles_data.sort(key=lambda c: c[2])

    # Extract just the (x, y) coordinates for the label
    # We also normalize coordinates to be between 0.0 and 1.0, which is good practice
    target_coords = [[x / img_size, y / img_size] for x, y, r in circles_data]
    
    return image, target_coords


def create_dataset(num_samples, num_circles, output_dir):
    """
    Generates and saves a full dataset (images and labels).
    """
    # Create directories if they don't exist
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    all_labels = {}

    print(f"Generating {num_samples} samples for {output_dir}...")
    
    # Use tqdm for a nice progress bar
    for i in tqdm(range(num_samples)):
        image, target = generate_single_sample(num_circles, IMG_SIZE)
        
        if image is not None:
            filename = f'{i:05d}.png'
            image.save(os.path.join(img_dir, filename))
            all_labels[filename] = target

    # Save all labels to a single JSON file
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(all_labels, f)
    
    print(f"Dataset successfully created at {output_dir}")

if __name__ == '__main__':
    # Generate the training dataset
    create_dataset(NUM_SAMPLES_TRAIN, NUM_CIRCLES_TRAIN, TRAIN_DIR)
    
    # Generate the validation dataset
    create_dataset(NUM_SAMPLES_VAL, NUM_CIRCLES_VAL, VAL_DIR)