import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class VisualSorterDataset(Dataset):
    """
    PyTorch Dataset for the Visual Sorter project.
    Loads images and their corresponding sorted coordinate labels.
    """
    def __init__(self, data_dir, max_seq_len=10):
        """
        Args:
            data_dir (string): Directory with all the images and the labels.json file.
            max_seq_len (int): The maximum number of circles to pad the sequence to.
                               This ensures all label tensors have the same size.
        """
        self.img_dir = os.path.join(data_dir, 'images')
        self.labels_path = os.path.join(data_dir, 'labels.json')
        self.max_seq_len = max_seq_len
        
        # Load the entire set of labels from the JSON file
        with open(self.labels_path, 'r') as f:
            self.labels_data = json.load(f)
            
        # Create a list of image filenames, which we can index
        self.image_files = list(self.labels_data.keys())
        
        # Define the image transformations
        # 1. Convert to grayscale (although it already is, this ensures 1 channel)
        # 2. Convert the PIL Image to a PyTorch Tensor
        # 3. Normalize pixel values from [0, 255] to [0.0, 1.0]
        #    (ToTensor() does this automatically if the input is a PIL Image)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Args:
            idx (int): The index of the sample to fetch.
            
        Returns:
            A tuple: (image_tensor, label_tensor, sequence_length)
        """
        # Get the image filename for the given index
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load the image
        image = Image.open(img_path).convert('L')
        
        # Apply transformations to the image
        image_tensor = self.transform(image)
        
        # --- Prepare the label ---
        # Get the coordinates from our loaded data
        coords = self.labels_data[img_name]
        
        # Get the actual number of circles in this specific sample
        sequence_length = len(coords)
        
        # Create a padded tensor for the labels
        # We'll use a special padding value, e.g., -1.0, that is outside the
        # normalized coordinate range [0.0, 1.0]. The model can learn to ignore this.
        label_tensor = torch.full((self.max_seq_len, 2), -1.0, dtype=torch.float32)
        
        # Fill the tensor with the actual coordinates
        if sequence_length > 0:
            label_tensor[:sequence_length, :] = torch.tensor(coords, dtype=torch.float32)

        return image_tensor, label_tensor, sequence_length

# --- Example Usage (for testing the dataset class) ---
if __name__ == '__main__':
    # Make sure you have run generate_data.py first!
    DATA_DIR = 'visual_sorter_data'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    
    print("Testing the VisualSorterDataset...")
    
    try:
        # Create an instance of the dataset for the training data
        train_dataset = VisualSorterDataset(data_dir=TRAIN_DIR)

        # Check the length
        print(f"Found {len(train_dataset)} samples in the training set.")

        # Get the first sample
        image, label, seq_len = train_dataset[0]

        # Print out the shapes and types to verify everything is correct
        print("\n--- Sample 0 ---")
        print(f"Image tensor shape: {image.shape}")
        print(f"Image tensor dtype: {image.dtype}")
        print(f"Label tensor shape: {label.shape}")
        print(f"Label tensor dtype: {label.dtype}")
        print(f"Actual sequence length: {seq_len}")
        print("Label tensor content (first 5 elements):")
        print(label[:5])

        # Example of how you would use it with a DataLoader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # Get one batch
        image_batch, label_batch, seq_len_batch = next(iter(train_loader))
        print("\n--- DataLoader Batch ---")
        print(f"Image batch shape: {image_batch.shape}") # Should be [4, 1, 64, 64]
        print(f"Label batch shape: {label_batch.shape}") # Should be [4, 10, 2]
        print(f"Seq len batch: {seq_len_batch}") # Should be a tensor of 4 numbers, e.g., [2, 2, 2, 2]

    except FileNotFoundError:
        print("\nError: Data not found.")
        print("Please run `python generate_data.py` before running this test.")