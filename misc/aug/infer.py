import torch
from PIL import Image
from torchvision import transforms
import os
from core.model import Generator

# check for cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model checkpoint
checkpoint_path = '100000_nets_ema.ckpt'
model = Generator()
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256, adjust based on model input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization
])

# Define output directory for saving results
output_dir = 'output_images/'
os.makedirs(output_dir, exist_ok=True)

# Process each image in the 'test/' folder
test_dir = 'test/'
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        # Load and preprocess the image
        image_path = os.path.join(test_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():  # No gradient tracking during inference
            output_image = model(image)  # Run the image through the model

        # Convert output tensor back to PIL image
        output_image = output_image.squeeze(0).cpu()  # Remove batch dimension
        output_image = transforms.ToPILImage()(output_image)

        # Save the output image
        output_image_path = os.path.join(output_dir, f'output_{filename}')
        output_image.save(output_image_path)
        print(f"Processed and saved: {output_image_path}")
