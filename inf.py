import torch
import numpy as np
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from einops import rearrange
import cv2
import os
from data import preprocess, prepare, apply_median_filter, apply_guassain_filter, img_enhance

# Define the panc_sam class (same as in training code)
class panc_sam(torch.nn.Module):
    def __init__(self, model_type="vit_h", checkpoint=None):
        super().__init__()
        from segment_anything import sam_model_registry
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.prompt_encoder = self.sam.prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        with torch.no_grad():
            image_embedding = self.sam.image_encoder(image)
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=box, masks=None
            )
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        return low_res_masks

# Function to save the output mask
def save_mask(mask, output_path):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255  # Convert to binary mask
    mask_img = Image.fromarray(mask)
    mask_img.save(output_path)

# Inference function
def infer_single_image(image_path, model_path, output_path, device="cuda:0", image_size=1024):
    # Load model
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    # Load and preprocess image
    if image_path.endswith(".png") or image_path.endswith(".jpg"):
        data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        data = np.load(image_path)
    x = rearrange(data, "H W -> 1 H W")
    x = torch.tensor(x, dtype=torch.float32)

    # Apply preprocessing
    x = preprocess(x)
    x = prepare(x, image_size)
    x = x.to(device)

    # Define bounding box (same as in training)
    box = torch.tensor([[200, 200, 750, 800]], dtype=torch.float32).to(device)

    # Run inference
    with torch.no_grad():
        low_res_masks = model(x, box)

    # Save output mask
    save_mask(low_res_masks, output_path)
    print(f"Output mask saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = "/home/dev/workspace/Abolfazl/Dev/de/samples/1/img.png"  # Replace with your image path
    model_path = "exps/0-your_experiment/sam_tuned_save.pth"  # Replace with your model checkpoint path
    output_path = "output_mask.png"  # Replace with desired output path
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    infer_single_image(image_path, model_path, output_path, device=device)