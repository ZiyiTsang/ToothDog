import os
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import ToothSegmentationModel, TOOTH_ID_TO_LABEL
from dataModule import get_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ToothInference:
    """Tooth segmentation inference class for specific tooth ID extraction"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize inference model
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Load model from checkpoint
        self.model = ToothSegmentationModel.load_from_checkpoint(
            checkpoint_path, 
            map_location=device
        )
        self.model.eval()
        self.model.to(device)
        
        # Get transformation for inference
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model name: {self.model.model_name}")
        print(f"Number of classes: {self.model.hparams.num_classes}")
    
    def preprocess_image(self, image_path):
        """Preprocess input image for inference"""
        # Read and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        augmented = self.transform(image=image)
        image_tensor = augmented['image']
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def tooth_id_to_class_index(self, tooth_id):
        """Convert tooth ID string to class index"""
        if tooth_id not in TOOTH_ID_TO_LABEL:
            raise ValueError(f"Tooth ID {tooth_id} not found in TOOTH_ID_TO_LABEL mapping")
        
        return TOOTH_ID_TO_LABEL[tooth_id]
    
    def inference(self, input_image, tooth_id):
        """
        Perform inference to extract specific tooth region
        
        Args:
            input_image: Input image tensor [1, 3, H, W]
            tooth_id: Target tooth ID string (e.g., "11", "12", etc.)
        
        Returns:
            mask_image: Binary mask for the specified tooth [H, W]
        """
        with torch.no_grad():
            # Get the class index for the target tooth ID
            target_class = self.tooth_id_to_class_index(tooth_id)
            
            # Move input to device
            input_image = input_image.to(self.device)
            
            # Forward pass - note: tooth_id parameter is not used in current model
            # but we pass it for compatibility
            outputs = self.model(input_image, tooth_id)
            
            # Get class probabilities
            probs = F.softmax(outputs, dim=1)  # [1, num_classes, H, W]
            
            # Extract probability for the target class
            target_prob = probs[0, target_class]  # [H, W]
            
            # Convert to binary mask using threshold
            mask = (target_prob > 0.5).float()
            
            return mask.cpu().numpy()
    
    def visualize_result(self, original_image, mask_image, tooth_id, save_path=None):
        """Visualize inference result"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image\nTarget Tooth: {tooth_id}')
        axes[0].axis('off')
        
        # Binary mask
        axes[1].imshow(mask_image, cmap='gray')
        axes[1].set_title(f'Predicted Mask\nTooth {tooth_id}')
        axes[1].axis('off')
        
        # Overlay
        overlay = original_image.copy()
        overlay[mask_image > 0.5] = [255, 0, 0]  # Red overlay
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\nTooth {tooth_id}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Result saved to {save_path}")
        
        plt.show()
    
    def process_single_image(self, image_path, tooth_id, output_dir=None):
        """Process a single image and return the mask"""
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Read original image for visualization
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        mask = self.inference(image_tensor, tooth_id)
        
        # Resize mask to match original image size
        mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(output_dir, f"{base_name}_tooth{tooth_id}_result.png")
            self.visualize_result(original_image, mask_resized, tooth_id, save_path)
        else:
            self.visualize_result(original_image, mask_resized, tooth_id)
        
        return mask_resized

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Tooth Segmentation Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--tooth_id', type=str, required=True,
                       help='Target tooth ID (e.g., 11, 12, 13, etc.)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save results (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize inference model
    inference_model = ToothInference(args.checkpoint, args.device)
    
    # Process the image
    mask = inference_model.process_single_image(
        args.image, 
        args.tooth_id, 
        args.output_dir
    )
    
    print(f"Inference completed for tooth {args.tooth_id}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask values range: [{mask.min()}, {mask.max()}]")

if __name__ == "__main__":
    main()