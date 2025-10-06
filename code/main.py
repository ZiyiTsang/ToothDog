#!/usr/bin/env python3
"""
Main training script for tooth segmentation with Comet ML integration
Supports multiple pretrained models and handles device compatibility
"""

import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import CometLogger
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model import ToothSegmentationModel
from dataModule import ToothDataModule


def setup_comet_logger(api_key=None, project_name="tooth-segmentation", experiment_name=None):
    """
    Setup Comet ML logger for experiment tracking
    
    Args:
        api_key: Comet ML API key (if None, will look for environment variable)
        project_name: Project name in Comet ML
        experiment_name: Experiment name (if None, will be auto-generated)
    
    Returns:
        CometLogger instance
    """
    try:
        logger = CometLogger(
            api_key=api_key,
            project_name=project_name,
            experiment_name=experiment_name,
            save_dir="./comet_logs"
        )
        print("✅ Comet ML logger initialized successfully")
        return logger
    except Exception as e:
        print(f"⚠️  Comet ML initialization failed: {e}")
        print("Continuing without Comet ML logging...")
        return None


def get_available_models():
    """
    Return available pretrained segmentation models
    
    Returns:
        dict: Model names and their corresponding torchvision functions
    """
    return {
        "deeplabv3_resnet50": "deeplabv3_resnet50",
        "deeplabv3_resnet101": "deeplabv3_resnet101", 
        "fcn_resnet50": "fcn_resnet50",
        "fcn_resnet101": "fcn_resnet101",
        "lraspp_mobilenet_v3_large": "lraspp_mobilenet_v3_large"
    }


def create_model(model_name, num_classes=1, learning_rate=1e-4, tooth_ids=None):
    """
    Create segmentation model with specified architecture
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes (1 for binary segmentation)
        learning_rate: Learning rate for training
        tooth_ids: List of tooth IDs to segment
    
    Returns:
        ToothSegmentationModel instance
    """
    available_models = get_available_models()
    
    if model_name not in available_models:
        raise ValueError(f"Model {model_name} not available. Available models: {list(available_models.keys())}")
    
    print(f"🔄 Creating {model_name} model...")
    return ToothSegmentationModel(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=learning_rate,
        tooth_ids=tooth_ids
    )


def setup_data_module(data_dir, batch_size, num_workers, tooth_ids=None):
    """
    Setup data module for training
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        tooth_ids: List of tooth IDs to include
    
    Returns:
        ToothDataModule instance
    """
    print("🔄 Setting up data module...")
    return ToothDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        tooth_ids=tooth_ids
    )


class ImageLoggingCallback(Callback):
    """Callback to log images to Comet ML every epoch"""
    
    def __init__(self, log_every_n_epochs=1, num_samples=4):
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log images at the end of each validation epoch"""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
            
        if trainer.logger is None:
            return
            
        # Get validation dataloader - handle both single dataloader and list
        val_dataloader = trainer.val_dataloaders
        if isinstance(val_dataloader, list):
            val_dataloader = val_dataloader[0]
        
        # Get first batch
        for batch_idx, (images, masks, tooth_ids) in enumerate(val_dataloader):
            if batch_idx >= 1:  # Only use first batch
                break
                
            # Move to device
            device = next(pl_module.parameters()).device
            images = images.to(device)
            masks = masks.to(device)
            tooth_ids = tooth_ids.to(device)
            
            # Get predictions for multi-class segmentation
            with torch.no_grad():
                outputs = pl_module(images, tooth_ids)
                preds = torch.softmax(outputs, dim=1)
                preds_class = torch.argmax(preds, dim=1)
            
            # Log images
            self._log_images_to_comet(trainer, pl_module, images, masks, preds_class, 'val')
            break
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log images at the end of each training epoch"""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
            
        if trainer.logger is None:
            return
            
        # Get training dataloader
        train_dataloader = trainer.train_dataloader
        
        # Get first batch
        for batch_idx, (images, masks, tooth_ids) in enumerate(train_dataloader):
            if batch_idx >= 1:  # Only use first batch
                break
                
            # Move to device
            device = next(pl_module.parameters()).device
            images = images.to(device)
            masks = masks.to(device)
            tooth_ids = tooth_ids.to(device)
            
            # Get predictions for multi-class segmentation
            with torch.no_grad():
                outputs = pl_module(images, tooth_ids)
                preds = torch.softmax(outputs, dim=1)
                preds_class = torch.argmax(preds, dim=1)
            
            # Log images
            self._log_images_to_comet(trainer, pl_module, images, masks, preds_class, 'train')
            break
    
    def _log_images_to_comet(self, trainer, pl_module, images, masks, preds, split):
        """Log images to Comet ML for visualization"""
        batch_size = min(self.num_samples, images.shape[0])
        
        for i in range(batch_size):
            # Original image (denormalize)
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            # Ground truth mask (multi-class)
            gt_mask = masks[i].cpu().numpy()
            
            # Prediction (multi-class)
            pred_mask = preds[i].cpu().numpy().astype(np.uint8)
            
            # Create figure with subplots
            if split == 'val':
                # For validation: use 3 subplots (original, ground truth mask, prediction mask)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(img_np)
                axes[0].set_title(f'{split.capitalize()} Original')
                axes[0].axis('off')
                
                # Ground truth mask (从左往右第二张图)
                axes[1].imshow(gt_mask, cmap='tab10', vmin=0, vmax=4)
                axes[1].set_title(f'{split.capitalize()} Ground Truth Mask\n(0:bg, 1:upper, 2:lower, 3:specific, 4:other)')
                axes[1].axis('off')
                
                # Prediction mask
                axes[2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=4)
                axes[2].set_title(f'{split.capitalize()} Prediction Mask\n(0:bg, 1:upper, 2:lower, 3:specific, 4:other)')
                axes[2].axis('off')
                
            else:
                # For training: use 4 subplots (original, ground truth mask, prediction mask, comparison)
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original image
                axes[0].imshow(img_np)
                axes[0].set_title(f'{split.capitalize()} Original')
                axes[0].axis('off')
                
                # Ground truth mask
                axes[1].imshow(gt_mask, cmap='tab10', vmin=0, vmax=4)
                axes[1].set_title(f'{split.capitalize()} Ground Truth Mask\n(0:bg, 1:upper, 2:lower, 3:specific, 4:other)')
                axes[1].axis('off')
                
                # Prediction mask
                axes[2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=4)
                axes[2].set_title(f'{split.capitalize()} Prediction Mask\n(0:bg, 1:upper, 2:lower, 3:specific, 4:other)')
                axes[2].axis('off')
                
                # Create overlay comparison
                overlay_img = img_np.copy()
                # Highlight differences between prediction and ground truth
                diff_mask = (pred_mask != gt_mask)
                overlay_img[diff_mask] = [1.0, 0.0, 0.0]  # Red for differences
                
                axes[3].imshow(overlay_img)
                axes[3].set_title(f'{split.capitalize()} Differences\n(Red: prediction ≠ ground truth)')
                axes[3].axis('off')
            
            plt.tight_layout()
            
            # Log to Comet ML
            trainer.logger.experiment.log_figure(
                figure_name=f'{split}_epoch_{trainer.current_epoch}_sample_{i}',
                figure=fig,
                step=trainer.current_epoch
            )
            
            plt.close(fig)


def setup_callbacks(monitor_metric='val_loss', checkpoint_dir='./checkpoints', model_name=None, monitor_iou=True):
    """
    Setup training callbacks
    
    Args:
        monitor_metric: Metric to monitor for checkpointing
        checkpoint_dir: Directory to save checkpoints
        model_name: Model name for organizing checkpoints
    
    Returns:
        List of callbacks
    """
    # Create model-specific checkpoint directory
    if model_name:
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Model checkpoint callbacks
    filename_prefix = f'{model_name}-' if model_name else 'best-model-'
    
    # Checkpoint for best validation loss
    checkpoint_loss_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename=f'{filename_prefix}' + 'best-loss-epoch={epoch:02d}-val_loss={val_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    # Checkpoint for best validation IoU (if monitoring IoU)
    if monitor_iou:
        checkpoint_iou_callback = ModelCheckpoint(
            monitor='val_iou',
            dirpath=checkpoint_dir,
            filename=f'{filename_prefix}' + 'best-iou-epoch={epoch:02d}-val_iou={val_iou:.4f}',
            save_top_k=3,
            mode='max'
        )
    
    # Early stopping callback - optimized for 5 epochs
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,   # Slightly less sensitive for short training
        patience=3,        # Allow some fluctuation in 5 epochs
        verbose=True,
        mode='min'
    )
    
    # Image logging callback - log images every epoch
    image_logging_callback = ImageLoggingCallback(log_every_n_epochs=1, num_samples=4)
    
    # Return all callbacks
    callbacks = [checkpoint_loss_callback, early_stop_callback, image_logging_callback]
    if monitor_iou:
        callbacks.append(checkpoint_iou_callback)
    
    return callbacks


def setup_trainer(max_epochs, devices, strategy, callbacks, logger=None, experiment_name=None):
    """
    Setup PyTorch Lightning trainer
    
    Args:
        max_epochs: Maximum number of training epochs
        devices: Number of devices to use
        strategy: Training strategy
        callbacks: List of callbacks
        logger: Logger instance
        experiment_name: Experiment name for logging
    
    Returns:
        Trainer instance
    """
    print("🔄 Setting up trainer...")
    
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False  # Disable deterministic algorithms for multi-class segmentation
    )


def test_model_loading(model_path, data_module):
    """
    Test model loading and inference to verify device compatibility
    
    Args:
        model_path: Path to saved model checkpoint
        data_module: Data module for testing
    
    Returns:
        bool: True if test successful, False otherwise
    """
    print("🧪 Testing model loading and inference...")
    
    try:
        # Check if checkpoint file exists (only rank 0 saves the model in distributed training)
        if not os.path.exists(model_path):
            print(f"⚠️  Checkpoint not found: {model_path}")
            return False
            
        # Load model with proper device handling
        if torch.cuda.is_available():
            loaded_model = ToothSegmentationModel.load_from_checkpoint(
                model_path,
                map_location='cuda'
            )
            loaded_model = loaded_model.cuda()
        else:
            loaded_model = ToothSegmentationModel.load_from_checkpoint(
                model_path,
                map_location='cpu'
            )
        
        loaded_model.eval()
        
        # Test with a single batch
        test_loader = data_module.test_dataloader()
        for batch_idx, (images, masks, tooth_ids) in enumerate(test_loader):
            if batch_idx >= 1:  # Only test one batch
                break
                
            # Move data to same device as model
            device = next(loaded_model.parameters()).device
            images = images.to(device)
            masks = masks.to(device)
            tooth_ids = tooth_ids.to(device)
            
            with torch.no_grad():
                outputs = loaded_model(images, tooth_ids)
                # For multi-class segmentation, masks should be long type
                masks_long = masks.long()
                loss = loaded_model.criterion(outputs, masks_long)
                
                # Calculate predictions for multi-class segmentation
                preds = torch.softmax(outputs, dim=1)
                preds_class = torch.argmax(preds, dim=1)
                
                # Calculate IoU using torchmetrics
                iou = loaded_model.val_iou(preds_class, masks_long)
                
            print(f"✅ Loaded model test - Loss: {loss.item():.4f}, IoU: {iou.item():.4f}")
            return True
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_results(model, data_module, result_dir='./results', num_samples=5, model_name=None):
    """
    Visualize training, validation, and test results
    
    Args:
        model: Trained model
        data_module: Data module
        result_dir: Directory to save results
        num_samples: Number of samples to visualize
        model_name: Model name for organizing results
    """
    # Create model-specific result directory with train/val/test subdirectories
    if model_name:
        result_dir = os.path.join(result_dir, model_name)
    
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(result_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    
    model.eval()
    
    # Get device
    device = next(model.parameters()).device
    
    # Visualize for each split
    for split in splits:
        print(f"📊 Visualizing {split} results...")
        
        # Get dataloader
        if split == 'train':
            dataloader = data_module.train_dataloader()
        elif split == 'val':
            dataloader = data_module.val_dataloader()
        else:
            dataloader = data_module.test_dataloader()
        
        # Process samples
        for batch_idx, (images, masks, tooth_ids) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            tooth_ids = tooth_ids.to(device)
            
            with torch.no_grad():
                outputs = model(images, tooth_ids)
                preds = torch.softmax(outputs, dim=1)
                preds_class = torch.argmax(preds, dim=1)
            
            # Convert tensors to numpy for visualization
            for i in range(images.shape[0]):
                # Original image (denormalize)
                img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                
                # Ground truth mask (multi-class)
                gt_mask = masks[i].cpu().numpy()
                
                # Prediction (multi-class)
                pred_mask = preds_class[i].cpu().numpy().astype(np.uint8)
                
                # Create overlay images with different colors for each class
                # Ground truth overlay (multi-color mask on original image)
                gt_overlay = img_np.copy()
                # Class 1: Upper teeth - Red
                gt_overlay[gt_mask == 1] = [1.0, 0.0, 0.0]
                # Class 2: Lower teeth - Green
                gt_overlay[gt_mask == 2] = [0.0, 1.0, 0.0]
                # Class 3: Specific tooth - Blue
                gt_overlay[gt_mask == 3] = [0.0, 0.0, 1.0]
                # Class 4: Other - Yellow
                gt_overlay[gt_mask == 4] = [1.0, 1.0, 0.0]
                
                # Prediction overlay (multi-color mask on original image)
                pred_overlay = img_np.copy()
                # Class 1: Upper teeth - Red
                pred_overlay[pred_mask == 1] = [1.0, 0.0, 0.0]
                # Class 2: Lower teeth - Green
                pred_overlay[pred_mask == 2] = [0.0, 1.0, 0.0]
                # Class 3: Specific tooth - Blue
                pred_overlay[pred_mask == 3] = [0.0, 0.0, 1.0]
                # Class 4: Other - Yellow
                pred_overlay[pred_mask == 4] = [1.0, 1.0, 0.0]
                
                split_dir = os.path.join(result_dir, split)
                
                # For train and val, save ground truth visualization
                if split in ['train', 'val']:
                    # Ground truth visualization
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Original image with ground truth overlay
                    axes[0].imshow(gt_overlay)
                    axes[0].set_title(f'{split.capitalize()} Ground Truth Overlay\n(Red:Upper, Green:Lower, Blue:Specific, Yellow:Other)')
                    axes[0].axis('off')
                    
                    # Ground truth mask
                    axes[1].imshow(gt_mask, cmap='tab10', vmin=0, vmax=4)
                    axes[1].set_title('Ground Truth Mask (Multi-class)')
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'{split_dir}/gt_sample_{batch_idx}_{i}.png', dpi=150, bbox_inches='tight')
                    plt.close()
                
                # Save prediction visualization for all splits
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # Original image with prediction overlay
                axes[0].imshow(pred_overlay)
                axes[0].set_title(f'{split.capitalize()} Prediction Overlay\n(Red:Upper, Green:Lower, Blue:Specific, Yellow:Other)')
                axes[0].axis('off')
                
                # Prediction mask
                axes[1].imshow(pred_mask, cmap='tab10', vmin=0, vmax=4)
                axes[1].set_title('Prediction Mask (Multi-class)')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{split_dir}/pred_sample_{batch_idx}_{i}.png', dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"✅ Results saved to {result_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train tooth segmentation model')
    
    parser.add_argument('--model', type=str, default='deeplabv3_resnet50',
                       choices=['deeplabv3_resnet50', 'deeplabv3_resnet101', 
                               'fcn_resnet50', 'fcn_resnet101', 
                               'lraspp_mobilenet_v3_large'],
                       help='Model architecture to use')
    
    parser.add_argument('--data_dir', type=str, default='../data/ToothSegmDataset',
                       help='Path to dataset directory')
    
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    
    parser.add_argument('--comet_api_key', type=str, default='',
                       help='Comet ML API key (optional)')
    
    parser.add_argument('--comet_project', type=str, default='tooth-segmentation',
                       help='Comet ML project name')
    
    parser.add_argument('--comet_experiment', type=str, default=None,
                       help='Comet ML experiment name')
    
    parser.add_argument('--result_dir', type=str, default='./results',
                       help='Directory to save visualization results')
    
    parser.add_argument('--gpus', type=str, default='0,2',
                       help='GPU devices to use (comma-separated)')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("🚀 Starting tooth segmentation training...")
    print(f"🎯 Using model: {args.model}")
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': 36,
        'max_epochs': args.epochs,
        'learning_rate': args.lr,
        'model_name': args.model,
        'devices': [int(gpu) for gpu in args.gpus.split(',')],
        'strategy': 'ddp_find_unused_parameters_true',
        'tooth_ids': ['11', '12', '13', '14', '15', '16', '17', 
                     '21', '22', '23', '24', '25', '26', '27',
                     '34', '35', '36', '37', '45', '46', '47'],  # All teeth
        'comet_api_key': args.comet_api_key,
        'comet_project': args.comet_project,
        'comet_experiment': args.comet_experiment or f'{args.model}_experiment',
        'result_dir': args.result_dir
    }
    
    # Set random seed for reproducibility
    L.seed_everything(42)
    
    # Display available models
    print("📋 Available models:")
    for model_name in get_available_models().keys():
        print(f"   - {model_name}")
    
    print(f"🎯 Using model: {config['model_name']}")
    
    # Setup Comet ML logger
    comet_logger = setup_comet_logger(
        api_key=config['comet_api_key'],
        project_name=config['comet_project'],
        experiment_name=config['comet_experiment']
    )
    
    # Setup data module
    data_module = setup_data_module(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        tooth_ids=config['tooth_ids']
    )
    
    # Create model
    model = create_model(
        model_name=config['model_name'],
        num_classes=5,  # Multi-class segmentation: 5 classes (0-4)
        learning_rate=config['learning_rate'],
        tooth_ids=config['tooth_ids']
    )
    
    # Setup callbacks with model name and IoU monitoring
    callbacks = setup_callbacks(
        monitor_metric='val_loss',
        checkpoint_dir='./checkpoints',
        model_name=config['model_name'],
        monitor_iou=True
    )
    
    # Setup trainer
    trainer = setup_trainer(
        max_epochs=config['max_epochs'],
        devices=config['devices'],
        strategy=config['strategy'],
        callbacks=callbacks,
        logger=comet_logger
    )
    
    # Start training
    print("🎬 Starting training...")
    try:
        trainer.fit(model, data_module)
        
        # Test the model
        print("🧪 Running model testing...")
        trainer.test(model, data_module)
        
        # Test model loading and inference (only on rank 0 to avoid file not found errors)
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback.best_model_path:
            best_model_path = trainer.checkpoint_callback.best_model_path
            print(f"🏆 Best model saved at: {best_model_path}")
            
            # Only test loading and visualization if the checkpoint file exists
            if os.path.exists(best_model_path):
                # Test model loading
                test_model_loading(best_model_path, data_module)
                
                # Load best model for visualization
                if torch.cuda.is_available():
                    best_model = ToothSegmentationModel.load_from_checkpoint(
                        best_model_path,
                        map_location='cuda'
                    )
                    best_model = best_model.cuda()
                else:
                    best_model = ToothSegmentationModel.load_from_checkpoint(
                        best_model_path,
                        map_location='cpu'
                    )
                
                # Generate visualization results with model name
                print("🎨 Generating visualization results...")
                visualize_results(best_model, data_module, result_dir=config['result_dir'], model_name=config['model_name'])
            else:
                print("⚠️  Best model checkpoint not found, skipping visualization")
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()