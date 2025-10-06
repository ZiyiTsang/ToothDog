#!/usr/bin/env python3
"""
Multi-task training script for simultaneous tooth segmentation and tooth ID classification
Supports multiple pretrained models and handles device compatibility
"""

import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateMonitor, GradientAccumulationScheduler
from lightning.pytorch.loggers import CometLogger
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model import ToothModel
from dataModule import ToothDataModule


def setup_comet_logger(api_key=None, project_name="tooth-multi-task", experiment_name=None):
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


def create_multi_task_model(model_name, num_seg_classes=5, num_tooth_classes=31, learning_rate=1e-4,
                           seg_loss_weight=0.7, cls_loss_weight=0.3, tooth_ids=None, mode='multi-task'):
    """
    Create unified tooth model supporting three training modes
    
    Args:
        model_name: Name of the model architecture
        num_seg_classes: Number of segmentation classes (5 for multi-class segmentation)
        num_cls_classes: Number of classification classes (31 for tooth IDs)
        learning_rate: Learning rate for training
        seg_loss_weight: Weight for segmentation loss
        cls_loss_weight: Weight for classification loss
        tooth_ids: List of tooth IDs to segment
        mode: Training mode - 'classification', 'segmentation', or 'multi-task'
    
    Returns:
        ToothModel instance
    """
    available_models = get_available_models()
    
    if model_name not in available_models:
        raise ValueError(f"Model {model_name} not available. Available models: {list(available_models.keys())}")
    
    print(f"🔄 Creating {mode} {model_name} model...")
    return ToothModel(
        model_name=model_name,
        num_seg_classes=num_seg_classes,
        num_tooth_classes=num_tooth_classes,
        learning_rate=learning_rate,
        seg_loss_weight=seg_loss_weight,
        cls_loss_weight=cls_loss_weight,
        tooth_ids=tooth_ids,
        mode=mode
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


class MultiTaskImageLoggingCallback(Callback):
    """Callback to log images and classification results to Comet ML every epoch"""
    
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
            
            # Get predictions for multi-task learning
            with torch.no_grad():
                seg_outputs, cls_outputs = pl_module(images, tooth_ids)
                seg_preds = torch.softmax(seg_outputs, dim=1)
                seg_preds_class = torch.argmax(seg_preds, dim=1)
                
                cls_preds = torch.softmax(cls_outputs, dim=1)
                cls_preds_class = torch.argmax(cls_preds, dim=1)
            
            # Log images and classification results
            self._log_multi_task_results(trainer, pl_module, images, masks, tooth_ids, 
                                       seg_preds_class, cls_preds_class, 'val')
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
            
            # Get predictions for multi-task learning
            with torch.no_grad():
                seg_outputs, cls_outputs = pl_module(images, tooth_ids)
                seg_preds = torch.softmax(seg_outputs, dim=1)
                seg_preds_class = torch.argmax(seg_preds, dim=1)
                
                cls_preds = torch.softmax(cls_outputs, dim=1)
                cls_preds_class = torch.argmax(cls_preds, dim=1)
            
            # Log images and classification results
            self._log_multi_task_results(trainer, pl_module, images, masks, tooth_ids,
                                       seg_preds_class, cls_preds_class, 'train')
            break
    
    def _log_multi_task_results(self, trainer, pl_module, images, masks, tooth_ids,
                              seg_preds, cls_preds, split):
        """Log multi-task results to Comet ML for visualization with enhanced classification and segmentation display"""
        batch_size = min(self.num_samples, images.shape[0])
        
        for i in range(batch_size):
            # Original image (denormalize)
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            # Ground truth mask (multi-class)
            gt_mask = masks[i].cpu().numpy()
            
            # Prediction (multi-class)
            pred_mask = seg_preds[i].cpu().numpy().astype(np.uint8)
            
            # Ground truth tooth ID (convert one-hot to class index)
            gt_tooth_id = torch.argmax(tooth_ids[i]).item()
            pred_tooth_id = cls_preds[i].item()
            
            # Get classification confidence scores
            with torch.no_grad():
                # Forward pass to get confidence scores - use eval mode to avoid BatchNorm issues
                pl_module.eval()
                seg_outputs, cls_outputs = pl_module(images[i:i+1])
                cls_probs = torch.softmax(cls_outputs, dim=1)[0]
                seg_probs = torch.softmax(seg_outputs, dim=1)[0]
                pl_module.train()  # Restore training mode
            
            cls_confidence = cls_probs.max().item()
            seg_confidence = seg_probs.max(dim=0)[0].mean().item()  # Average confidence across pixels
            
            # Create figure with subplots - larger figure for more detailed visualization
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            
            # Row 1: Original image and masks
            # Original image
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title(f'{split.capitalize()} Original Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Ground truth mask
            axes[0, 1].imshow(gt_mask, cmap='tab10', vmin=0, vmax=4)
            axes[0, 1].set_title(f'{split.capitalize()} Ground Truth Mask\n(0:bg, 1:upper, 2:lower, 3:specific, 4:other)',
                               fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Prediction mask
            axes[0, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=4)
            axes[0, 2].set_title(f'{split.capitalize()} Prediction Mask\n(0:bg, 1:upper, 2:lower, 3:specific, 4:other)',
                               fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            
            # Row 2: Comparison and confidence visualization
            # Create overlay comparison
            overlay_img = img_np.copy()
            diff_mask = (pred_mask != gt_mask)
            overlay_img[diff_mask] = [1.0, 0.0, 0.0]  # Red for differences
            
            axes[1, 0].imshow(overlay_img)
            axes[1, 0].set_title(f'{split.capitalize()} Differences\n(Red: prediction ≠ ground truth)',
                               fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            # Segmentation confidence heatmap
            seg_confidence_map = seg_probs.max(dim=0)[0].cpu().numpy()
            im = axes[1, 1].imshow(seg_confidence_map, cmap='viridis', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Segmentation Confidence Heatmap\n(Mean: {seg_confidence:.3f})',
                               fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # Classification confidence bar chart
            top_k = 5
            top_probs, top_indices = torch.topk(cls_probs, top_k)
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            bars = axes[1, 2].bar(range(top_k), top_probs, color=['red' if idx == pred_tooth_id else 'blue' for idx in top_indices])
            axes[1, 2].set_title(f'Top {top_k} Classification Probabilities', fontsize=12, fontweight='bold')
            axes[1, 2].set_ylabel('Probability')
            axes[1, 2].set_xticks(range(top_k))
            axes[1, 2].set_xticklabels([f'ID:{idx}' for idx in top_indices], rotation=45)
            axes[1, 2].set_ylim(0, 1)
            
            # Add probability values on bars
            for bar, prob in zip(bars, top_probs):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Row 3: Detailed classification results and tooth highlighting
            # Tooth ID classification results
            classification_correct = (pred_tooth_id == gt_tooth_id)
            result_color = 'green' if classification_correct else 'red'
            
            axes[2, 0].text(0.1, 0.7,
                           f'Ground Truth Tooth ID: {gt_tooth_id}\n'
                           f'Predicted Tooth ID: {pred_tooth_id}\n'
                           f'Classification: {"✓ CORRECT" if classification_correct else "✗ WRONG"}',
                           fontsize=14, transform=axes[2, 0].transAxes,
                           color=result_color, fontweight='bold')
            axes[2, 0].set_title('Tooth ID Classification Results', fontsize=12, fontweight='bold')
            axes[2, 0].axis('off')
            
            # Highlight specific tooth in segmentation mask
            # Create highlighted mask where only the predicted tooth is shown
            highlighted_mask = np.zeros_like(pred_mask)
            
            # Find the specific tooth region (class 3 in segmentation mask)
            specific_tooth_mask = (pred_mask == 3)
            
            if np.any(specific_tooth_mask):
                # Create overlay with highlighted specific tooth
                highlight_img = img_np.copy()
                # Color the specific tooth region in yellow
                highlight_img[specific_tooth_mask] = [1.0, 1.0, 0.0]  # Yellow highlight
                
                axes[2, 1].imshow(highlight_img)
                axes[2, 1].set_title(f'Highlighted Tooth ID: {pred_tooth_id}\n(Yellow: specific tooth region)',
                                   fontsize=12, fontweight='bold')
                axes[2, 1].axis('off')
                
                # Log highlighted tooth image separately to Comet ML for better visualization
                fig_highlight, ax_highlight = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image with highlighted tooth
                ax_highlight[0].imshow(highlight_img)
                ax_highlight[0].set_title(f'Highlighted Tooth ID: {pred_tooth_id}', fontsize=14, fontweight='bold')
                ax_highlight[0].axis('off')
                
                # Segmentation mask comparison
                ax_highlight[1].imshow(pred_mask, cmap='tab10', vmin=0, vmax=4)
                ax_highlight[1].set_title(f'Segmentation Mask\n(Class 3: Specific Tooth)', fontsize=14, fontweight='bold')
                ax_highlight[1].axis('off')
                
                plt.tight_layout()
                
                # Log highlighted tooth image to Comet ML
                trainer.logger.experiment.log_figure(
                    figure_name=f'{split}_highlighted_tooth_{pred_tooth_id}_epoch_{trainer.current_epoch}_sample_{i}',
                    figure=fig_highlight,
                    step=trainer.current_epoch
                )
                
                plt.close(fig_highlight)
                
            else:
                # If no specific tooth found, show confidence information
                axes[2, 1].text(0.1, 0.7,
                               f'Classification Confidence: {cls_confidence:.3f}\n'
                               f'Segmentation Confidence: {seg_confidence:.3f}\n'
                               f'Predicted Probability: {cls_probs[pred_tooth_id].item():.3f}',
                               fontsize=12, transform=axes[2, 1].transAxes)
                axes[2, 1].set_title('Confidence Scores', fontsize=12, fontweight='bold')
                axes[2, 1].axis('off')
            
            # Performance metrics summary
            current_metrics = {k: v for k, v in trainer.callback_metrics.items() if 'val' in k or 'train' in k}
            metrics_text = f'Epoch: {trainer.current_epoch}\n'
            if 'val_total_loss' in current_metrics:
                metrics_text += f'Total Loss: {current_metrics["val_total_loss"]:.4f}\n'
            if 'val_seg_iou' in current_metrics:
                metrics_text += f'Seg IoU: {current_metrics["val_seg_iou"]:.4f}\n'
            if 'val_cls_acc' in current_metrics:
                metrics_text += f'Cls Acc: {current_metrics["val_cls_acc"]:.4f}'
            
            axes[2, 2].text(0.1, 0.7, metrics_text, fontsize=12, transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Current Metrics', fontsize=12, fontweight='bold')
            axes[2, 2].axis('off')
            
            plt.tight_layout()
            
            # Log to Comet ML
            trainer.logger.experiment.log_figure(
                figure_name=f'{split}_epoch_{trainer.current_epoch}_sample_{i}',
                figure=fig,
                step=trainer.current_epoch
            )
            
            plt.close(fig)


def setup_callbacks(monitor_metric='val_total_loss', checkpoint_dir='./multi_task_checkpoints',
                   model_name=None, monitor_iou=True, mode='multi-task'):
    """
    Setup training callbacks for multi-task learning
    
    Args:
        monitor_metric: Metric to monitor for checkpointing
        checkpoint_dir: Directory to save checkpoints
        model_name: Model name for organizing checkpoints
        monitor_iou: Whether to monitor IoU metrics
        mode: Training mode - determines which metrics to monitor
    
    Returns:
        List of callbacks
    """
    # Create model-specific checkpoint directory
    if model_name:
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Model checkpoint callbacks
    filename_prefix = f'{model_name}-' if model_name else f'{mode}-'
    
    callbacks = []
    
    # Always monitor total loss
    checkpoint_loss_callback = ModelCheckpoint(
        monitor='val_total_loss',
        dirpath=checkpoint_dir,
        filename=f'{filename_prefix}' + 'best-total-loss-epoch={epoch:02d}-val_total_loss={val_total_loss:.4f}',
        save_top_k=1,  # Save only the best model
        mode='min',
        save_last=False  # Don't save last checkpoint to save storage
    )
    callbacks.append(checkpoint_loss_callback)
    
    # Monitor segmentation metrics only for segmentation and multi-task modes
    if mode in ['segmentation', 'multi-task']:
        if monitor_iou:
            checkpoint_iou_callback = ModelCheckpoint(
                monitor='val_seg_iou',
                dirpath=checkpoint_dir,
                filename=f'{filename_prefix}' + 'best-iou-epoch={epoch:02d}-val_seg_iou={val_seg_iou:.4f}',
                save_top_k=1,  # Save only the best model
                mode='max'
            )
            callbacks.append(checkpoint_iou_callback)
    
    # Monitor classification metrics only for classification and multi-task modes
    if mode in ['classification', 'multi-task']:
        checkpoint_cls_callback = ModelCheckpoint(
            monitor='val_cls_acc',
            dirpath=checkpoint_dir,
            filename=f'{filename_prefix}' + 'best-cls-acc-epoch={epoch:02d}-val_cls_acc={val_cls_acc:.4f}',
            save_top_k=1,  # Save only the best model
            mode='max'
        )
        callbacks.append(checkpoint_cls_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_total_loss',
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)
    
    # Multi-task image logging callback
    image_logging_callback = MultiTaskImageLoggingCallback(log_every_n_epochs=1, num_samples=4)
    callbacks.append(image_logging_callback)
    
    # Learning rate monitor callback
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor_callback)
    
    # Gradient accumulation scheduler (optional)
    gradient_accumulation_callback = GradientAccumulationScheduler(scheduling={0: 1})
    callbacks.append(gradient_accumulation_callback)
    
    print(f"✅ Setup {len(callbacks)} callbacks for {mode} training:")
    for i, callback in enumerate(callbacks):
        print(f"   {i+1}. {callback.__class__.__name__}")
    
    return callbacks


def get_advanced_training_config():
    """
    Get advanced training configuration with comprehensive monitoring options
    
    Returns:
        dict: Advanced training configuration
    """
    return {
        'gradient_clip_val': 1.0,
        'gradient_clip_algorithm': "norm",
        'accumulate_grad_batches': 1,
        'num_sanity_val_steps': 2,
        'overfit_batches': 0,
        'val_check_interval': None,
        'precision': '32-true',
        'detect_anomaly': False,
        'profiler': None,
        'benchmark': True,  # Enable cudnn benchmark for faster training
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': False,
        'log_every_n_steps': 10,
        'check_val_every_n_epoch': 1,
    }


def setup_trainer(max_epochs, devices, strategy, callbacks, logger=None, experiment_name=None):
    """
    Setup PyTorch Lightning trainer with enhanced monitoring and debugging
    
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
    print("🔄 Setting up trainer with enhanced monitoring...")
    
    # Get advanced training configuration
    advanced_config = get_advanced_training_config()
    
    print("📊 Advanced training configuration:")
    for key, value in advanced_config.items():
        print(f"   - {key}: {value}")
    
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        **advanced_config
    )


def test_multi_task_model_loading(model_path, data_module):
    """
    Test multi-task model loading and inference to verify device compatibility
    
    Args:
        model_path: Path to saved model checkpoint
        data_module: Data module for testing
    
    Returns:
        bool: True if test successful, False otherwise
    """
    print("🧪 Testing multi-task model loading and inference...")
    
    try:
        # Check if checkpoint file exists
        if not os.path.exists(model_path):
            print(f"⚠️  Checkpoint not found: {model_path}")
            return False
            
        # Load model with proper device handling
        if torch.cuda.is_available():
            loaded_model = ToothModel.load_from_checkpoint(
                model_path,
                map_location='cuda'
            )
            loaded_model = loaded_model.cuda()
        else:
            loaded_model = ToothModel.load_from_checkpoint(
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
                seg_outputs, cls_outputs = loaded_model(images, tooth_ids)
                
                # Calculate segmentation metrics
                masks_long = masks.long()
                seg_loss = loaded_model.seg_criterion(seg_outputs, masks_long)
                seg_preds = torch.softmax(seg_outputs, dim=1)
                seg_preds_class = torch.argmax(seg_preds, dim=1)
                seg_iou = loaded_model.val_seg_iou(seg_preds_class, masks_long)
                
                # Calculate classification metrics
                cls_targets = torch.argmax(tooth_ids, dim=1)
                cls_loss = loaded_model.cls_criterion(cls_outputs, cls_targets)
                cls_preds = torch.softmax(cls_outputs, dim=1)
                cls_preds_class = torch.argmax(cls_preds, dim=1)
                cls_accuracy = loaded_model.val_cls_acc(cls_preds_class, cls_targets)
                
                total_loss = loaded_model.seg_loss_weight * seg_loss + loaded_model.cls_loss_weight * cls_loss
            
            print(f"✅ Loaded multi-task model test - Total Loss: {total_loss.item():.4f}")
            print(f"   Segmentation - Loss: {seg_loss.item():.4f}, IoU: {seg_iou.item():.4f}")
            print(f"   Classification - Loss: {cls_loss.item():.4f}, Accuracy: {cls_accuracy.item():.4f}")
            return True
            
    except Exception as e:
        print(f"❌ Multi-task model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """Parse command line arguments for multi-task training"""
    parser = argparse.ArgumentParser(description='Train multi-task tooth segmentation and classification model')
    
    parser.add_argument('--model', type=str, default='deeplabv3_resnet50',
                       choices=['deeplabv3_resnet50', 'deeplabv3_resnet101',
                               'fcn_resnet50', 'fcn_resnet101',
                               'lraspp_mobilenet_v3_large'],
                       help='Model architecture to use')
    
    parser.add_argument('--data_dir', type=str, default='../data/ToothSegmDataset',
                       help='Path to dataset directory')
    
    parser.add_argument('--batch_size', type=int, default=48,
                       help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    
    parser.add_argument('--seg_loss_weight', type=float, default=0.7,
                       help='Weight for segmentation loss')
    
    parser.add_argument('--cls_loss_weight', type=float, default=0.3,
                       help='Weight for classification loss')
    
    parser.add_argument('--mode', type=str, default='multi-task',
                       choices=['classification', 'segmentation', 'multi-task'],
                       help='Training mode: classification only, segmentation only, or multi-task')
    
    parser.add_argument('--comet_api_key', type=str, default='LXYMHm1xV9Y09OfGVdmB0YQUy',
                       help='Comet ML API key (optional)')
    
    parser.add_argument('--comet_project', type=str, default='tooth-segmentation',
                       help='Comet ML project name')
    
    parser.add_argument('--comet_experiment', type=str, default=None,
                       help='Comet ML experiment name')
    
    parser.add_argument('--result_dir', type=str, default='./multi_task_results',
                       help='Directory to save visualization results')
    
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                       help='GPU devices to use (comma-separated)')
    
    return parser.parse_args()


def main():
    """Main multi-task training function"""
    args = parse_args()
    
    print("🚀 Starting multi-task tooth segmentation and classification training...")
    print(f"🎯 Using model: {args.model}")
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': 36,
        'max_epochs': args.epochs,
        'learning_rate': args.lr,
        'seg_loss_weight': args.seg_loss_weight,
        'cls_loss_weight': args.cls_loss_weight,
        'model_name': args.model,
        'mode': args.mode,
        'devices': [int(gpu) for gpu in args.gpus.split(',')],
        'strategy': 'ddp_find_unused_parameters_true',
        'tooth_ids': ['11', '12', '13', '14', '15', '16', '17',
                     '21', '22', '23', '24', '25', '26', '27',
                     '34', '35', '36', '37', '45', '46', '47'],  # All teeth
        'comet_api_key': args.comet_api_key,
        'comet_project': args.comet_project,
        'comet_experiment': args.comet_experiment or f'{args.model}_multi_task_experiment',
        'result_dir': args.result_dir
    }
    
    # Set random seed for reproducibility
    L.seed_everything(42)
    
    # Display available models
    print("📋 Available models:")
    for model_name in get_available_models().keys():
        print(f"   - {model_name}")
    
    print(f"🎯 Using model: {config['model_name']}")
    print(f"⚖️  Loss weights - Segmentation: {config['seg_loss_weight']}, Classification: {config['cls_loss_weight']}")
    
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
    
    # Create unified tooth model
    model = create_multi_task_model(
        model_name=config['model_name'],
        num_seg_classes=5,  # Multi-class segmentation: 5 classes (0-4)
        num_tooth_classes=31,  # Classification: 31 tooth ID classes
        learning_rate=config['learning_rate'],
        seg_loss_weight=config['seg_loss_weight'],
        cls_loss_weight=config['cls_loss_weight'],
        tooth_ids=config['tooth_ids'],
        mode=config['mode']
    )
    
    # Setup callbacks with model name and IoU monitoring
    callbacks = setup_callbacks(
        monitor_metric='val_total_loss',
        checkpoint_dir='./multi_task_checkpoints',
        model_name=config['model_name'],
        monitor_iou=True,
        mode=config['mode']
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
    print("🎬 Starting multi-task training...")
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
                test_multi_task_model_loading(best_model_path, data_module)
            else:
                print("⚠️  Best model checkpoint not found, skipping loading test")
        
        print("✅ Multi-task training completed successfully!")
        
    except Exception as e:
        print(f"❌ Multi-task training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()