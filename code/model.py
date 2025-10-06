
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import (
    deeplabv3_resnet50, deeplabv3_resnet101,
    fcn_resnet50, fcn_resnet101,
    lraspp_mobilenet_v3_large
)
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import cm
import torchmetrics
# Define mapping from tooth ID to class index (copied from dataModule.py)
TOOTH_ID_TO_LABEL = {
    "11": 0, "12": 1, "13": 2, "14": 3, "15": 4,
    "16": 5, "17": 6, "21": 8, "22": 9, "23": 10,
    "24": 11, "25": 12, "26": 13, "27": 14,
    "34": 19, "35": 20, "36": 21, "37": 22,
    "45": 28, "46": 29, "47": 30
}

# Model creation functions
MODEL_CREATORS = {
    'deeplabv3_resnet50': lambda: deeplabv3_resnet50(weights='DEFAULT'),
    'deeplabv3_resnet101': lambda: deeplabv3_resnet101(weights='DEFAULT'),
    'fcn_resnet50': lambda: fcn_resnet50(weights='DEFAULT'),
    'fcn_resnet101': lambda: fcn_resnet101(weights='DEFAULT'),
    'lraspp_mobilenet_v3_large': lambda: lraspp_mobilenet_v3_large(weights='DEFAULT')
}




class ToothModel(L.LightningModule):
    """Unified tooth model supporting three training modes: classification only, segmentation only, and multi-task"""
    
    def __init__(self, model_name='deeplabv3_resnet50', num_seg_classes=5, num_tooth_classes=31,
                 learning_rate=1e-4, tooth_ids=None, seg_loss_weight=0.7, cls_loss_weight=0.3, mode='multi-task'):
        super().__init__()
        self.save_hyperparameters()
        
        # Validate model name
        if model_name not in MODEL_CREATORS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(MODEL_CREATORS.keys())}")
        
        # Use specified pretrained model
        self.backbone = MODEL_CREATORS[model_name]()
        
        # Unified segmentation head modification for all backbone architectures
        # All models will use consistent segmentation head structure
        if 'deeplabv3' in model_name:
            # Replace the final classifier layer with unified segmentation head
            self.backbone.classifier[4] = nn.Conv2d(256, num_seg_classes, kernel_size=(1, 1))
        elif 'fcn' in model_name:
            # Replace the final classifier layer with unified segmentation head
            self.backbone.classifier[4] = nn.Conv2d(512, num_seg_classes, kernel_size=(1, 1))
        elif 'lraspp' in model_name:
            # Replace both low and high classifier layers with unified segmentation head
            self.backbone.classifier.low_classifier = nn.Conv2d(40, num_seg_classes, kernel_size=(1, 1))
            self.backbone.classifier.high_classifier = nn.Conv2d(128, num_seg_classes, kernel_size=(1, 1))
        
        # Add tooth ID classification head
        self._add_classification_head(model_name, num_tooth_classes)
        
        # Loss functions
        self.seg_criterion = nn.CrossEntropyLoss()  # For segmentation
        self.cls_criterion = nn.CrossEntropyLoss()  # For classification
        
        # Loss weights for multi-task learning
        self.seg_loss_weight = seg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        
        # Segmentation metrics
        self.train_seg_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_seg_classes)
        self.val_seg_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_seg_classes)
        self.test_seg_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_seg_classes)
        
        self.train_seg_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_seg_classes, average='macro')
        self.val_seg_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_seg_classes, average='macro')
        self.test_seg_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_seg_classes, average='macro')
        
        # Classification metrics
        self.train_cls_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_tooth_classes)
        self.val_cls_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_tooth_classes)
        self.test_cls_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_tooth_classes)
        
        self.train_cls_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_tooth_classes, average='macro')
        self.val_cls_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_tooth_classes, average='macro')
        self.test_cls_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_tooth_classes, average='macro')
        
        # Additional metrics for comprehensive monitoring
        self.train_cls_precision = torchmetrics.Precision(task='multiclass', num_classes=num_tooth_classes, average='macro')
        self.val_cls_precision = torchmetrics.Precision(task='multiclass', num_classes=num_tooth_classes, average='macro')
        self.test_cls_precision = torchmetrics.Precision(task='multiclass', num_classes=num_tooth_classes, average='macro')
        
        self.train_cls_recall = torchmetrics.Recall(task='multiclass', num_classes=num_tooth_classes, average='macro')
        self.val_cls_recall = torchmetrics.Recall(task='multiclass', num_classes=num_tooth_classes, average='macro')
        self.test_cls_recall = torchmetrics.Recall(task='multiclass', num_classes=num_tooth_classes, average='macro')
        
        # Per-class metrics for detailed analysis
        self.train_cls_acc_per_class = torchmetrics.Accuracy(task='multiclass', num_classes=num_tooth_classes, average='none')
        self.val_cls_acc_per_class = torchmetrics.Accuracy(task='multiclass', num_classes=num_tooth_classes, average='none')
        
        # Segmentation per-class metrics
        self.train_seg_acc_per_class = torchmetrics.Accuracy(task='multiclass', num_classes=num_seg_classes, average='none')
        self.val_seg_acc_per_class = torchmetrics.Accuracy(task='multiclass', num_classes=num_seg_classes, average='none')
        
        # Learning rate
        self.learning_rate = learning_rate
        
        # Tooth IDs to segment
        self.tooth_ids = tooth_ids or list(TOOTH_ID_TO_LABEL.keys())
        
        # Store model name and class counts for reference
        self.model_name = model_name
        self.num_seg_classes = num_seg_classes
        self.num_tooth_classes = num_tooth_classes
        
        # Training mode: 'classification', 'segmentation', or 'multi-task'
        self.mode = mode
        self._validate_mode()
        
        # Add tooth ID conditioning for segmentation (only for segmentation and multi-task modes)
        if self.mode in ['segmentation', 'multi-task']:
            self.tooth_id_embedding = nn.Embedding(num_tooth_classes, 64)
            self.tooth_id_projection = nn.Linear(64, num_seg_classes)
    
    def _validate_mode(self):
        """Validate the training mode parameter"""
        valid_modes = ['classification', 'segmentation', 'multi-task']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Valid modes are: {valid_modes}")
    
    def _add_classification_head(self, model_name, num_tooth_classes):
        """Add unified classification head for all backbone architectures"""
        # Unified classification head with consistent structure and parameters
        # All models will use the same hidden dimensions, dropout rate, and layer structure
        hidden_dim = 512
        dropout_rate = 0.5
        
        # Determine input dimension based on backbone architecture
        if 'deeplabv3' in model_name or 'fcn' in model_name:
            input_dim = 2048  # ResNet-based backbones
        elif 'lraspp' in model_name:
            input_dim = 960   # MobileNet-based backbone (corrected from 128 to 960)
        else:
            input_dim = 512   # Default fallback
        
        # Unified classification head structure - identical for all models
        # The only difference is the input dimension due to backbone architecture
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_tooth_classes)
        )
    
    def forward(self, x, tooth_ids=None):
        """Forward pass through the model
        
        Args:
            x: Input images [B, 3, H, W]
            tooth_ids: Optional tooth IDs [B, num_classes] (one-hot encoded)
        
        Returns:
            seg_output: Segmentation outputs [B, num_seg_classes, H, W]
            cls_output: Classification outputs [B, num_tooth_classes]
        """
        # IMPORTANT: Extract classification features BEFORE applying tooth ID conditioning
        # This ensures classification task doesn't have access to tooth ID information
        
        # Get features for classification from the backbone (before any conditioning)
        if 'deeplabv3' in self.model_name:
            # Use features from the backbone before ASPP
            features = self.backbone.backbone(x)['out']
        elif 'fcn' in self.model_name:
            # Use features from the backbone
            features = self.backbone.backbone(x)['out']
        elif 'lraspp' in self.model_name:
            # For lraspp, we need to extract the correct high-level features
            # The backbone returns a dict with 'low' and 'high' keys
            backbone_features = self.backbone.backbone(x)
            features = backbone_features['high']  # Use high-level features for classification
        
        # Ensure features have the correct dimensions for pooling
        if features.dim() == 4:  # [B, C, H, W]
            # Apply global average pooling and flatten for classification
            features_pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features_flattened = torch.flatten(features_pooled, 1)
        else:
            # If features are already flattened or have wrong dimensions, handle appropriately
            features_flattened = features
        
        # Classification output (computed from unconditioned features)
        cls_output = self.classifier(features_flattened)
        
        # Get segmentation output from backbone
        backbone_output = self.backbone(x)
        seg_output = backbone_output['out']
        
        # Apply tooth ID conditioning for segmentation (only for segmentation and multi-task modes)
        if self.mode in ['segmentation', 'multi-task'] and tooth_ids is not None:
            # Get tooth class indices from one-hot encoding
            tooth_class = torch.argmax(tooth_ids, dim=1)  # [B]
            
            # Get tooth ID embeddings
            tooth_embeddings = self.tooth_id_embedding(tooth_class)  # [B, 64]
            
            # Project embeddings to segmentation space and add as bias
            tooth_bias = self.tooth_id_projection(tooth_embeddings)  # [B, num_seg_classes]
            tooth_bias = tooth_bias.unsqueeze(-1).unsqueeze(-1)  # [B, num_seg_classes, 1, 1]
            
            # Add tooth ID bias to segmentation features
            seg_output = seg_output + tooth_bias
        
        return seg_output, cls_output
    
    def training_step(self, batch, batch_idx):
        """Training step supporting three modes: classification only, segmentation only, and multi-task"""
        images, masks, tooth_ids = batch
        
        # Convert masks to long for CrossEntropyLoss
        masks_long = masks.long()
        
        # Get ground truth tooth class indices from one-hot encoding
        tooth_class = torch.argmax(tooth_ids, dim=1)
        
        # Forward pass - pass tooth_ids for tooth-specific conditioning
        seg_output, cls_output = self(images, tooth_ids)
        
        # Initialize losses
        seg_loss = torch.tensor(0.0, device=self.device)
        cls_loss = torch.tensor(0.0, device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Calculate losses based on training mode
        if self.mode in ['segmentation', 'multi-task']:
            seg_loss = self.seg_criterion(seg_output, masks_long)
        
        if self.mode in ['classification', 'multi-task']:
            cls_loss = self.cls_criterion(cls_output, tooth_class)
        
        # Calculate total loss based on mode
        if self.mode == 'segmentation':
            total_loss = seg_loss
        elif self.mode == 'classification':
            total_loss = cls_loss
        else:  # multi-task
            total_loss = self.seg_loss_weight * seg_loss + self.cls_loss_weight * cls_loss
        
        # Calculate segmentation predictions
        seg_preds = torch.softmax(seg_output, dim=1)
        seg_preds_class = torch.argmax(seg_preds, dim=1)
        
        # Calculate classification predictions
        cls_preds = torch.softmax(cls_output, dim=1)
        cls_preds_class = torch.argmax(cls_preds, dim=1)
        
        # Update segmentation metrics (only if segmentation is active)
        if self.mode in ['segmentation', 'multi-task']:
            self.train_seg_iou(seg_preds_class, masks_long)
            self.train_seg_acc(seg_preds_class, masks_long)
            self.train_seg_acc_per_class(seg_preds_class, masks_long)
        
        # Update classification metrics (only if classification is active)
        if self.mode in ['classification', 'multi-task']:
            self.train_cls_acc(cls_preds_class, tooth_class)
            self.train_cls_f1(cls_preds_class, tooth_class)
            self.train_cls_precision(cls_preds_class, tooth_class)
            self.train_cls_recall(cls_preds_class, tooth_class)
            self.train_cls_acc_per_class(cls_preds_class, tooth_class)
        
        # Log losses and metrics based on mode
        self.log('train_total_loss', total_loss, prog_bar=True, sync_dist=True)
        
        if self.mode in ['segmentation', 'multi-task']:
            self.log('train_seg_loss', seg_loss, prog_bar=False, sync_dist=True)
            self.log('train_seg_iou', self.train_seg_iou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_seg_acc', self.train_seg_acc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.mode in ['classification', 'multi-task']:
            self.log('train_cls_loss', cls_loss, prog_bar=False, sync_dist=True)
            self.log('train_cls_acc', self.train_cls_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_cls_f1', self.train_cls_f1, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_cls_precision', self.train_cls_precision, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_cls_recall', self.train_cls_recall, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log per-class accuracy for detailed analysis (only log mean for progress bar)
        if self.mode in ['classification', 'multi-task']:
            cls_acc_per_class = self.train_cls_acc_per_class.compute()
            self.log('train_cls_acc_mean_per_class', cls_acc_per_class.mean(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.mode in ['segmentation', 'multi-task']:
            seg_acc_per_class = self.train_seg_acc_per_class.compute()
            self.log('train_seg_acc_mean_per_class', seg_acc_per_class.mean(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step supporting three modes: classification only, segmentation only, and multi-task"""
        images, masks, tooth_ids = batch
        
        # Convert masks to long for CrossEntropyLoss
        masks_long = masks.long()
        
        # Get ground truth tooth class indices from one-hot encoding
        tooth_class = torch.argmax(tooth_ids, dim=1)
        
        # Forward pass - pass tooth_ids for tooth-specific conditioning
        seg_output, cls_output = self(images, tooth_ids)
        
        # Initialize losses
        seg_loss = torch.tensor(0.0, device=self.device)
        cls_loss = torch.tensor(0.0, device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Calculate losses based on training mode
        if self.mode in ['segmentation', 'multi-task']:
            seg_loss = self.seg_criterion(seg_output, masks_long)
        
        if self.mode in ['classification', 'multi-task']:
            cls_loss = self.cls_criterion(cls_output, tooth_class)
        
        # Calculate total loss based on mode
        if self.mode == 'segmentation':
            total_loss = seg_loss
        elif self.mode == 'classification':
            total_loss = cls_loss
        else:  # multi-task
            total_loss = self.seg_loss_weight * seg_loss + self.cls_loss_weight * cls_loss
        
        # Calculate segmentation predictions
        seg_preds = torch.softmax(seg_output, dim=1)
        seg_preds_class = torch.argmax(seg_preds, dim=1)
        
        # Calculate classification predictions
        cls_preds = torch.softmax(cls_output, dim=1)
        cls_preds_class = torch.argmax(cls_preds, dim=1)
        
        # Update segmentation metrics (only if segmentation is active)
        if self.mode in ['segmentation', 'multi-task']:
            self.val_seg_iou(seg_preds_class, masks_long)
            self.val_seg_acc(seg_preds_class, masks_long)
            self.val_seg_acc_per_class(seg_preds_class, masks_long)
        
        # Update classification metrics (only if classification is active)
        if self.mode in ['classification', 'multi-task']:
            self.val_cls_acc(cls_preds_class, tooth_class)
            self.val_cls_f1(cls_preds_class, tooth_class)
            self.val_cls_precision(cls_preds_class, tooth_class)
            self.val_cls_recall(cls_preds_class, tooth_class)
            self.val_cls_acc_per_class(cls_preds_class, tooth_class)
        
        # Log losses and metrics based on mode
        self.log('val_total_loss', total_loss, prog_bar=True, sync_dist=True)
        
        if self.mode in ['segmentation', 'multi-task']:
            self.log('val_seg_loss', seg_loss, prog_bar=False, sync_dist=True)
            self.log('val_seg_iou', self.val_seg_iou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_seg_acc', self.val_seg_acc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.mode in ['classification', 'multi-task']:
            self.log('val_cls_loss', cls_loss, prog_bar=False, sync_dist=True)
            self.log('val_cls_acc', self.val_cls_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_cls_f1', self.val_cls_f1, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_cls_precision', self.val_cls_precision, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_cls_recall', self.val_cls_recall, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log per-class accuracy for detailed analysis (only log mean for progress bar)
        if self.mode in ['classification', 'multi-task']:
            cls_acc_per_class = self.val_cls_acc_per_class.compute()
            self.log('val_cls_acc_mean_per_class', cls_acc_per_class.mean(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.mode in ['segmentation', 'multi-task']:
            seg_acc_per_class = self.val_seg_acc_per_class.compute()
            self.log('val_seg_acc_mean_per_class', seg_acc_per_class.mean(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step for multi-task learning"""
        images, masks, tooth_ids = batch
        
        # Convert masks to long for CrossEntropyLoss
        masks_long = masks.long()
        
        # Get ground truth tooth class indices from one-hot encoding
        tooth_class = torch.argmax(tooth_ids, dim=1)
        
        # Forward pass - pass tooth_ids for tooth-specific conditioning
        seg_output, cls_output = self(images, tooth_ids)
        
        # Calculate segmentation predictions
        seg_preds = torch.softmax(seg_output, dim=1)
        seg_preds_class = torch.argmax(seg_preds, dim=1)
        
        # Calculate classification predictions
        cls_preds = torch.softmax(cls_output, dim=1)
        cls_preds_class = torch.argmax(cls_preds, dim=1)
        
        # Update test metrics
        self.test_seg_iou(seg_preds_class, masks_long)
        self.test_seg_acc(seg_preds_class, masks_long)
        self.test_cls_acc(cls_preds_class, tooth_class)
        self.test_cls_f1(cls_preds_class, tooth_class)
        
        # Log test metrics
        self.log('test_seg_iou', self.test_seg_iou, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_seg_acc', self.test_seg_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log('test_cls_acc', self.test_cls_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_cls_f1', self.test_cls_f1, prog_bar=False, on_step=False, on_epoch=True)
        
        # Save predictions for visualization
        return {
            'images': images,
            'masks': masks_long,
            'seg_preds': seg_preds_class,
            'cls_preds': cls_preds_class,
            'tooth_ids': tooth_ids
        }
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss',
                'frequency': 1
            }
        }
