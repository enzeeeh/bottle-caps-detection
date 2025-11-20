"""
WandB logger for bottle-sorter with public tracking and model versioning.
"""
import os
import wandb
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class WandbLogger:
    """Enhanced WandB logger with public accessibility and model versioning."""
    
    def __init__(self, cfg: Any) -> None:
        """Initialize WandB logger with enhanced features.
        
        Args:
            cfg: Configuration object with wandb settings.
        """
        self.cfg = cfg
        self.run = None
        self.model_artifact = None
        self.enabled = "WANDB_API_KEY" in os.environ or os.path.exists(os.path.expanduser("~/.netrc"))
        
    def start_run(self) -> None:
        """Start a new WandB run with public visibility and comprehensive tracking."""
        if not self.enabled:
            self._print_setup_instructions()
            return
            
        try:
            wandb_config = getattr(self.cfg, 'wandb', None)
            if not wandb_config:
                print("⚠️ WandB configuration not found in settings.yaml")
                return
            
            # Prepare comprehensive configuration
            run_config = self._prepare_run_config()
            
            # Initialize WandB run with public visibility
            self.run = wandb.init(
                project=getattr(wandb_config, 'project_name', 'bottle-caps-detection'),
                entity=getattr(wandb_config, 'entity', None),
                tags=getattr(wandb_config, 'tags', []),
                notes=getattr(wandb_config, 'notes', 'Bottle cap detection training'),
                job_type=getattr(wandb_config, 'job_type', 'training'),
                public=getattr(wandb_config, 'public', True),  # Public accessibility
                config=run_config,
                name=f"bsort-3color-{wandb.util.generate_id()[:8]}"
            )
            
            # Log system information
            self._log_system_info()
            
            logger.info(f"🚀 WandB run started: {self.run.name}")
            print(f"📊 WandB Dashboard (PUBLIC): {self.run.url}")
            print(f"🔗 Share this link: {self.run.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self._print_setup_instructions()
            self.run = None
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log training metrics to WandB with enhanced tracking.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step/epoch number
        """
        if not self.run:
            return
            
        try:
            # Log all metrics with proper organization
            logged_metrics = {}
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Organize metrics by category
                    if 'loss' in key.lower():
                        logged_metrics[f"loss/{key}"] = value
                    elif any(metric in key.lower() for metric in ['acc', 'precision', 'recall', 'f1', 'map']):
                        logged_metrics[f"metrics/{key}"] = value
                    elif 'lr' in key.lower() or 'learning' in key.lower():
                        logged_metrics[f"learning/{key}"] = value
                    else:
                        logged_metrics[key] = value
            
            # Log with step if provided
            if step is not None:
                self.run.log(logged_metrics, step=step)
            else:
                self.run.log(logged_metrics)
                
        except Exception as e:
            logger.warning(f"Failed to log metrics to WandB: {e}")
    
    def log_model_artifact(self, model_path: str, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Log model as artifact with versioning and metadata.
        
        Args:
            model_path: Path to the saved model
            epoch: Training epoch number
            metrics: Model performance metrics
            is_best: Whether this is the best model so far
        """
        if not self.run:
            return
            
        try:
            # Create model artifact with comprehensive metadata
            artifact_name = "bottle-caps-model"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Bottle caps detection model - Epoch {epoch}",
                metadata={
                    "epoch": epoch,
                    "framework": "YOLOv8",
                    "task": "object_detection",
                    "classes": ["red", "green", "blue"],
                    "is_best": is_best,
                    **metrics
                }
            )
            
            # Add model file to artifact
            if os.path.exists(model_path):
                artifact.add_file(model_path, name="model.pt")
                
                # Log the artifact with appropriate alias
                aliases = ["latest", f"epoch_{epoch}"]
                if is_best:
                    aliases.append("best")
                    
                self.run.log_artifact(artifact, aliases=aliases)
                
                logger.info(f"📦 Model artifact logged: {artifact_name} (Epoch {epoch})")
                print(f"🏆 Best model logged!" if is_best else f"📊 Model checkpoint logged (Epoch {epoch})")
                
        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")
    
    def log_dataset_artifact(self, data_path: str, split: str = "train") -> None:
        """Log dataset as artifact for reproducibility.
        
        Args:
            data_path: Path to dataset directory
            split: Dataset split (train/val/test)
        """
        if not self.run:
            return
            
        try:
            artifact = wandb.Artifact(
                name=f"bottle-caps-dataset-{split}",
                type="dataset",
                description=f"Bottle caps detection dataset - {split} split"
            )
            
            if os.path.exists(data_path):
                artifact.add_dir(data_path, name=split)
                self.run.log_artifact(artifact)
                logger.info(f"📁 Dataset artifact logged: {split} split")
                
        except Exception as e:
            logger.warning(f"Failed to log dataset artifact: {e}")
    
    def finish_run(self) -> None:
        """Finish WandB run with summary and public access information."""
        if not self.run:
            return
            
        try:
            # Add final summary
            summary = {
                "status": "completed",
                "public_url": self.run.url,
                "model_artifacts": len([a for a in self.run.logged_artifacts() if a.type == "model"]),
                "dataset_artifacts": len([a for a in self.run.logged_artifacts() if a.type == "dataset"])
            }
            
            # Update run summary
            for key, value in summary.items():
                self.run.summary[key] = value
            
            print(f"✅ Training completed! View results at: {self.run.url}")
            print(f"🌐 Public dashboard: {self.run.url}")
            
            wandb.finish()
            logger.info("WandB run finished successfully")
            
        except Exception as e:
            logger.warning(f"Error finishing WandB run: {e}")
            wandb.finish()
    
    def _prepare_run_config(self) -> Dict[str, Any]:
        """Prepare comprehensive run configuration."""
        config = {}
        
        # Add model configuration
        if hasattr(self.cfg, 'model'):
            config['model'] = {
                'architecture': getattr(self.cfg.model, 'name', 'yolov8'),
                'variant': getattr(self.cfg.model, 'variant', 'yolov8n'),
                'classes': getattr(self.cfg.model, 'classes', ["red", "green", "blue"]),
                'input_size': getattr(self.cfg.model, 'input_size', 640)
            }
        
        # Add training configuration
        if hasattr(self.cfg, 'training'):
            config['training'] = {
                'epochs': getattr(self.cfg.training, 'epochs', 100),
                'batch_size': getattr(self.cfg.training, 'batch_size', 16),
                'learning_rate': getattr(self.cfg.training, 'lr', 0.01),
                'optimizer': getattr(self.cfg.training, 'optimizer', 'AdamW'),
                'patience': getattr(self.cfg.training, 'patience', 50)
            }
        
        # Add data configuration
        if hasattr(self.cfg, 'data'):
            config['data'] = {
                'train_split': getattr(self.cfg.data, 'train_split', 0.8),
                'val_split': getattr(self.cfg.data, 'val_split', 0.2),
                'augmentation': getattr(self.cfg.data, 'augmentation', True)
            }
        
        return config
    
    def _log_system_info(self) -> None:
        """Log system and environment information."""
        if not self.run:
            return
            
        try:
            import platform
            import psutil
            
            system_info = {
                "system/platform": platform.platform(),
                "system/python_version": platform.python_version(),
                "system/cpu_count": psutil.cpu_count(),
                "system/memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            }
            
            # Add GPU info if available
            try:
                import torch
                if torch.cuda.is_available():
                    system_info["system/gpu_name"] = torch.cuda.get_device_name(0)
                    system_info["system/gpu_memory_gb"] = round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                    )
                    system_info["system/cuda_version"] = torch.version.cuda
                else:
                    system_info["system/gpu"] = "CPU only"
            except ImportError:
                system_info["system/gpu"] = "PyTorch not available"
            
            self.run.log(system_info)
            
        except Exception as e:
            logger.warning(f"Could not log system info: {e}")
    
    def _print_setup_instructions(self) -> None:
        """Print WandB setup instructions for public tracking."""
        print("\n" + "="*60)
        print("🔧 WANDB SETUP REQUIRED FOR PUBLIC MODEL TRACKING")
        print("="*60)
        print("To enable public model tracking and versioning:")
        print()
        print("1. Create account: https://wandb.ai/signup")
        print("2. Get API key: https://wandb.ai/authorize")
        print("3. Set API key: wandb login")
        print("   Or set: $env:WANDB_API_KEY='your-key-here'")
        print()
        print("📊 Benefits of public tracking:")
        print("   • Share model results with public URLs")
        print("   • Track model versions and artifacts")
        print("   • Compare training runs")
        print("   • Professional ML portfolio")
        print()
        print("Training will continue without tracking...")
        print("="*60 + "\n")

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics to wandb."""
        if self.enabled and self.run:
            wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str) -> None:
        """Log model artifact to wandb."""
        if self.enabled and self.run:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def start_run(self) -> None:
        """Start a new wandb run."""
        if self.enabled and not self.run:
            self.run = wandb.init()

    def finish_run(self) -> None:
        """Finish the current wandb run."""
        if self.enabled and self.run:
            self.run.finish()
            self.run = None
    
    def _config_to_dict(self, cfg: Any) -> dict:
        """Convert config object to dictionary for WandB."""
        config_dict = {}
        for attr in dir(cfg):
            if not attr.startswith('_'):
                val = getattr(cfg, attr)
                if hasattr(val, '__dict__'):
                    # Handle nested config objects
                    config_dict[attr] = {k: v for k, v in val.__dict__.items() if not k.startswith('_')}
                else:
                    config_dict[attr] = val
        return config_dict
