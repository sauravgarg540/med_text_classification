from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from dataclasses import dataclass
from src.utils.log_utils import Logger

logger = Logger.get_logger(__name__)

@dataclass
class ExperimentConfig:
    """Data class to hold experiment configuration"""
    project: Dict[str, Any]
    data: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    lora: Dict[str, Any]
    wandb: Dict[str, Any]

class ConfigManager:
    """Manager class for handling experiment configurations"""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the base configuration file
        """
        self.config_path = Path(config_path)
        self.config: Optional[ExperimentConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load the configuration from the YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            self.config = ExperimentConfig(
                project=config_dict['project'],
                data=config_dict['data'],
                model=config_dict['model'],
                training=config_dict['training'],
                lora=config_dict['lora'],
                wandb=config_dict['wandb']
            )
            logger.info(f"Successfully loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise RuntimeError(f"Configuration loading failed: {str(e)}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.model
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.data
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.training
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration."""
        return self.config.lora
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """Get Weights & Biases configuration."""
        return self.config.wandb
    
    def get_project_config(self) -> Dict[str, Any]:
        """Get project configuration."""
        return self.config.project
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.
        
        Args:
            updates (Dict[str, Any]): Dictionary containing configuration updates
        """
        try:
            for section, values in updates.items():
                if hasattr(self.config, section):
                    current = getattr(self.config, section)
                    current.update(values)
                    setattr(self.config, section, current)
            logger.info("Successfully updated configuration")
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            raise RuntimeError(f"Configuration update failed: {str(e)}")
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            path (Optional[str]): Path to save the configuration. If None, uses the original path.
        """
        try:
            save_path = Path(path) if path else self.config_path
            config_dict = {
                'project': self.config.project,
                'data': self.config.data,
                'model': self.config.model,
                'training': self.config.training,
                'lora': self.config.lora,
                'wandb': self.config.wandb
            }
            
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Successfully saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise RuntimeError(f"Configuration saving failed: {str(e)}") 