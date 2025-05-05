from yacs.config import CfgNode as CN
from ml.utils.log_utils import logger
import yaml
import os.path as op

_C = CN()

# Project settings
_C.PROJECT = CN()
_C.PROJECT.NAME = "text-classification"
_C.PROJECT.RUN_NAME = "base-run"
_C.PROJECT.SEED = 42
_C.PROJECT.DEVICE = "cuda"
_C.PROJECT.TRAIN=True

# Data settings
_C.DATA = CN()
_C.DATA.PATH = "data/processed_data.xlsx"
_C.DATA.TEST_SIZE = 0.3
_C.DATA.MAX_LENGTH = 512
_C.DATA.REQUIRED_COLUMNS = ["text", "label"]

# Model settings
_C.MODEL = CN()
_C.MODEL.NAME = "bert-base-uncased"
_C.MODEL.DO_LOWER_CASE = True
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.LABELS = ["Non-Cancer", "Cancer"]
_C.MODEL.LORA_ADAPTER = None

# Training settings
_C.TRAINING = CN()
_C.TRAINING.LEARNING_RATE = 2e-5
_C.TRAINING.BATCH_SIZE = 16
_C.TRAINING.NUM_EPOCHS = 3
_C.TRAINING.WEIGHT_DECAY = 0.01
_C.TRAINING.WARMUP_STEPS = 0
_C.TRAINING.GRADIENT_ACCUMULATION_STEPS = 1
_C.TRAINING.MAX_GRAD_NORM = 1.0
_C.TRAINING.SAVE_STRATEGY = "epoch"
_C.TRAINING.EVALUATION_STRATEGY = "epoch"
_C.TRAINING.LOGGING_STRATEGY = "epoch"
_C.TRAINING.LOAD_BEST_MODEL_AT_END = True
_C.TRAINING.METRIC_FOR_BEST_MODEL = "f1"
_C.TRAINING.GREATER_IS_BETTER = True
_C.TRAINING.SAVE_TOTAL_LIMIT = 2
_C.TRAINING.OPTIM = "adam"
_C.TRAINING.SAVE_STEPS = 100


# LoRA settings
_C.LORA = CN()
_C.LORA.ENABLED = True
_C.LORA.TASK_TYPE = "SEQ_CLS"
_C.LORA.R = 4
_C.LORA.ALPHA = 32
_C.LORA.DROPOUT = 0.01
_C.LORA.TARGET_MODULES = ["query", "key", "value"]

# Weights & Biases settings
_C.WANDB = CN()
_C.WANDB.ENABLED = True
_C.WANDB.PROJECT = "text-classification"
_C.WANDB.TAGS = ["bert", "text-classification"]
_C.WANDB.NOTES = "Base experiment configuration"


def _update_config_from_file(config, cfg_file):
    """Update config from file.
    
    Args:
        config: config to update
        cfg_file: path to config file
    """

    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, op.join(op.dirname(cfg_file), cfg))
    logger.info('Merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    """Update config from file and command line arguments.
    
    Args:
        config: config to update
        args: arguments
    """
    _update_config_from_file(config, args.cfg)

    # config.defrost()
    # config.merge_from_list(args.opts)
    # config.freeze()


def save_config(cfg, path):
    """Save config to file.
    
    Args:
        cfg: config to save
        path: path to save config
    """
    with open(path, 'w') as f:
        f.write(cfg.dump())