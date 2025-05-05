from transformers import PreTrainedTokenizer

from .phi3_model import PHI3Model

def get_model(cfg, *args, **kwargs):
    if cfg.NAME == "microsoft/Phi-3-mini-4k-instruct":
        return PHI3Model(cfg, *args, **kwargs)
    else:
        raise ValueError('Unknown model name: {}'.format(cfg.NAME))


__all__ = [
    "get_model"
]
