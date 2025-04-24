import torch
import numpy as np
import random
import yaml
from argparse import Namespace

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def dict_to_namespace(d):
    ns = Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns

def load_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)