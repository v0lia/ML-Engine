# checkpoint.py

from pathlib import Path
from datetime import datetime

import torch

from src.optimizer.get_optimizer import get_optimizer
from src.models.get_model import get_model
from src.utils.logger import get_logger
from src.utils.defaults import DATETIME_FORMAT, PROJECT_ROOT, default_checkpoints_path

def save_checkpoint(run_dir, model, optimizer, epoch, config, name=None) -> Path:
    logger = get_logger()
        
    checkpoint_path = get_checkpoint_path(run_dir=run_dir, name=name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    saved_config = config if config is not None else make_optimizer_config()

    torch.save({
        "model_class": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "config": saved_config,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch)
    }, checkpoint_path)

    logger.info(f"[CHECKPOINT] saved to: {checkpoint_path}")
    return checkpoint_path.resolve()

def load_checkpoint(checkpoint_path, config=None, device:torch.device=torch.device('cpu')):
    logger = get_logger()
    checkpoint_path = find_checkpoint_by_name(checkpoint_path)
    if checkpoint_path is None:
        error_text = f"[CHECKPOINT.load_checkpoint] Not found any .pt files, nothing to load"
        logger.error(error_text)
        raise RuntimeError(error_text)

    logger.info(f"[CHECKPOINT] Loading from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_class = checkpoint.get("model_class")
    if model_class is None:
        error_text = f'[CHECKPOINT] Not found model_class {model_class} in checkpoint: {checkpoint_path}'
        logger.error(error_text)
        raise RuntimeError(error_text)    
    
    model = get_model(model_class)
    model.load_state_dict(checkpoint["model_state_dict"])

    saved_config = checkpoint.get("config", None)
    if saved_config is None:
        if config is not None:
            error_text = f"[CHECKPOINT] Not found optimizer_config in checkpoint: {checkpoint_path}. Using general config"
            logger.warning(error_text)
            saved_config = config
        else:
            error_text = f"[CHECKPOINT] Not found optimizer_config in checkpoint: {checkpoint_path}"
            logger.error(error_text)     
            raise RuntimeError(error_text)
  
    optimizer = get_optimizer(model, saved_config)

    if "optimizer_state_dict" in checkpoint:
        try:
            opt_state = checkpoint["optimizer_state_dict"]
            for _, v in opt_state.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                       if isinstance(sv, torch.Tensor):
                        v[sk] = sv.to('cpu') 
            optimizer.load_state_dict(opt_state)
        except Exception as e:
            logger.error(f"[CHECKPOINT] optimizer.load_state_dict failed: {e}")
            raise e

    start_epoch = int(checkpoint.get("epoch", 0))
    return model.to(device), model_class, optimizer, start_epoch

def get_checkpoint_path(run_dir=None, name=None):
    if name:
        filename = name
    elif run_dir:
        filename = run_dir.stem
    else:
        filename = f"{datetime.now():{DATETIME_FORMAT}}"
        
    if run_dir is None:
        return default_checkpoints_path / f'{filename}.pt'
    
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f'{filename}.pt'
    return checkpoint_path.resolve()

def _sort_key(p: Path):
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (-mtime, str(p))    

def find_latest_checkpoint(root=PROJECT_ROOT) -> Path | None:# -> Path:
    logger = get_logger()
    root = Path(root)
    if not root.exists():
        logger.warning(f"[CHECKPOINT.find_latest_checkpoint] given root {root} does not exist. Using default: {PROJECT_ROOT}")
        root = PROJECT_ROOT

    candidates = list(root.rglob('*.pt'))
    if not candidates:
        logger.info(f"[CHECKPOINT] Not found any .pt files under {root}")
        return None
    candidates.sort(key=_sort_key)
    latest = candidates[0].resolve()
    logger.info(f"[CHECKPOINT] Latest checkpoint found: {latest}")
    return latest

def find_checkpoint_by_name(raw) -> Path | None:
    """
    Try several ways to resolve checkpoint argument to an actual file path.
    Order:
      1) Path(raw) as-is
      2) Path(raw + '.pt')
      3) default checkpoints folder (both as-is and .pt)
      4) any file named raw(.pt) under PROJECT_ROOT (first match)
    """
    logger = get_logger()
    raw_path = Path(raw)    
    
    # 1: as is
    if raw_path.exists() and raw_path.is_file():
        return raw_path.resolve() 
    
    # 2: try with .pt
    cand = raw_path.with_suffix('.pt')
    if cand.exists() and cand.is_file():
        logger.info(f"[CHECKPOINT] Found: {cand.resolve() }")
        return cand.resolve()     
    
    # 3: try default checkpoints folder (both as-is and .pt)
    direct = default_checkpoints_path / raw_path.name
    if direct.exists() and direct.is_file():
        logger.info(f"[CHECKPOINT] Found: {direct.resolve()}")
        return direct.resolve()    
    direct_pt = direct.with_suffix('.pt')
    
    if direct_pt.exists() and direct_pt.is_file(): 
        logger.info(f"[CHECKPOINT] Found: {direct_pt.resolve()}")
        return direct_pt.resolve()  
    
    # 4: search under PROJECT_ROOT
    pattern = f"{raw_path.name}"
    found = list(Path(PROJECT_ROOT).rglob(pattern))
    if found:   # pick newest
        found.sort(key=_sort_key)
        logger.info(f"[CHECKPOINT] Found checkpoint by search: {found[0].resolve()}")
        return found[0].resolve()  

    pattern_pt = f"{raw_path.name}.pt"
    found = list(Path(PROJECT_ROOT).rglob(pattern_pt))
    if found:
        found.sort(key=_sort_key)
        logger.info(f"[CHECKPOINT] Found checkpoint by search: {found[0].resolve()}")
        return found[0].resolve()   

    logger.warning(f"[CHECKPOINT] Not found: {raw}")        
    return None

def make_optimizer_config(config=None):
    if not isinstance(config, dict) or config is None:
        return {"optimizer": {}}
    opt_cfg = config.get("optimizer",{})
    if not isinstance(opt_cfg, dict):
        opt_cfg = {}
    return {"optimizer": opt_cfg}
