import copy, hashlib, json, yaml
from pathlib import Path

def load_config(path="config.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def deep_update(cfg, **flat):
    """deep_update(cfg, **{'service_level': 0.9, 'po.late_factor': 0}) -> копия."""
    out = copy.deepcopy(cfg)
    for k, v in flat.items():
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = v
    return out

def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
