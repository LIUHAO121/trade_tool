import json
import logging
from pathlib import Path


def load_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data


def dump_json(json_path, data):
    with open(json_path, "w") as json_file:
        json.dump(data, json_file)
        

def setup_logging(log_dir, log_level, trace_id):
    log_cfg = {
        "level": log_level,
        "format": f"$asctime|$process|$name|$filename:$lineno|$levelname|] $message",
        "style": "$",
    }
    if log_dir:
        log_fn = Path(log_dir)
        log_cfg.update(
            {
                "filename": str(log_fn / f"{trace_id}.log"),
                "filemode": "a",
            }
        )
        log_fn.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(**log_cfg)
    logging.captureWarnings(True)
 
def get_current_logger():
    return logging.getLogger()   