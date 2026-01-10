import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

from DiffusionGan.adversarialDiffusion import AdversarialIRLDiffusion
from MaxEntIRL.irl_config import default_config


def load_config_from_json(json_path: str):
    config = default_config

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)

    valid_fields = set(config.__dataclass_fields__.keys())

    updated_fields = []
    invalid_fields = []

    for key, value in json_config.items():
        if key in valid_fields:
            setattr(config, key, value)
            updated_fields.append(key)
        else:
            invalid_fields.append(key)
            warnings.warn(
                f"Warn: 'Key {key}' not found in default config",
                UserWarning
            )

    # Extra init for wandb
    if config.use_wandb and config.pkl_label:
        config.wandb_run_name = config.pkl_label

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    CURRENT_DIR = Path(__file__).resolve().parent

    parser.add_argument("--config", type=str,default=f"{CURRENT_DIR}/../config/MaxEntIRL.json")
    args = parser.parse_args()

    if args.config is not None:
           config = load_config_from_json(args.config)
    else:
        config = default_config

    # Initialize adversarial trainer
    trainer = AdversarialIRLDiffusion(config)

    # Setup environment
    trainer.setup_environment()

    # Run adversarial training
    final_theta, theta_ema, norm_mean, norm_std = trainer.train_adversarial(
        num_iterations=config.num_iterations)

    print(f"Final learned reward weights: {final_theta}")

    weights_dir = os.path.join(config.output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    checkpoint_path = os.path.join(weights_dir, f"adversarial_irl_results_{config.scene_location}_{config.pkl_label}.pkl")

    # Save final results
    with open(checkpoint_path, "wb") as f:
        pickle.dump({
            "features": config.feature_names,
            "final_theta": final_theta,
            "theta_ema": theta_ema,
            "norm_mean": norm_mean,
            "norm_std": norm_std
        }, f)