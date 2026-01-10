import argparse
import os

from DiffusionGan.inference import AdversarialIRLDiffusionInference
from MaxEntIRL.irl_config import default_config
from scripts.train import load_config_from_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    if args.config is not None:
        config = load_config_from_json(args.config)
    else:
        config = default_config

    # special fix for infer
    if config.num_scenes_to_infer:
        config.num_scenes_to_evaluate = config.num_scenes_to_infer

    output_dir = "./infer_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run inference with rendering
    inferencer = AdversarialIRLDiffusionInference(config, output_dir)
    inferencer.inference_adversarial(
        render_to_video=True,  # Set to True to generate videos
        render_to_img=False  # Set to True to generate only first frame images
    )