import os
import sys
import argparse
from pathlib import Path
from utils import data_io

sys.path.append(os.path.join(Path(os.getcwd()), 'retrieval/models/InternVideo2'))
import interface
from demo.utils import InternVideo2_Stage2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video features using InternVideo2 model.")
    parser.add_argument("--config_path", type=str, default="retrieval/models/InternVideo2/demo/internvideo2_stage2_config.py", help="Path to model configuration file")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint InternVideo2-stage2_1b-224p-f4.pt")
    parser.add_argument('--is_synthetic', action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def load_model(config_path: str, model_path: str) -> InternVideo2_Stage2:
    return interface.load_model(config_path, model_path)


def extract_text_and_visual_features(
    base_dataset: list,
    model: InternVideo2_Stage2,
    video_dir: str = './datasets/videos'
) -> (dict, dict):
    return (
        interface.extract_text_features(
            base_dataset, 
            model
        ), 
        interface.extract_visual_features(
            video_dir,
            model
        )
    )


def main() -> None:
    args = parse_arguments()

    base_dataset = data_io.get_base_data(is_synthetic=args.is_synthetic)
    model = load_model(
        config_path=args.config_path,
        model_path=args.model_path
    )

    text_features, visual_features = extract_text_and_visual_features(
        base_dataset=base_dataset,
        model=model
    )

    data_io.save_features(
        features=text_features,
        save_path=f'./datasets/retrieval/{"synthetic" if args.is_synthetic else "wikihow"}/text_features.pkl'
    )
    data_io.save_features(
        features=visual_features,
        save_path=f'./datasets/retrieval/{"synthetic" if args.is_synthetic else "wikihow"}/visual_features.pkl'
    )


if __name__ == "__main__":
    main()