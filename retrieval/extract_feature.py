import os
import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(os.getcwd())
sys.path.append(str(PROJECT_ROOT / 'models/InternVideo2'))

from extract_feats import load_model, extract_video_feat, extract_text_feat


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video features using InternVideo2 model.")
    parser.add_argument("--config", type=str, default="models/InternVideo2/demo/internvideo2_stage2_config.py", help="Path to model configuration file")
    parser.add_argument("--model_path", type=str, default="/c1/kangsan/checkpoints/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt", help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="/c2/soyeong/video-rag/dataset/HowTo100M/preprocessed/video_data", help="Directory containing input video data")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save extracted features")
    parser.add_argument("--ann_path", type=str, default="/home/kangsan/VideoRAG/overlapped_with_vids.json", help="Path to annotation file")

    return parser.parse_args()


def process_videos(
    config: str,
    model_path: str,
    data_dir: str,
    ann_path: str,
    output_dir: str
) -> None:
    intern_model = load_model(config, model_path)
    extract_video_feat(data_dir, output_dir, intern_model)
    extract_text_feat(ann_path, output_dir, intern_model)


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    process_videos(
        config=args.config,
        model_path=args.model_path,
        data_dir=args.data_dir,
        ann_path=args.ann_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()