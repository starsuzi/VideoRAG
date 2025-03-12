import os
import json
import torch
import pickle
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", type=str, default=".", help="Directory containing preprocessed features")
    parser.add_argument("--ann_path", type=str, default="/home/kangsan/VideoRAG/overlapped_with_vids.json", help="Path to annotation file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save retrieval results")
    return parser.parse_args()


def load_data(feat_dir, ann_path):
    text_feats = pickle.load(open(os.path.join(feat_dir, "text_feats.pkl"), "rb"))
    video_feats = pickle.load(open(os.path.join(feat_dir, "video_feats.pkl"), "rb"))
    annotations = json.load(open(ann_path, "r"))
    return text_feats, video_feats, annotations


def process_data(text_feats, video_feats):
    text_ids, text_feats = zip(*text_feats.items())
    vid_ids, vid_feats = zip(*video_feats.items())

    text_ids, text_feats = list(text_ids), list(text_feats)
    vid_ids, vid_feats = list(vid_ids), list(vid_feats)

    vid_feats = torch.Tensor(np.stack(vid_feats)).squeeze()
    text_feats = torch.Tensor(np.stack(text_feats)).squeeze()

    return text_ids, text_feats, vid_ids, vid_feats


def calculate_similarity_rankings(text_feats: torch.Tensor, vid_feats: torch.Tensor) -> torch.Tensor:
    similarity_matrix = torch.matmul(text_feats, vid_feats.T)
    return torch.sort(similarity_matrix, dim=1, descending=True)[1]


def evaluate_rankings(
    rankings: torch.Tensor,
    text_ids: list,
    vid_ids: list,
    gold_rankings: dict,
    top_k: int = 1,
    save_top_k: int = 200
) -> tuple[float, dict]:
    correct_predictions = 0
    total_predictions = 0
    predictions = {}
    
    for i, pred_rank in enumerate(rankings):
        text_id = text_ids[i]
        pred_rank_vids = [vid_ids[idx] for idx in pred_rank]
        predictions[text_id] = pred_rank_vids[:save_top_k]
        
        if text_id not in gold_rankings:
            continue
            
        correct_answer = gold_rankings[text_id][0]
        if correct_answer in pred_rank_vids[:top_k]:
            correct_predictions += 1
        total_predictions += 1
    
    recall = correct_predictions / (total_predictions + 1e-10)
    return recall, predictions


def save_predictions(predictions: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(predictions, f)


def main():
    args = parse_arguments()
    text_feats, video_feats, annotations = load_data(args.feat_dir, args.ann_path)
    gold_rankings = {data['qid']: [x['video_id'] for x in data['videos']] for data in annotations}

    text_ids, text_feats, vid_ids, vid_feats = process_data(text_feats, video_feats)

    rankings = calculate_similarity_rankings(text_feats, vid_feats)
    recall, predictions = evaluate_rankings(
        rankings=rankings,
        text_ids=text_ids,
        vid_ids=vid_ids,
        gold_rankings=gold_rankings
    )

    print(f"Recall@1: {recall:.5f}")
    output_path = os.path.join(args.output_dir, 'retrieval_results.json')
    save_predictions(predictions, output_path)


if __name__ == "__main__":
    main()
