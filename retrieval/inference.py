import os
import torch
import numpy as np
import argparse
from utils import data_io


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_synthetic', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_data(is_synthetic):
    return (
        data_io.get_base_data(is_synthetic=is_synthetic),
        data_io.load_features(os.path.join(f'./datasets/retrieval/{"synthetic" if is_synthetic else "wikihow"}', 'query_features.pkl')),
        data_io.load_features(os.path.join(f'./datasets/retrieval/{"synthetic" if is_synthetic else "wikihow"}', 'video_features.pkl'))
    )


def preprocess_features(features):
    ids, features = zip(*features.items())
    return list(ids), torch.Tensor(np.stack(list(features))).squeeze()


def calculate_similarity_rankings(
    query_features: torch.Tensor, 
    video_features: torch.Tensor
) -> torch.Tensor:
    return torch.sort(
        torch.matmul(query_features, video_features.T), 
        dim=1, 
        descending=True
    )[1]


def evaluate_rankings(
    query_ids: list,
    video_ids: list,
    pred_rankings: torch.Tensor,
    gold_rankings: dict,
    top_k: int = 1,
    save_top_k: int = 200
) -> tuple[float, dict]:
    predictions, correct_predictions, total_predictions = {}, 0, 0

    for query_id, pred_ranking in zip(query_ids, pred_rankings):
        if query_id not in gold_rankings:
            continue

        pred_ranking_videos = [video_ids[index] for index in pred_ranking]

        correct_answer = gold_rankings[query_id][0]
        if correct_answer in pred_ranking_videos[:top_k]:
            correct_predictions += 1
        total_predictions += 1

        predictions[query_id] = pred_ranking_videos[:save_top_k]
    
    recall = correct_predictions / (total_predictions + 1e-10)
    return recall, predictions


def main():
    args = parse_arguments()

    base_dataset, query_features, video_features = load_data(args.is_synthetic)
    (query_ids, query_features), (video_ids, video_features) = \
        preprocess_features(query_features), preprocess_features(video_features)
    
    gold_rankings = {sample['qid']: [video['video_id'] for video in sample['videos']] for sample in base_dataset}
    pred_rankings = calculate_similarity_rankings(query_features, video_features)

    recall, predictions = evaluate_rankings(
        query_ids=query_ids,
        video_ids=video_ids,
        pred_rankings=pred_rankings,
        gold_rankings=gold_rankings
    )

    print(f"Recall@1: {recall:.5f}")
    data_io.save_results(predictions, f'./results/retrieval/{"synthetic" if args.is_synthetic else "wikihow"}/predictions.json')


if __name__ == "__main__":
    main()
