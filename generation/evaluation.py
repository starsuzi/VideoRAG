import json
import argparse
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from utils import data_io


def calculate_rouge(results):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return np.mean(
        [
            scorer.score(result['gt'], result['pred'])['rougeL'].fmeasure 
            for result in results.values()
        ]
    )


def calculate_bleu(results):
    return np.mean(
        [
            sentence_bleu([result['gt'].split()], result['pred'].split(), smoothing_function=SmoothingFunction().method1)
            for result in results.values()
        ]
    )


def calculate_bert_score(results):
    _, _, f1_scores = bert_score(
        [result['pred'].replace('\n', ' ') for result in results.values()], 
        [result['gt'].replace('\n', ' ') for result in results.values()], 
        lang='en', 
        verbose=False
    )
    return np.mean(f1_scores.tolist())


def evaluate(results):
    return {
        'ROUGE-L': round(calculate_rouge(results), 5),
        'BLEU-4': round(calculate_bleu(results), 5),
        'BERTScore': round(calculate_bert_score(results), 5)
    }


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file_path', type=str, default='./results/generation/wikihow/InternVL2_5-8B/VideoRAG-VT.json')
    args = argparser.parse_args()

    results = data_io.load_results(args.file_path)

    print(f"Evaluation results for file: {args.file_path}")
    print(json.dumps(evaluate(results), indent=4))
