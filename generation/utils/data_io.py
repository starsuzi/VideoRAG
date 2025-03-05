import os
import json 


def get_base_data(file_path: str = '', is_synthetic: bool = False):
    file_path = file_path if file_path else (
        f'./datasets/qa/{"synthetic" if is_synthetic else "wikihow"}.json'
    )
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def get_retrieved_videos(file_path: str = '', is_synthetic: bool = False):
    file_path = file_path if file_path else (
        f'./datasets/retrieval/{"synthetic" if is_synthetic else "wikihow"}_query2videos.json'
    )
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    results = {}
    for key, value in raw_data.items():
        results[key] = [{'video_id': video_id, 'rank': index+1} for index, video_id in enumerate(value)]
    return results


def get_scripts_for_videos(
    videos: list, 
    script_paths: list = [
        './datasets/scripts/original/{video}.txt',
        './datasets/scripts/asr/{video}.txt'
    ]
):
    scripts = []
    for video in videos:
        script = ''
        for script_path in script_paths:
            if os.path.exists(script_path.format(video=video['video_id'])):
                with open(script_path.format(video=video['video_id']), 'r') as f:
                    script = ' '.join([x.strip() for x in f.readlines()])
                break
        scripts.append(script)
    return scripts


def load_results(file_path: str):
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


def save_results(results: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
