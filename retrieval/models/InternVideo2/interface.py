import os
import json
import pickle
import cv2
from tqdm import tqdm
from demo.config import Config, eval_dict_leaf
from demo.utils import _frame_from_video, setup_internvideo2, frames2tensor, get_text_feat_dict
import decord
from decord import VideoReader


def load_model(config_path, model_path, device='cuda'):
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)
    config.model.vision_encoder.pretrained = model_path
    config['pretrained_path'] = model_path

    model, _ = setup_internvideo2(config)
    model.to(device)

    return model


def extract_video_features(video_dir, model, device='cuda', fn=4, size_t=224):
    results = {}

    for video_id in tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_id)
        if not video_path.endswith(('.mp4', '.webm')):
            print(f'[WARNING] Video path does not end with .mp4 or .webm: {video_path}')
            continue

        try:
            vr = VideoReader(video_path, ctx=decord.cpu(), num_threads=1)
        except Exception as e:
            print(f'[ERROR] Failed to load video: {video_path}')
            print(f'[ERROR] Error: {e}')
            continue

        total_frames = len(vr)
        if total_frames == 0:
            print('[ERROR] Video length is zero', video_path)
            continue

        indices = [x + (total_frames // fn) // 2 for x in range(0, total_frames, total_frames // fn)[:fn]]
        indices[-1] = min(indices[-1], total_frames - 1)
        frames = [x[..., ::-1] for x in vr.get_batch(indices).asnumpy()]
        frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)
        video_feature = model.get_vid_feat(frames_tensor).cpu().numpy()
        results[video_id.split('.')[0]] = video_feature

    return results


def extract_query_features(data, model):
    query2qid = {item['howto100m_query_text']: item['qid'] for item in data}
    queries = [item['howto100m_query_text'] for item in data]

    return {
        query2qid[query]: feature.cpu().numpy() 
        for query, feature 
        in get_text_feat_dict(queries, model).items()
    }
