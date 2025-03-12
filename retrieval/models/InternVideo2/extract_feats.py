import os
import sys
sys.path.append(os.getcwd())
import json
import pickle
import cv2
from tqdm import tqdm
from demo.config import Config, eval_dict_leaf
from demo.utils import _frame_from_video, setup_internvideo2, frames2tensor, get_text_feat_dict
import decord
from decord import VideoReader

def load_model(config_path, model_path):
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)
    config['pretrained_path'] = model_path
    device = 'cuda'
    intern_model, _ = setup_internvideo2(config)
    intern_model.to(device)

    return intern_model


def extract_video_feat(data_dir, output_dir, intern_model, device='cuda'):
    video_feats = {}
    all_videos = os.listdir(data_dir)
    fn = 4
    size_t = 224
    for video_id in tqdm(all_videos):
        video_path = os.path.join(data_dir, video_id)
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
        video_feature = intern_model.get_vid_feat(frames_tensor).cpu().numpy()
        video_feats[video_id.split('.')[0]] = video_feature
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'video_feats.pkl'), 'wb') as f:
        pickle.dump(video_feats, f)
    
    return


def extract_text_feat(ann_path, output_dir, intern_model):
    with open(ann_path, 'r') as f:
        data = json.load(f)
    query2qid = {item['howto100m_query_text']: item['qid'] for item in data}

    queries = [item['howto100m_query_text'] for item in data]
    text_feats = get_text_feat_dict(queries, intern_model)
    results = {query2qid[query]: feat.cpu().numpy() for query, feat in text_feats.items()}

    with open(os.path.join(output_dir, f'text_feats.pkl'), 'wb') as f:
        pickle.dump(results, f)

    return