from models.criterions import get_sim
import pickle
import json
import torch
import numpy as np
from tqdm import tqdm

# with open('/c1/kangsan/videorag/howto100m_feats/howto100m_textfeats_all.pkl', 'rb') as f:
with open('/c1/kangsan/videorag/howto100m_feats/synthetic_textfeats_all.pkl', 'rb') as f:
    text_dict = pickle.load(f)
with open('/c1/kangsan/videorag/validset/mlp_vidfeats4f_k16_pick3_all.pkl', 'rb') as f:
# with open('/c1/kangsan/videorag/validset/mlp_vidfeats4f_all.pkl', 'rb') as f:
    vid_dict = pickle.load(f) 
with open('/c1/kangsan/videorag/howto100m_feats/howto100m_scripttextfeats_all_asr.pkl', 'rb') as f:
    script_dict = pickle.load(f) 
# with open('/home/kangsan/VideoRAG/overlapped_with_vids.json', 'r') as f:
with open('/c2/soyeong/video-rag/data_generation/outputs/video_total/0_178/same_format_w_howto100m/gpt_qa_data.json', 'r') as f:
    jf = json.load(f)
gold_rankings = {}
for data in jf:
    gold_rankings[data['qid']] = [x['video_id'] for x in data['videos']]

text_ids, vid_ids = [], []
text_feats, vid_feats = [], []
script_ids, script_feats = [], []
weighted_ids, weighted_feats = [], []

for k, v in text_dict.items():
    text_ids.append(k)
    text_feats.append(v.squeeze().cpu())
for k, v in vid_dict.items():
    vid_ids.append(k)
    vid_feats.append(v.squeeze())
for k, v in script_dict.items():
    script_ids.append(k)
    script_feats.append(v.squeeze().cpu())

    if k in vid_dict:
        weighted_ids.append(k)
        vfeat = vid_dict[k].squeeze()
        alpha = 0.1
        weighted_feats.append(alpha * v.squeeze().cpu() + (1-alpha) * vfeat)

vid_feats = np.stack(vid_feats)
vid_feats = torch.Tensor(vid_feats) # Shape : [8633, 512]
text_feats = np.stack(text_feats)
text_feats = torch.Tensor(text_feats) # Shape : [534, 512]
script_feats = np.stack(script_feats)
script_feats = torch.Tensor(script_feats)
weighted_feats = torch.Tensor(np.stack(weighted_feats))

vid_feats = weighted_feats
vid_ids = weighted_ids

similarity_matrix = torch.matmul(text_feats, vid_feats.T)
_, rankings = torch.sort(similarity_matrix, dim=1, descending=True)
# print('Alpha', alpha)
n_candidates = [1, 5, 10]
for n_cand in n_candidates:
    score = []
    saves = {}
    for i, pred_rank in enumerate(rankings):
        pred_rank_vids = [vid_ids[x] for x in pred_rank]
        saves[text_ids[i]] = pred_rank_vids[:200]
        candidates = pred_rank_vids[:n_cand]
        try:
            # answer = gold_rankings[text_ids[i]][:5]
            answer = gold_rankings[text_ids[i]][0]
        except:
            continue
        if answer in candidates:
        # if any(x in candidates for x in answer):
            score.append(1)
        else:
            score.append(0)
    
    print(f"Recall@{n_cand}: {round(sum(score)/(len(score)+1e-10), 5)}")

    with open('/home/kangsan/VideoRAG/retrieval_results_ensemble05_mlp_k16_pick3_synthetic.json', 'w') as f:
        json.dump(saves, f)
    break