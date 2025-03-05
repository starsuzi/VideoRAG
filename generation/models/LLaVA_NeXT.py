# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import warnings
from decord import VideoReader, cpu
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from utils import prompt


warnings.filterwarnings("ignore")


def get_model_package(
    path: str = 'lmms-lab/LLaVA-Video-7B-Qwen2'
):
    tokenizer, model, processor, max_length = load_pretrained_model(
        path, None, 'llava_qwen',
        torch_dtype='bfloat16',
        device_map='auto'
    )
    return model.eval(), tokenizer, processor, dict(max_length=max_length)


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def load_videos(video_paths, processor, max_frames_num, fps=1, force_sample=False):
    return [
        processor.preprocess(
            load_video(video_path, max_frames_num, fps, force_sample)[0],
            return_tensors='pt'
        )['pixel_values'].cuda().bfloat16()
        for video_path in video_paths
    ]


def prepare_inputs(tokenizer: AutoTokenizer, videos: list, prompt_format: str, query: str, scripts: list):
    query = f"{DEFAULT_IMAGE_TOKEN}\n" * len(videos) + \
            prompt_format.format(
                query=query,
                scripts=prompt.linearize_scripts(scripts)
            )    
    conv = copy.deepcopy(conv_templates['qwen_1_5'])
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')


def generate(
    tokenizer: AutoTokenizer, model: AutoModel, processor: AutoProcessor,
    generation_config: dict, query: str, videos: list, scripts: list,
    prompt_format: str, max_frames: int = 32,
    video_path: str = './datasets/videos/{video_id}.mp4'
):
    videos = load_videos(
        [video_path.format(video_id=video['video_id']) for video in videos],
        processor, max_frames
    )
    input_ids = prepare_inputs(tokenizer, videos, prompt_format, query, scripts)

    generated_ids = model.generate(
        input_ids,
        images=videos if len(videos) else None,
        modalities=['video' for _ in range(len(videos))] if len(videos) else 'text',
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return output_text
