import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModel, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import prompt


def get_model_and_processor(
    path: str = 'Qwen/Qwen2.5-VL-7B-Instruct'
):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map='auto'
    )
    processor = AutoProcessor.from_pretrained(path)
    return model, processor


def get_message(
    videos: list, video_path: str, prompt_format: str, query: str, scripts: list, max_frames: int
):
    return [
        {
            'role': 'user',
            'content': (
                [
                    {
                        'type': 'video',
                        'video': video_path.format(video_id=video['video_id']),
                        'max_pixels': 224 * 224,
                        'max_frames': max(2, max_frames),
                        'min_frames': 1,
                        'fps': 1,
                    }
                    for video in videos
                ] + [
                    {
                        'type': 'text', 
                        'text': prompt_format.format(
                            query=query, 
                            scripts=prompt.linearize_scripts(scripts)
                        )
                    }
                ]
            ),
        }
    ]


def prepare_inputs(
    processor: AutoProcessor, videos: list, video_path: str,
    prompt_format: str, query: str, scripts: list, max_frames: int
):
    message = get_message(videos, video_path, prompt_format, query, scripts, max_frames)
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
        **video_kwargs,
    ).to('cuda')
    return inputs


def generate(
    tokenizer: AutoTokenizer, model: AutoModel, processor: AutoProcessor,
    generation_config: dict, query: str, videos: list, scripts: list,
    prompt_format: str, max_frames: int = 32,
    video_path: str = './datasets/videos/{video_id}.mp4'
):
    inputs = prepare_inputs(processor, videos, video_path, prompt_format, query, scripts, max_frames)

    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text
