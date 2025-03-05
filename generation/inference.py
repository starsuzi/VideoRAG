import argparse
from tqdm import tqdm
from models import InternVL, QwenVL, LLaVA_NeXT
from utils import common, data_io, prompt
from transformers import AutoModel, AutoTokenizer, AutoProcessor


def get_model_package(model_name: str):
    model, tokenizer, processor, generation_config = None, None, None, None

    if 'InternVL2_5' in model_name:
        model, tokenizer = InternVL.get_model_and_tokenizer(f'OpenGVLab/{model_name}')
        generation_config = InternVL.get_generation_config(tokenizer)
    elif 'Qwen2.5-VL' in model_name:
        model, processor = QwenVL.get_model_and_processor(f'Qwen/{model_name}')
    elif 'LLaVA-Video' in model_name:
        model, tokenizer, processor, generation_config = LLaVA_NeXT.get_model_package(f'lmms-lab/{model_name}')
    else:
        raise ValueError(f'Invalid model name: {model_name}')

    return model, tokenizer, processor, generation_config


def get_knowledge_flags(method: str):
    if method not in ['Naive', 'VideoRAG-V', 'VideoRAG-VT', 'Oracle-V', 'Oracle-VT']:
        raise ValueError(f'Invalid method: {method}')

    is_script = True if (method in ['VideoRAG-VT', 'Oracle-VT']) else False
    is_video = True if (method in ['VideoRAG-V', 'VideoRAG-VT', 'Oracle-V', 'Oracle-VT']) else False
    return is_script, is_video


def get_generation_func(model_name: str):
    generate_func = None

    if 'InternVL2_5' in model_name:
        generate_func = InternVL.generate
    elif 'Qwen2.5-VL' in model_name:
        generate_func = QwenVL.generate
    elif 'LLaVA-Video' in model_name:
        generate_func = LLaVA_NeXT.generate
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
    return generate_func


def inference(
    base_dataset: list, model: AutoModel, tokenizer: AutoTokenizer, processor: AutoProcessor,
    generation_config: dict, model_name: str = 'InternVL2_5', method: str = 'Naive',
    num_videos: int = 1, is_synthetic: bool = False
):
    is_oracle = True if 'Oracle' in method else False
    is_script, is_video = get_knowledge_flags(method)
    query2videos = data_io.get_retrieved_videos(is_synthetic=is_synthetic) if (is_video or is_script) else {}
    
    generate_func = get_generation_func(model_name)
    
    results = {}
    for sample in tqdm(base_dataset):
        query = common.normalize_text(sample['wikihow_query_text'])
        videos = (
            sample['videos'][:num_videos]
            if is_oracle else query2videos[str(sample['qid'])][:num_videos]
        ) if (is_video or is_script) else []
        scripts = data_io.get_scripts_for_videos(videos) if is_script else []

        response = generate_func(
            tokenizer=tokenizer, model=model, processor=processor,
            generation_config=generation_config,
            query=query,
            scripts=scripts,
            videos=(videos if is_video else []),
            prompt_format=prompt.METHOD2PROMPT[method],
            max_frames=32,
        )

        results[sample['qid']] = {
            'question': sample['wikihow_query_text'],
            'gt': sample['answer_text'],
            'pred': response
        }

    return results


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--model', type=str, default='InternVL2_5-8B',
        choices=['InternVL2_5-8B', 'LLaVA-Video-7B-Qwen2', 'Qwen2.5-VL-3B-Instruct', '...']
    )
    argparser.add_argument('--method', type=str, default='Naive')
    argparser.add_argument('--is_synthetic', action=argparse.BooleanOptionalAction, default=False)
    args = argparser.parse_args()

    base_dataset = data_io.get_base_data(is_synthetic=args.is_synthetic)
    model, tokenizer, processor, generation_config = get_model_package(args.model)
    
    results = inference(
        base_dataset=base_dataset, 
        model=model, 
        tokenizer=tokenizer, 
        processor=processor, 
        generation_config=generation_config, 
        model_name=args.model, 
        method=args.method,
        is_synthetic=args.is_synthetic,
    )
    data_io.save_results(results, f'./results/generation/{"synthetic" if args.is_synthetic else "wikihow"}/{args.model}/{args.method}.json')
