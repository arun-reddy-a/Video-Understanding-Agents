"""Answer each question from each condition's frames the way lmms-eval's videomme task does with Qwen2.5-VL

Protocol (lmms-eval qwen2_5_vl wrapper and videomme task as of early 2026):
  system prompt 'You are a helpful assistant.'; one video item before the text, no timestamps
  text: the task's instruction, the question, the options one per line, then
        "\\nAnswer with the option's letter from the given choices directly."
  greedy decoding, at most 16 new tokens; the first A-D in the reply (after stripping answer
  prefixes) is the prediction
  video temporal positions use the processor's default 2 fps, as lmms-eval passes no fps
Frames come from export_videomme.py already at qwen-vl-utils' size; results go to <bundle_dir>/reader_results.jsonl. Each row also keeps the
first decoding step's distribution over A-D, which the analysis needs and lmms-eval does not report.

Usage: python read_videomme.py --bundle-dir DIR [--conditions uniform evidential]
"""
import re
import json
import time
import tyro
import torch
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from jsonl import read_resumable
from lmms_protocol.qwen_video_frames import FPS, LMMS_MAX_PIXELS, LMMS_MIN_PIXELS

HERE = Path(__file__).resolve().parent

LETTERS = 'ABCD'

SYSTEM_PROMPT = 'You are a helpful assistant.'
OPTION_PROMPT = (
    'Select the best answer to the following multiple-choice question based on the video and the subtitles. '
    'Respond with only the letter (A, B, C, or D) of the correct option.'
)
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."

# lmms-eval's videomme extract_characters_regex, before the April 2026 parser change; the missing
# commas are in the original and make two of these prefixes single concatenated strings
ANSWER_PREFIXES = [
    'The best answer is',
    'The correct answer is',
    'The answer is',
    'The answer',
    'The best option is' 'The correct option is',
    'Best answer:' 'Best option:',
]


@dataclass
class Config:
    bundle_dir:     Path      = HERE / 'bundles' / 'evidential_v1'
    conditions:     list[str] = field(default_factory=list)    # default: every condition in the manifest
    model_id:       str       = 'Qwen/Qwen2.5-VL-7B-Instruct'
    max_new_tokens: int       = 16
    device:         str       = 'cuda'


def build_prompt(item):
    question = item['question'] + '\n' + '\n'.join(item['options'])

    return OPTION_PROMPT + '\n' + question + '\n' + POST_PROMPT


def extract_answer(response):
    response = response.strip()
    for prefix in ANSWER_PREFIXES:
        response = response.replace(prefix, '')

    # transcribed line for line from lmms-eval, though the next check alone gives the same result
    if len(response.split()) > 10 and not re.search('[ABCD]', response):
        return ''

    match = re.search('[ABCD]', response)

    return match[0] if match else ''


def load_video(bundle_dir, item, frame_keys):
    """Frames in temporal order (the manifest lists them so) as the uint8 (num_frames, 3, height, width) tensor
    qwen-vl-utils would hand the processor"""
    frames = [
        np.asarray(Image.open(bundle_dir / 'frames' / item['video'] / f'{frame_key}.jpg').convert('RGB'))
        for frame_key
        in frame_keys
    ]

    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


def load_done(output_path):
    """(question id, condition) pairs already answered, so a run can resume"""
    return {(row['qid'], row['condition']) for row in read_resumable(output_path)}


@torch.no_grad()
def answer(model, processor, letter_token_ids, item, video, device, max_new_tokens):
    content = ([{'type': 'video'}] if video is not None else []) + [{'type': 'text', 'text': build_prompt(item)}]
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # lmms-eval passes no fps, so the processor's default (qwen-vl-utils' 2 fps) sets the temporal positions
    video_kwargs = {} if video is None else {'videos': [video], 'fps': FPS, 'do_sample_frames': False}
    inputs = processor(text=[text], return_tensors='pt', **video_kwargs).to(device)

    start_time = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_logits=True,
        return_dict_in_generate=True
    )
    seconds = round(time.time() - start_time, 2)

    num_prompt_tokens = inputs['input_ids'].shape[1]
    response = processor.batch_decode(output.sequences[:, num_prompt_tokens:], skip_special_tokens=True)[0]

    # raw first-step logits: output.scores would already carry the generation config's repetition penalty
    letter_probabilities = torch.softmax(output.logits[0][0, letter_token_ids].float(), dim=-1).tolist()

    return response, letter_probabilities, num_prompt_tokens, seconds


def main(config: Config):
    print(config)

    items = [json.loads(line) for line in (config.bundle_dir / 'manifest.jsonl').read_text().splitlines()]

    output_path = config.bundle_dir / 'reader_results.jsonl'
    done = load_done(output_path)

    processor = AutoProcessor.from_pretrained(config.model_id, min_pixels=LMMS_MIN_PIXELS, max_pixels=LMMS_MAX_PIXELS)
    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(config.model_id, dtype=torch.bfloat16, attn_implementation='sdpa')
        .to(config.device)
        .eval()
    )

    letter_token_ids = [
        processor.tokenizer.encode(letter, add_special_tokens=False)[0]
        for letter
        in LETTERS
    ]

    with output_path.open('a') as output_file:
        for item in items:
            for condition, frame_keys in item['conditions'].items():
                skipped = config.conditions and condition not in config.conditions
                if skipped or (item['qid'], condition) in done:
                    continue

                video = load_video(config.bundle_dir, item, frame_keys) if frame_keys else None

                response, probabilities, num_tokens, seconds = answer(
                    model,
                    processor,
                    letter_token_ids,
                    item,
                    video,
                    config.device,
                    config.max_new_tokens
                )
                prediction = extract_answer(response)

                # keys follow analysis/redundancy/reader.py, plus the raw response
                row = {
                    'qid':       item['qid'],
                    'video':     item['video'],
                    'task':      item['task'],
                    'condition': condition,
                    'n_frames':  len(frame_keys),
                    'n_tokens':  num_tokens,
                    'response':  response,
                    'pred':      prediction,
                    'answer':    item['answer'],
                    'correct':   prediction == item['answer'],
                    'p_correct': probabilities[LETTERS.index(item['answer'])],
                    'p':         probabilities,
                    'seconds':   seconds,
                }

                output_file.write(json.dumps(row) + '\n')
                output_file.flush()

                print(
                    f"{item['qid']:7} {condition:16} {num_tokens:6d} tok  {response!r:12} pred={prediction or '-'} "
                    f"ans={item['answer']}  p_correct={row['p_correct']:.2f}  {seconds}s"
                )


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
