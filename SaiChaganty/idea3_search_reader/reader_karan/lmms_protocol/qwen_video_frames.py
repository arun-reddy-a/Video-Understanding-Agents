"""Frame indices and resizing exactly as lmms-eval's qwen2_5_vl wrapper and qwen-vl-utils produce them for Video-MME"""
import math
import torch
import numpy as np
import torch.nn.functional as F

# qwen-vl-utils vision_process.py
IMAGE_FACTOR        = 28
FPS                 = 2.0
FPS_MAX_FRAMES      = 768
VIDEO_MAX_PIXELS    = 768 * 28 * 28
VIDEO_TOTAL_PIXELS  = int(128000 * 28 * 28 * 0.9)
FRAME_FACTOR        = 2

# lmms-eval qwen2_5_vl defaults
LMMS_MIN_PIXELS     = 256 * 28 * 28
LMMS_MAX_PIXELS     = 1605632
LMMS_MAX_NUM_FRAMES = 32


def qwen_decode_count(total_frames, native_fps):
    """smart_nframes with no fps or nframes given: 2 fps, capped at 768, rounded down to even"""
    num_frames = total_frames / native_fps * FPS
    num_frames = min(max(num_frames, 4), FPS_MAX_FRAMES, total_frames)

    return int(num_frames // FRAME_FACTOR * FRAME_FACTOR)


def lmms_uniform_indices(total_frames, native_fps):
    """Native frame numbers lmms-eval feeds the reader: qwen-vl-utils decodes its 2 fps frames, then lmms-eval keeps 32 of them"""
    num_decoded = qwen_decode_count(total_frames, native_fps)
    decoded = torch.linspace(0, total_frames - 1, num_decoded).round().long().numpy()

    kept = np.unique(np.linspace(0, num_decoded - 1, LMMS_MAX_NUM_FRAMES, dtype=int))
    if num_decoded - 1 not in kept:
        kept = np.unique(np.append(kept, num_decoded - 1))

    return decoded[kept]


def qwen_max_pixels(total_frames, native_fps):
    """Per-frame pixel cap: set by the 2 fps decode count, not by the 32 frames kept (235,200 for any long video)"""
    num_decoded = qwen_decode_count(total_frames, native_fps)
    max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / num_decoded * FRAME_FACTOR), int(LMMS_MIN_PIXELS * 1.05))

    return min(LMMS_MAX_PIXELS, max_pixels)


def smart_resize(height, width, min_pixels, max_pixels, factor=IMAGE_FACTOR):
    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)

    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor

    return resized_height, resized_width


def resize_like_qwen(frames, max_pixels):
    """uint8 frames (num_frames, height, width, 3) -> uint8 frames at qwen-vl-utils' size (torchvision resize, bicubic, antialias)"""
    _, height, width, _ = frames.shape
    resized_height, resized_width = smart_resize(height, width, LMMS_MIN_PIXELS, max_pixels)

    pixels = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # num_frames, 3, height, width

    resized = F.interpolate(pixels, size=(resized_height, resized_width), mode='bicubic', antialias=True, align_corners=False)

    return (
        resized
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .numpy()  # num_frames, resized_height, resized_width, 3
    )
