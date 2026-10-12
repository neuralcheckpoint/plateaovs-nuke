"""Plate AOVs: albedo, normals, roughness, metallic and depth as an EXR sequence.

The plate goes to the inverse renderer in windows of 57 frames (the only
video length its tokenizer takes; one frame alone also works), consecutive windows
sharing ``overlap`` frames that are cross-faded, so the passes don't jump
where one window ends. Each window is stretched to 1280x704 (as the official
loader does) and every pass is stretched back to the plate's display window.
Frames are written as soon as no later window touches them.
"""

import json
import os
import time

import numpy as np
import OpenImageIO as oiio
import torch
import torch.nn.functional as F

from aitools_engine import imageio
from aitools_engine.events import emit as _emit, read_manifest as _read_manifest

from .model import FRAMES, HEIGHT, WIDTH, InverseRenderer

CACHE_VERSION = 3
MANIFEST_NAME = "plateaovs.json"
SOFTWARE = "PlateAOVs (Cosmos DiffusionRenderer)"
PASSES = ("basecolor", "normal", "roughness", "metallic", "depth")


def windows(count, window, overlap):
    """(start, length) of the windows covering ``count`` frames.

    The last window is moved back to end on the last frame rather than
    padded, so it overlaps its neighbour by more; a plate shorter than one
    window gets one window of its own length.
    """
    if count <= window:
        return [(0, count)]
    step = window - overlap
    starts = list(range(0, count - window + 1, step))
    if starts[-1] + window < count:
        starts.append(count - window)
    return [(start, window) for start in starts]


def weights(spans):
    """Per-window cross-fade weights: a ramp across each overlap, 1 elsewhere."""
    result = []
    for index, (start, length) in enumerate(spans):
        w = np.ones(length, np.float32)
        if index > 0:
            before = spans[index - 1][0] + spans[index - 1][1] - start
            w = np.minimum(w, np.arange(1, length + 1, dtype=np.float32) / (before + 1))
        if index + 1 < len(spans):
            after = start + length - spans[index + 1][0]
            w = np.minimum(w, np.arange(length, 0, -1, dtype=np.float32) / (after + 1))
        result.append(w)
    return result


def model_length(length):
    """The frame count the model takes for ``length`` frames: 1, or a full window."""
    return 1 if length == 1 else FRAMES


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


# Nuke layer of each pass. Depth gets its own layer: it is 0 (near) to 1 (far)
# per window, not the 1/distance Nuke's depth.Z holds.
LAYERS = {"basecolor": "albedo", "normal": "normal", "roughness": "roughness", "metallic": "metallic",
          "depth": "depth_relative"}


def channels(name, value):
    """EXR channels of one pass from its blended [3, H, W] value."""
    layer = LAYERS[name]
    if name == "basecolor":
        rgb = srgb_to_linear(value.clamp(0, 1))
        return {layer + ".R": rgb[0], layer + ".G": rgb[1], layer + ".B": rgb[2]}
    if name == "normal":
        n = value / value.norm(dim=0, keepdim=True).clamp(min=1e-6)
        return {layer + ".R": n[0], layer + ".G": n[1], layer + ".B": n[2]}
    return {layer + ".R": value.mean(dim=0)}


def _fade_into(values, path, share):
    """``share`` of ``values`` over the rest from the same channels in the EXR at ``path``."""
    image = oiio.ImageBuf(path)
    names = list(image.spec().channelnames)
    old = image.get_pixels(oiio.FLOAT)
    out = {key: share * value + (1 - share) * old[..., names.index(key)] for key, value in values.items()}
    if "normal.R" in out:
        normal = np.stack([out["normal." + c] for c in "RGB"])
        normal /= np.maximum(np.linalg.norm(normal, axis=0), 1e-6)
        out.update({"normal." + c: normal[i] for i, c in enumerate("RGB")})
    return out


def run(input_pattern, first, last, output_pattern, passes=PASSES, window=57, overlap=8, steps=8,
        transfer="auto", force=False, progress=None, renderer=None, precision="bf16", low_vram=False):
    """Write one EXR per frame with the requested passes; returns the frames written.

    The cache remembers, per frame, the input file it came from, whether
    it was made alone (a single-frame request) or inside video windows, and
    which windows it is the blend of.
    With the same parameters a request only redoes its frames that are
    missing, whose input changed, or that were made alone (unless this
    request is a single frame too); frames outside the range are never
    touched, so a one-frame look doesn't cost the shot. Other parameters
    (passes, steps, input...) make a new cache: every frame of the old one
    is deleted, so the cache never mixes frames with different passes.
    ``renderer`` replaces the model in tests: anything with
    ``infer(frames, pass) -> frames``.
    """
    if last < first:
        raise ValueError("Last frame {} is before first frame {}".format(last, first))
    if precision not in ("bf16", "int8"):
        raise ValueError("Precision must be bf16 or int8")
    unknown = [name for name in passes if name not in PASSES]
    if unknown or not passes:
        raise ValueError("Passes must be some of {}, got {}".format(", ".join(PASSES), passes))
    if window not in (1, FRAMES):
        raise ValueError("Window must be 1 or {} frames, got {}".format(FRAMES, window))
    if window > 1 and not 0 <= overlap < window:
        raise ValueError("Overlap must be between 0 and the window minus one")
    if not imageio.has_frame_token(output_pattern):
        raise ValueError("Output {!r} has no frame number; use a pattern such as "
                         "cache/aovs.####.exr".format(output_pattern))
    passes = [name for name in PASSES if name in passes]
    overlap = overlap if window > 1 else 0
    output_dir = os.path.dirname(os.path.abspath(output_pattern))
    os.makedirs(output_dir, exist_ok=True)

    frames = list(range(first, last + 1))
    params = {
        "version": CACHE_VERSION,
        "input": input_pattern,
        "passes": passes,
        "window": window,
        "overlap": overlap,
        "steps": steps,
        "transfer": transfer,
    }
    # Keep existing bf16 caches valid; int8 is a distinct render, never mixed.
    if precision != "bf16":
        params["precision"] = precision
    if low_vram:
        params["low_vram"] = True
    manifest = os.path.join(output_dir, MANIFEST_NAME)
    previous = _read_manifest(manifest)
    if previous.get("params") == params:
        made = previous.get("frames", {})
    else:
        # Another cache: delete all of it, in range or not.
        made = {}
        for frame in list(previous.get("frames", {})) + frames:
            path = imageio.frame_path(output_pattern, int(frame))
            if os.path.isfile(path):
                os.remove(path)
    alone = len(frames) == 1 or window == 1
    stamps = {stamp[0]: stamp[1:3] for stamp in imageio.sequence_stamp(input_pattern, frames, hashes=0)}

    def done(frame):
        record = made.get(str(frame))
        return (not force and record is not None and record[:2] == stamps[frame]
                and (alone or not record[2]) and os.path.isfile(imageio.frame_path(output_pattern, frame)))

    missing = [i for i, frame in enumerate(frames) if not done(frame)]
    for i in missing:
        made.pop(str(frames[i]), None)
    _emit(progress, "start", frames=len(frames), todo=len(missing))
    if not missing:
        _emit(progress, "done", written=0)
        return 0

    spans = windows(len(frames), window, overlap)
    fades = weights(spans)
    todo = set(missing)
    # Only windows touching a frame to redo run; the others fed frames on disk.
    runs = [any(start + i in todo for i in range(length)) for start, length in spans]
    cover = np.zeros(len(frames))
    for (start, length), fade in zip(spans, fades):
        cover[start:start + length] += fade

    reference = imageio.read_frame(imageio.frame_path(input_pattern, first))
    linear = reference.is_float if transfer == "auto" else transfer == "linear"
    _emit(progress, "setup", width=reference.width, height=reference.height, process_width=WIDTH,
          process_height=HEIGHT, windows=sum(runs), passes=passes)
    if renderer is None:
        _emit(progress, "status", percent=0, message="loading Cosmos DiffusionRenderer (7B)")
        renderer = InverseRenderer(steps=steps, precision=precision, low_vram=low_vram)
    device = getattr(renderer, "device", "cuda" if torch.cuda.is_available() else "cpu")

    def save():
        with open(manifest, "w") as handle:
            json.dump({"params": params, "frames": made}, handle, indent=2)

    save()

    def load(index):
        frame = imageio.read_frame(imageio.frame_path(input_pattern, frames[index]))
        rgb = torch.from_numpy(frame.rgb).to(device).permute(2, 0, 1)[None]
        rgb = F.interpolate(imageio.to_display(rgb, linear), size=(HEIGHT, WIDTH), mode="bilinear",
                            antialias=True)
        return rgb[0]

    # Running sums, on the CPU (the GPU is the model's), over frames some
    # window still adds to: index -> (pass -> weighted sum, total weight).
    pending = {}
    written = 0
    total_work = sum(runs) * len(passes)
    finished = 0
    started = time.time()
    for number, (start, length) in enumerate(spans):
        if runs[number]:
            fade = torch.from_numpy(fades[number])
            clip = torch.stack([load(start + i) for i in range(length)])
            padded = model_length(length)
            if padded > length:
                clip = torch.cat([clip, clip[-1:].expand(padded - length, -1, -1, -1)])
            if low_vram:
                clip = clip.cpu()
            for step, name in enumerate(passes):
                left = "" if not finished else ", ~{:.0f} min left".format(
                    (time.time() - started) / finished * (total_work - finished) / 60)
                _emit(progress, "status", percent=int(100 * finished / total_work),
                      message="window {}/{}: {}{}".format(finished // len(passes) + 1, sum(runs), name, left))
                out = renderer.infer(clip, name)[:length].float().cpu()
                finished += 1
                for i in range(length):
                    sums, weight = pending.setdefault(start + i, ({}, [0.0]))
                    sums[name] = sums.get(name, 0) + out[i] * fade[i]
                    if step == 0:
                        weight[0] += float(fade[i])
        # Frames no later window covers are final; every window covering a
        # frame to redo has run.
        limit = spans[number + 1][0] if number + 1 < len(spans) else len(frames)
        for index in sorted(k for k in pending if k < limit):
            sums, weight = pending.pop(index)
            frame = frames[index]
            windows_here = [[frames[s], frames[s + n - 1]] for s, n in spans if s <= index < s + n]
            if index not in todo and made[str(frame)][3] == windows_here:
                continue  # on disk already as the blend of these same windows
            share = weight[0] / cover[index]
            values = {}
            for name in passes:
                value = sums[name] / weight[0]
                value = F.interpolate(value[None], size=(reference.height, reference.width),
                                      mode="bilinear", antialias=True, align_corners=False)[0]
                values.update({key: array.numpy() for key, array in channels(name, value).items()})
            path = imageio.frame_path(output_pattern, frame)
            if index in todo:
                written += 1
            elif share < 1 - 1e-6:
                # Made with other windows (the range grew) and also covered
                # by one that didn't run: the frame on disk stands for that
                # one, so the new windows fade in over the overlap as in a
                # single run, instead of starting with a jump.
                values = _fade_into(values, path, share)
            made[str(frame)] = stamps[frame] + [alone, windows_here]
            imageio.write_channels(path, reference, values, SOFTWARE)
        save()
    _emit(progress, "done", written=written)
    return written
