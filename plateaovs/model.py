"""Cosmos DiffusionRenderer's inverse renderer: plate frames in, one G-buffer pass out.

The 7B DiT is loaded once (bf16, ~14 GB of VRAM) and kept for every pass and
window; the official offload path reloads the 29 GB checkpoint from disk per
pass. A window is encoded once for all its passes. For a video window the
tokenizer doesn't fit beside the DiT on a 24 GB card, so the DiT waits in RAM
while the tokenizer encodes or decodes (a few seconds per move). Transformer
Engine is replaced by the PyTorch equivalent of the three pieces the DiT
uses (RMSNorm, rotary embedding, attention), so the engine installs from
wheels alone.
Optional int8 stores Linear weights per channel, dequantizing one active
layer at a time for bf16 matmul; the tokenizer and activations stay bf16.
Each Linear is quantized as the checkpoint loads, so the bf16 DiT never sits in RAM.
Low VRAM adds spatial tokenizer tiles (approximate) and per-block CPU offload.
"""

import os
import sys
import types
from contextlib import contextmanager, nullcontext
from functools import partial

# Read at the first CUDA allocation: lets the tokenizer's big blocks reuse
# the room the DiT leaves.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor", "cosmos1-diffusion-renderer")
# The video tokenizer encodes 57 frames at a time (or one image), nothing else.
FRAMES = 57
CHECKPOINTS = os.path.join(ROOT, "models", "checkpoints")
CHECKPOINT = "Diffusion_Renderer_Inverse_Cosmos_7B"

# The resolution the model was trained at; frames are stretched to it, as the
# official loader does, and the passes stretched back.
HEIGHT, WIDTH = 704, 1280
# The model's context index of each pass (GBUFFER_INDEX_MAPPING upstream).
PASSES = {"basecolor": 0, "metallic": 1, "roughness": 2, "normal": 3, "depth": 4}


class _RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, **_):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class _DotProductAttention(torch.nn.Module):
    def __init__(self, num_attention_heads, kv_channels, qkv_format="sbhd", **_):
        super().__init__()
        self.qkv_format = qkv_format

    def forward(self, q, k, v, **_):
        # sbhd or bshd in, b h s d for SDPA, back out with the heads merged.
        order = (1, 2, 0, 3) if self.qkv_format == "sbhd" else (0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q.permute(order), k.permute(order), v.permute(order))
        out = out.permute(2, 0, 1, 3) if self.qkv_format == "sbhd" else out.permute(0, 2, 1, 3)
        return out.reshape(out.shape[0], out.shape[1], -1)


def _apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", **_):
    """Transformer Engine's non-interleaved rotary embedding; freqs is [s, 1, 1, d]."""
    freqs = freqs[: t.shape[0]] if tensor_format == "sbhd" else freqs[: t.shape[1]].transpose(0, 1)
    d = freqs.shape[-1]
    rotated, rest = t[..., :d].float(), t[..., d:]
    first, second = rotated.chunk(2, dim=-1)
    # Rotate halves separately: avoid two full-size fp32 temporaries on long videos.
    sine = freqs.sin()
    out = rotated * freqs.cos()
    out[..., :d // 2].sub_(second * sine[..., :d // 2])
    out[..., d // 2:].add_(first * sine[..., d // 2:])
    out = out.to(t.dtype)
    return out if d == t.shape[-1] else torch.cat((out, rest), dim=-1)


def _install_transformer_engine_stand_in():
    # Megatron probes for the real Transformer Engine and falls back without
    # it; it has to finish that probe before the stand-in exists.
    import megatron.core  # noqa: F401

    te = types.ModuleType("transformer_engine")
    te.pytorch = types.ModuleType("transformer_engine.pytorch")
    te.pytorch.RMSNorm = _RMSNorm
    te.pytorch.attention = types.ModuleType("transformer_engine.pytorch.attention")
    te.pytorch.attention.DotProductAttention = _DotProductAttention
    te.pytorch.attention.apply_rotary_pos_emb = _apply_rotary_pos_emb
    sys.modules.setdefault("transformer_engine", te)
    sys.modules.setdefault("transformer_engine.pytorch", te.pytorch)
    sys.modules.setdefault("transformer_engine.pytorch.attention", te.pytorch.attention)


class _Int8Linear(torch.nn.Module):
    """Symmetric per-output-channel int8 weights; only the active layer is dequantized.

    Weight-only: matmul still runs in bf16. Saves VRAM, not an int8 speedup.
    Built in place of ``nn.Linear`` (see ``_int8_linears``): the weight is a
    meta placeholder until ``load_state_dict`` quantizes the checkpoint's.
    """
    source_dtype = torch.bfloat16

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.register_buffer("weight", torch.empty(out_features, in_features, device="meta"))
        self.register_buffer("scale", None)
        self.register_buffer("bias", torch.empty(out_features, device=device, dtype=dtype) if bias else None)

    def reset_parameters(self):
        """Cosmos' skip_init_linear swaps this out; the checkpoint provides the weights."""

    def _apply(self, fn, recurse=True):
        # The model-wide dtype/device cast before loading skips the placeholder.
        if not self.weight.is_meta:
            return super()._apply(fn, recurse)
        placeholder = self._buffers.pop("weight")
        try:
            return super()._apply(fn, recurse)
        finally:
            self._buffers["weight"] = placeholder

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        if prefix + "weight" not in state_dict:
            missing_keys.append(prefix + "weight")
            return
        # Rounded to the model dtype first: the same numbers the bf16 DiT would hold.
        weight = state_dict[prefix + "weight"].to(self.source_dtype).float()
        scale = (weight.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12)
        self.weight = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        self.scale = scale
        if self.bias is not None:
            if prefix + "bias" in state_dict:
                self.bias.copy_(state_dict[prefix + "bias"])
            else:
                missing_keys.append(prefix + "bias")

    def forward(self, x):
        weight = self.weight.to(x.dtype).mul_(self.scale.to(x.dtype))
        return F.linear(x, weight, self.bias)


@contextmanager
def _int8_linears(source_dtype):
    """Every ``nn.Linear`` built inside is an unloaded ``_Int8Linear``."""
    linear = torch.nn.Linear
    torch.nn.Linear = type("_Int8Linear", (_Int8Linear,), {"source_dtype": source_dtype})
    try:
        yield
    finally:
        torch.nn.Linear = linear


@torch.no_grad()
def _tiled_tokenizer(function, value, encode):
    """2x2 spatial tiles with a 128-pixel halo; temporal context is never split.

    Blend on CPU so the full decoded video doesn't accumulate on the GPU.
    The tokenizer's spatial normalization changes slightly: this is opt-in.
    """
    halo = 128 if encode else 16
    ratio = 1 / 8 if encode else 8
    h, w = value.shape[-2:]
    ys = ((0, h // 2 + halo), (h // 2 - halo, h))
    xs = ((0, w // 2 + halo), (w // 2 - halo, w))
    output = total = None
    for iy, (y0, y1) in enumerate(ys):
        for ix, (x0, x1) in enumerate(xs):
            tile = function(value[..., y0:y1, x0:x1].contiguous()).to(device="cpu", dtype=torch.float32)
            if output is None:
                height, width = round(h * ratio), round(w * ratio)
                output = torch.zeros(*tile.shape[:-2], height, width)
                total = torch.zeros(height, width)
            th, tw = tile.shape[-2:]
            overlap = round(2 * halo * ratio)
            wy, wx = torch.ones(th), torch.ones(tw)
            ramp = torch.linspace(0, 1, overlap)
            if iy == 0:
                wy[-overlap:] = ramp.flip(0)
            else:
                wy[:overlap] = ramp
            if ix == 0:
                wx[-overlap:] = ramp.flip(0)
            else:
                wx[:overlap] = ramp
            weight = wy[:, None] * wx[None, :]
            y, x = round(y0 * ratio), round(x0 * ratio)
            output[..., y:y + th, x:x + tw] += tile.mul_(weight)
            total[y:y + th, x:x + tw] += weight
    return output.div_(total).to(value.device, value.dtype)


def _view(buffer, offset, like):
    """``like``'s dtype and shape over the bytes of ``buffer`` from ``offset``."""
    return buffer[offset:offset + like.nbytes].view(like.dtype).view(like.shape)


def _put(module, name, value):
    if name in module._parameters:
        module._parameters[name].data = value
    else:
        module._buffers[name] = value


class _BlockStreamer(object):
    """Low VRAM: DiT blocks live in pinned RAM and take turns in two GPU slots.

    While a block computes, the next one copies in on a side stream; CUDA
    events order the copies against the compute, so the CPU never waits.
    Weights are read-only: after its forward a block points back at its RAM
    copy, nothing is copied back.
    """

    def __init__(self, blocks):
        self.blocks = []
        for index, block in enumerate(blocks):
            tensors = [(module, name, value) for module in block.modules()
                       for name, value in list(module._parameters.items()) + list(module._buffers.items())
                       if value is not None]
            # One pinned buffer per block, tensors 512-byte aligned as the CUDA allocator does.
            offsets = [0]
            for _, _, value in tensors:
                offsets.append(offsets[-1] + -(-value.nbytes // 512) * 512)
            host = torch.empty(offsets[-1], dtype=torch.uint8, pin_memory=True)
            entries = []
            for (module, name, value), offset in zip(tensors, offsets):
                view = _view(host, offset, value)
                view.copy_(value)
                _put(module, name, view)
                entries.append((module, name, offset, view))
            self.blocks.append((host, entries))
            block.register_forward_pre_hook(partial(self._enter, index))
            block.register_forward_hook(partial(self._exit, index), always_call=True)
        self.size = max(host.numel() for host, _ in self.blocks)
        self.stream = torch.cuda.Stream()
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.free = [torch.cuda.Event(), torch.cuda.Event()]
        self.slots = None
        self.loaded = [None, None]

    def _fetch(self, index, slot):
        host = self.blocks[index][0]
        self.stream.wait_event(self.free[slot])
        with torch.cuda.stream(self.stream):
            self.slots[slot][:host.numel()].copy_(host, non_blocking=True)
        self.ready[slot].record(self.stream)
        self.loaded[slot] = index

    def _enter(self, index, _block, _args):
        if self.slots is None:
            self.slots = [torch.empty(self.size, dtype=torch.uint8, device="cuda") for _ in range(2)]
            for event in self.free:
                event.record()  # copies into the new memory wait for its previous users
        if index not in self.loaded:
            self._fetch(index, self.loaded.index(None) if None in self.loaded else 0)
        slot = self.loaded.index(index)
        torch.cuda.current_stream().wait_event(self.ready[slot])
        for module, name, offset, view in self.blocks[index][1]:
            _put(module, name, _view(self.slots[slot], offset, view))
        # Wraps around: the next forward's first block copies in during the sampler step.
        following = (index + 1) % len(self.blocks)
        if following not in self.loaded:
            self._fetch(following, 1 - slot)

    def _exit(self, index, _block, _args, _output):
        for module, name, _, view in self.blocks[index][1]:
            _put(module, name, view)
        if index in self.loaded:
            self.free[self.loaded.index(index)].record()

    def release(self):
        """Free the slots for the tokenizer; the next block forward takes them again."""
        torch.cuda.current_stream().wait_stream(self.stream)  # no copy lands in reused memory
        self.slots = None
        self.loaded = [None, None]


class InverseRenderer(object):
    def __init__(self, steps=8, seed=1000, precision="bf16", low_vram=False):
        if precision not in ("bf16", "int8"):
            raise ValueError("Precision must be bf16 or int8")
        self.precision = precision
        self.low_vram = low_vram
        if VENDOR not in sys.path:
            sys.path.insert(0, VENDOR)
        _install_transformer_engine_stand_in()
        # Not the official DiffusionRendererPipeline: its module imports the
        # prompt upsamplers and the autoregressive training code.
        from cosmos_predict1.diffusion.inference.inference_utils import (
            load_model_by_config, load_network_model, load_tokenizer_model)
        from cosmos_predict1.diffusion.model.model_diffusion_renderer import DiffusionRendererModel

        self.steps = steps
        self.seed = seed
        self._encoded = None
        # Cosmos finds its config files relative to the working directory.
        cwd = os.getcwd()
        os.chdir(VENDOR)
        try:
            self.model = load_model_by_config(
                config_job_name=CHECKPOINT,
                config_file="cosmos_predict1/diffusion/config/diffusion_renderer_config.py",
                model_class=DiffusionRendererModel,
            )
        finally:
            os.chdir(cwd)
        # Load/quantize in RAM before any CUDA transfer in the memory-saving modes.
        cpu_load = precision == "int8" or low_vram
        if cpu_load:
            self.model.tensor_kwargs["device"] = "cpu"
            self.model.cuda = lambda *args, **kwargs: self.model
        try:
            # mmap keeps the fp32 checkpoint in the page cache instead of copying it.
            load = torch.load
            torch.load = partial(load, mmap=True)
            linears = _int8_linears(self.model.tensor_kwargs["dtype"]) if precision == "int8" else nullcontext()
            try:
                with linears:
                    load_network_model(self.model, os.path.join(CHECKPOINTS, CHECKPOINT, "model.pt"))
            finally:
                torch.load = load
            # Cosmos only logs missing keys; an unloaded int8 weight would fail mid-render.
            if any(t.is_meta for t in self.model.model.state_dict().values()):
                raise RuntimeError("DiT checkpoint is missing Linear weights")
            self.model.tensor_kwargs["device"] = "cuda"
            load_tokenizer_model(self.model, os.path.join(CHECKPOINTS, "Cosmos-Tokenize1-CV8x8x8-720p"))
        finally:
            if cpu_load:
                del self.model.cuda
        if low_vram:
            self.model.tokenizer.to("cuda")
            self._move_dit("cuda")
            self._blocks = _BlockStreamer(self.model.net.blocks.values())
        elif cpu_load:
            self.model.cuda()

    def _move_dit(self, device):
        if not self.low_vram or device == "cpu":
            self.model.model.to(device)
            if self.low_vram:
                self._blocks.release()
        else:
            # The pinned Cosmos net has no direct parameters/buffers: only children.
            # Transformer blocks stay in RAM; the block streamer brings them in two at a time.
            for name, child in self.model.net.named_children():
                if name != "blocks":
                    child.to(device)
            self.model.conditioner.to(device)
            self.model.logvar.to(device)

    def _tokenizer(self, function, value, encode=False):
        """Run the tokenizer with the DiT parked in RAM; optionally tile spatially."""
        park = self.low_vram or value.shape[2] > 1
        if park:
            self._move_dit("cpu")
            torch.cuda.empty_cache()
        try:
            return _tiled_tokenizer(function, value, encode) if self.low_vram else function(value)
        finally:
            if park:
                self._move_dit("cuda")

    @torch.no_grad()
    def infer(self, frames, gbuffer):
        """``frames`` [T, 3, HEIGHT, WIDTH] display-referred 0-1 on the GPU, T = 1 or FRAMES.

        Returns the pass as float [T, 3, HEIGHT, WIDTH], 0-1 (normals as
        unit vectors in -1..1). The same ``frames`` tensor for every pass of
        a window reuses its encoding.
        """
        count = frames.shape[0]
        video = (frames * 2 - 1).permute(1, 0, 2, 3)[None].to(device="cuda", dtype=torch.bfloat16)
        if self._encoded is None or self._encoded[0] is not frames:
            self._encoded = None
            self._encoded = (frames, self._tokenizer(self.model.encode, video, encode=True).contiguous())
        latent = self._encoded[1]
        batch = {
            "video": video,
            "rgb": video,
            "context_index": torch.full((1, 1), PASSES[gbuffer], dtype=torch.long),
            "is_preprocessed": True,
            "num_frames": torch.tensor([float(count)]),
            "image_size": torch.tensor([[HEIGHT, WIDTH]]),
            "fps": torch.tensor([24.0]),
            "padding_mask": torch.zeros(1, 1, HEIGHT, WIDTH),
            "t5_text_embeddings": torch.zeros(1, 512, 1024),
            "t5_text_mask": torch.cat((torch.ones(1, 1), torch.zeros(1, 511)), dim=1),
        }
        # Floats in bf16 on the GPU, as the official misc.to(batch, "cuda", bfloat16).
        batch = {key: value if not torch.is_tensor(value) else
                 value.to("cuda", torch.bfloat16) if value.is_floating_point() else value.to("cuda")
                 for key, value in batch.items()}
        tokenizer = self.model.tokenizer
        state_shape = [
            tokenizer.channel, (count - 1) // 8 + 1,
            HEIGHT // tokenizer.spatial_compression_factor, WIDTH // tokenizer.spatial_compression_factor,
        ]
        # The sampler draws its noise from the global generator and ignores
        # its seed argument: seed it, so the same window gives the same pass.
        torch.manual_seed(self.seed)
        # The condition is the window's latent, already encoded.
        self.model.encode = lambda _state: latent
        try:
            sample = self.model.generate_samples_from_batch(
                batch, guidance=0.0, state_shape=state_shape, num_steps=self.steps,
                is_negative_prompt=False, seed=self.seed,
            )
        finally:
            del self.model.encode
        out = self._tokenizer(self.model.decode, sample)[0].float().permute(1, 0, 2, 3)  # [T, 3, H, W], -1..1
        if gbuffer == "normal":
            return out / out.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return ((out + 1) / 2).clamp(0, 1)
