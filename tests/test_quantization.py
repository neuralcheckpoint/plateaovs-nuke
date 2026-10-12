"""Weight-only int8 error bound, including an all-zero channel and optional bias."""
import pytest
import torch

from plateaovs.model import _int8_linears


def _quantized_copy(build, reference, source_dtype=torch.float32):
    with _int8_linears(source_dtype):
        module = build()
    module.load_state_dict(reference.state_dict())
    return module


@pytest.mark.parametrize("bias", [False, True])
def test_int8_linear_preserves_output_within_channel_rounding_bound(bias):
    generator = torch.Generator().manual_seed(7)
    linear = torch.nn.Linear(16, 4, bias=bias)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(4, 16, generator=generator))
        linear.weight[0].zero_()
        linear.weight[1].mul_(0.001)
        if bias:
            linear.bias.copy_(torch.tensor([0.2, -0.1, 0.3, -0.4]))
    x = torch.randn(2, 3, 16, generator=generator)
    expected = linear(x).detach()
    scale = (linear.weight.detach().abs().amax(dim=1) / 127).clamp_min(1e-12)
    bound = x.abs().sum(dim=-1, keepdim=True) * scale / 2 + 1e-5
    module = _quantized_copy(lambda: torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(16, 4, bias=bias))),
                             torch.nn.Sequential(torch.nn.Sequential(linear)))
    actual = module(x)
    assert torch.isfinite(actual).all()
    assert ((actual - expected).abs() <= bound).all()
    torch.testing.assert_close(actual[..., 0], expected[..., 0])


def test_load_time_int8_matches_quantizing_the_cast_model_after_cosmos_init_and_cast():
    generator = torch.Generator().manual_seed(11)
    reference = torch.nn.Linear(32, 8)
    with torch.no_grad():
        reference.weight.copy_(torch.randn(8, 32, generator=generator))
        reference.bias.copy_(torch.randn(8, generator=generator))
    with _int8_linears(torch.bfloat16):
        module = torch.nn.Linear(32, 8)
    # What Cosmos does before loading: init the weights, cast the whole model.
    torch.nn.init.normal_(module.weight, std=0.02)
    torch.nn.init.constant_(module.weight, 0)
    module.to(dtype=torch.bfloat16)
    module.load_state_dict(reference.state_dict())
    weight = reference.weight.detach().to(torch.bfloat16).float()
    scale = (weight.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12)
    assert module.weight.dtype == torch.int8 and module.scale.dtype == torch.float32
    torch.testing.assert_close(module.weight, (weight / scale).round().clamp(-127, 127).to(torch.int8), rtol=0, atol=0)
    torch.testing.assert_close(module.scale, scale, rtol=0, atol=0)
    torch.testing.assert_close(module.bias, reference.bias.detach().to(torch.bfloat16), rtol=0, atol=0)
    assert torch.nn.Linear is not type(module)  # restored after the context
    with _int8_linears(torch.bfloat16):
        unloaded = torch.nn.Linear(32, 8)
    with pytest.raises(RuntimeError, match="Missing key"):
        unloaded.load_state_dict({"bias": reference.bias.detach()})


@pytest.mark.parametrize("layout", ["sbhd", "bshd"])
def test_rotary_preserves_both_halves_and_unrotated_tail(layout):
    from plateaovs.model import _apply_rotary_pos_emb

    generator = torch.Generator().manual_seed(17)
    x = torch.randn(5, 2, 3, 10, generator=generator).to(torch.bfloat16)
    if layout == "bshd":
        x = x.transpose(0, 1)
    freqs = torch.randn(5, 1, 1, 8, generator=generator)
    angles = freqs if layout == "sbhd" else freqs.transpose(0, 1)
    rotated = x[..., :8].float()
    first, second = rotated.chunk(2, -1)
    expected = rotated * angles.cos() + torch.cat((-second, first), -1) * angles.sin()
    expected = torch.cat((expected.to(x.dtype), x[..., 8:]), -1)
    torch.testing.assert_close(_apply_rotary_pos_emb(x, freqs, tensor_format=layout), expected, rtol=0, atol=0)


@pytest.mark.parametrize("encode", [True, False])
def test_spatial_tiles_blend_without_a_seam_for_a_local_filter(encode):
    from plateaovs.model import _tiled_tokenizer
    import torch.nn.functional as F

    generator = torch.Generator().manual_seed(23)
    shape = (1, 2, 3, 272, 384) if encode else (1, 2, 3, 34, 48)
    value = torch.randn(*shape, generator=generator)

    def local_filter(x):
        if not encode:
            x = F.interpolate(x, scale_factor=(1, 8, 8), mode="nearest")
        x = F.avg_pool3d(x.sin(), kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        return x[..., ::8, ::8] if encode else x

    expected = local_filter(value)
    actual = _tiled_tokenizer(local_filter, value, encode)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_streamed_blocks_match_resident_blocks_and_stay_in_pinned_ram():
    import copy
    from plateaovs.model import _BlockStreamer

    generator = torch.Generator().manual_seed(29)
    # Odd count and unequal sizes: the prefetch wraps around into either slot.
    def build():
        return torch.nn.ModuleDict({"block{}".format(i): torch.nn.Sequential(
            torch.nn.Linear(64, 32 * (i + 1)), torch.nn.LayerNorm(32 * (i + 1)), torch.nn.Linear(32 * (i + 1), 64))
            for i in range(5)})
    torch.manual_seed(29)
    # int8 buffers beside LayerNorm parameters, as in the DiT
    blocks = _quantized_copy(build, build())
    x = torch.randn(256, 64, generator=generator).cuda()

    def run(modules):
        out = x
        for block in modules.values():
            out = block(out)
        return out

    with torch.no_grad():
        expected = run(copy.deepcopy(blocks).cuda())
        streamer = _BlockStreamer(blocks.values())
        for _ in range(2):  # the second forward starts from the wrapped-around prefetch
            torch.testing.assert_close(run(blocks), expected, rtol=0, atol=0)
            assert all(t.device.type == "cpu" and t.is_pinned() for t in blocks.state_dict().values())
        streamer.release()
        torch.testing.assert_close(run(blocks), expected, rtol=0, atol=0)