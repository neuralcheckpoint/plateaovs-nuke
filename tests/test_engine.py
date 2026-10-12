"""Engine tests on the CPU with a stand-in for the 7B model."""

import os

import numpy as np
import OpenImageIO as oiio
import pytest
import torch

from aitools_engine import imageio
from plateaovs import pipeline


@pytest.mark.parametrize("count,window,overlap", [(1, 57, 8), (40, 57, 8), (57, 57, 8), (58, 57, 8),
                                                  (130, 57, 8), (100, 17, 4), (10, 1, 0)])
def test_windows_cover_every_frame_once_weighted_to_one(count, window, overlap):
    spans = pipeline.windows(count, window, overlap)
    total = np.zeros(count)
    for (start, length), fade in zip(spans, pipeline.weights(spans)):
        assert 0 <= start and start + length <= count and length <= window
        total[start:start + length] += fade
    # Every frame gets weight from somewhere, and no overlap is wider than asked except the last.
    assert (total > 0).all()
    for previous, current in zip(spans[:-2], spans[1:-1]):
        assert previous[0] + previous[1] - current[0] == overlap


def test_a_short_shot_is_padded_to_a_full_window_and_one_frame_stays_alone():
    assert [pipeline.model_length(n) for n in (1, 2, 40, 57)] == [1, 57, 57, 57]


class _Echo(object):
    """Hands each frame back as every pass, so outputs show which input frame they came from."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def infer(self, frames, name):
        assert frames.shape[0] in (1, 57)
        self.calls.append((frames.shape[0], name))
        return frames.clone()


def _plate(folder, count, width=64, height=36):
    pattern = os.path.join(folder, "plate.####.exr")
    for index in range(count):
        frame = 1001 + index
        spec = oiio.ImageSpec(width, height, 3, oiio.FLOAT)
        image = oiio.ImageBuf(spec)
        oiio.ImageBufAlgo.fill(image, [(index + 1) / (count + 1)] * 3)
        assert image.write(imageio.frame_path(pattern, frame))
    return pattern


def _read(path, channel):
    image = oiio.ImageBuf(path)
    names = list(image.spec().channelnames)
    return image.get_pixels(oiio.FLOAT)[..., names.index(channel)]


def test_frames_land_on_their_own_frame_across_windows(tmp_path):
    count = 70
    plate = _plate(str(tmp_path), count)
    output = str(tmp_path / "cache" / "aovs.####.exr")
    echo = _Echo()
    written = pipeline.run(plate, 1001, 1000 + count, output, passes=["roughness", "normal"],
                           transfer="display", renderer=echo)
    assert written == count and len(echo.calls) == 4, echo.calls
    for index in range(count):
        path = imageio.frame_path(output, 1001 + index)
        # The echo returns the input; cross-faded windows of the same frame still give that frame.
        np.testing.assert_allclose(_read(path, "roughness.R"), (index + 1) / (count + 1), atol=2e-3)
        normal = np.stack([_read(path, "normal." + c) for c in "RGB"])
        np.testing.assert_allclose(np.linalg.norm(normal, axis=0), 1.0, atol=2e-3)


def test_same_question_twice_runs_nothing_and_a_missing_frame_is_redone(tmp_path):
    count = 20
    plate = _plate(str(tmp_path), count)
    output = str(tmp_path / "cache" / "aovs.####.exr")
    kwargs = dict(passes=["roughness"], transfer="display")
    pipeline.run(plate, 1001, 1000 + count, output, renderer=_Echo(), **kwargs)
    echo = _Echo()
    assert pipeline.run(plate, 1001, 1000 + count, output, renderer=echo, **kwargs) == 0
    assert echo.calls == []
    os.remove(imageio.frame_path(output, 1015))
    assert pipeline.run(plate, 1001, 1000 + count, output, renderer=echo, **kwargs) == 1
    np.testing.assert_allclose(_read(imageio.frame_path(output, 1015), "roughness.R"), 15 / (count + 1), atol=2e-3)
    # Other passes are another cache: the old one goes, frames outside the range too.
    kwargs["passes"] = ["roughness", "metallic"]
    assert pipeline.run(plate, 1001, 1005, output, renderer=_Echo(), **kwargs) == 5
    assert not os.path.exists(imageio.frame_path(output, 1006))


def test_a_one_frame_look_does_not_cost_the_shot(tmp_path):
    count = 70
    plate = _plate(str(tmp_path), count)
    output = str(tmp_path / "cache" / "aovs.####.exr")
    kwargs = dict(passes=["roughness"], transfer="display")
    assert pipeline.run(plate, 1001, 1000 + count, output, renderer=_Echo(), **kwargs) == count
    # The frame is already there, made in its window: kept.
    echo = _Echo()
    assert pipeline.run(plate, 1006, 1006, output, renderer=echo, **kwargs) == 0 and echo.calls == []
    # Made alone, it is a look only: the shot redoes that frame and no other window.
    os.remove(imageio.frame_path(output, 1066))
    assert pipeline.run(plate, 1066, 1066, output, renderer=echo, **kwargs) == 1
    assert pipeline.run(plate, 1001, 1000 + count, output, renderer=echo, **kwargs) == 1
    assert echo.calls == [(1, "roughness"), (57, "roughness")]
    assert pipeline.run(plate, 1001, 1000 + count, output, renderer=echo, **kwargs) == 0
    # A shorter range touches only its frames.
    assert pipeline.run(plate, 1001, 1010, output, renderer=_Echo(), force=True, **kwargs) == 10
    assert pipeline.run(plate, 1001, 1000 + count, output, renderer=echo, **kwargs) == 0


def test_a_changed_input_frame_is_redone(tmp_path):
    plate = _plate(str(tmp_path), 20)
    output = str(tmp_path / "cache" / "aovs.####.exr")
    kwargs = dict(passes=["roughness"], transfer="display")
    pipeline.run(plate, 1001, 1020, output, renderer=_Echo(), **kwargs)
    path = imageio.frame_path(plate, 1007)
    os.utime(path, ns=(0, 0))
    assert pipeline.run(plate, 1001, 1020, output, renderer=_Echo(), **kwargs) == 1


class _PerWindow(_Echo):
    """Every frame of a window gets one value, that of the window's first frame: windows disagree."""

    def infer(self, frames, name):
        super().infer(frames, name)
        return torch.full_like(frames, float(frames[0].mean()))


def test_growing_the_range_fades_in_like_a_single_run(tmp_path):
    count = 72
    plate = _plate(str(tmp_path), count)
    kwargs = dict(passes=["metallic"], transfer="display")
    fresh = str(tmp_path / "fresh" / "aovs.####.exr")
    grown = str(tmp_path / "grown" / "aovs.####.exr")
    pipeline.run(plate, 1001, 1000 + count, fresh, renderer=_PerWindow(), **kwargs)
    pipeline.run(plate, 1001, 1057, grown, renderer=_PerWindow(), **kwargs)
    echo = _PerWindow()
    assert pipeline.run(plate, 1001, 1000 + count, grown, renderer=echo, **kwargs) == 15
    assert len(echo.calls) == 1, "only the new window runs"
    value = lambda pattern, frame: float(_read(imageio.frame_path(pattern, frame), "metallic.R").mean())
    for frame in range(1001, 1001 + count):
        assert abs(value(grown, frame) - value(fresh, frame)) < 2e-3, frame
    # And the fresh run has no jump where the second window starts or the first ends.
    steps = [abs(value(fresh, f + 1) - value(fresh, f)) for f in range(1001, 1000 + count)]
    assert max(steps) < 2 * (value(fresh, 1072) - value(fresh, 1001)) / 42, max(steps)


def test_redoing_one_frame_leaves_the_blended_overlap_as_it_was(tmp_path):
    count = 70
    plate = _plate(str(tmp_path), count)
    output = str(tmp_path / "cache" / "aovs.####.exr")
    kwargs = dict(passes=["metallic"], transfer="display")
    pipeline.run(plate, 1001, 1000 + count, output, renderer=_PerWindow(), **kwargs)
    value = lambda frame: float(_read(imageio.frame_path(output, frame), "metallic.R").mean())
    before = {frame: value(frame) for frame in range(1001, 1001 + count)}
    os.remove(imageio.frame_path(output, 1066))
    echo = _PerWindow()
    assert pipeline.run(plate, 1001, 1000 + count, output, renderer=echo, **kwargs) == 1
    assert len(echo.calls) == 1
    for frame, old in before.items():
        assert abs(value(frame) - old) < 1e-3, (frame, value(frame), old)


def test_albedo_is_linearised_from_srgb():
    value = torch.full((3, 2, 2), 0.5)
    albedo = pipeline.channels("basecolor", value)["albedo.R"]
    assert abs(float(albedo[0, 0]) - 0.214) < 1e-3


def test_one_frame_alone_goes_in_as_an_image(tmp_path):
    plate = _plate(str(tmp_path), 3)
    echo = _Echo()
    assert pipeline.run(plate, 1002, 1002, str(tmp_path / "aovs.####.exr"), passes=["depth"],
                        transfer="display", renderer=echo) == 1
    assert echo.calls == [(1, "depth")]
    assert abs(float(_read(str(tmp_path / "aovs.1002.exr"), "depth_relative.R").mean()) - 0.5) < 2e-3


@pytest.mark.parametrize("window", [0, 17, 56])
def test_windows_are_one_frame_or_the_trained_57(tmp_path, window):
    with pytest.raises(ValueError):
        pipeline.run("plate.####.exr", 1, 10, str(tmp_path / "aovs.####.exr"), window=window, renderer=_Echo())


@pytest.mark.parametrize("before,after", [
    ({"precision": "bf16"}, {"precision": "int8"}),
    ({"precision": "int8"}, {"precision": "bf16"}),
    ({"low_vram": False}, {"low_vram": True}),
    ({"low_vram": True}, {"low_vram": False}),
])
def test_render_mode_change_replaces_cache_without_mixed_frames(tmp_path, before, after):
    plate = _plate(str(tmp_path), 3)
    output = str(tmp_path / "cache" / "aovs.####.exr")
    kwargs = dict(passes=["roughness"], transfer="display")
    pipeline.run(plate, 1001, 1003, output, renderer=_Echo(), **before, **kwargs)
    pipeline.run(plate, 1001, 1001, output, renderer=_Echo(), **after, **kwargs)
    assert not os.path.exists(imageio.frame_path(output, 1002))
    assert not os.path.exists(imageio.frame_path(output, 1003))
    assert abs(float(_read(imageio.frame_path(output, 1001), "roughness.R").mean()) - 0.25) < 2e-3
