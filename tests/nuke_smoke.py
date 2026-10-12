"""End-to-end check inside Nuke, on the GPU (~15 min on a 3090).

Run from the project root:
    Nuke17.1 -i --nukex -t tests/nuke_smoke.py <scratch dir>

Computes PlateAOVs on 12 frames of a Tears of Steel plate (CC BY; set
PLATEAOVS_TEST_PLATE to use another) through the node, then one frame
through the render path. The plate has two people on a bridge over a
canal, trees and sky behind.
"""

import faulthandler
import os
import sys

import nuke

faulthandler.enable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nuke_plugin"))
import plateaovs_nuke  # noqa: E402

# EXR sequence (frames 1001-1012) with a person facing the camera standing on a floor.
PLATE = os.environ["PLATEAOVS_TEST_PLATE"]
FIRST, LAST = 1001, 1012
# Nuke pixel coordinates (y up) in the 1920x1012 plate.
PERSON = (720, 412)
GROUND = (960, 15)
scratch = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "scratch", "nuke_smoke"))
os.makedirs(os.path.join(scratch, "cache"), exist_ok=True)

nuke.root()["first_frame"].setValue(FIRST)
nuke.root()["last_frame"].setValue(LAST)
plateaovs_nuke.add_layers()

read = nuke.nodes.Read(file=PLATE, first=FIRST, last=LAST, origfirst=FIRST, origlast=LAST)
nuke.root()["format"].setValue(read.format())
for node in nuke.allNodes():
    node.setSelected(False)
read.setSelected(True)
group = plateaovs_nuke.create()
assert plateaovs_nuke.is_plateaovs(group)
assert group.input(0).name() == read.name()
assert (group["range_first"].value(), group["range_last"].value()) == (FIRST, LAST)
assert plateaovs_nuke.selected_passes(group) == ["basecolor", "normal", "roughness", "metallic"]
group["cache_pattern"].setValue(os.path.join(scratch, "cache") + "/")


def sample(node, channel, point, frame):
    # Node.sample's frame argument is ignored in a terminal session; it
    # reads the current frame.
    nuke.frame(frame)
    return node.sample(channel, point[0], point[1])


def vector(node, layer, point, frame):
    return [sample(node, "{}.{}".format(layer, c), point, frame) for c in ("red", "green", "blue")]


job = plateaovs_nuke.compute(group, force=True, wait=True)
print("compute:", group["status"].value())
assert job.error is None, (job.error, "\n".join(job.log[-20:]))
assert job.result["written"] == LAST - FIRST + 1

channels = set(group.channels())
for name in ("albedo.red", "albedo.green", "albedo.blue", "normal.red", "normal.green", "normal.blue",
             "roughness.red", "metallic.red"):
    assert name in channels, name
assert "depth_relative.red" not in channels, "depth was not asked for"

albedo = vector(group, "albedo", PERSON, 1006)
person = vector(group, "normal", PERSON, 1006)
ground = vector(group, "normal", GROUND, 1006)
roughness = sample(group, "roughness.red", PERSON, 1006)
print("albedo {}, normal person {}, ground {}, roughness {:.2f}".format(
    [round(v, 3) for v in albedo], [round(v, 2) for v in person], [round(v, 2) for v in ground], roughness))
assert all(0.0 <= v <= 1.0 for v in albedo) and sum(albedo) > 0.0, albedo
assert abs(sum(v * v for v in person) - 1.0) < 0.02 and person[2] > 0, "faces the camera: {}".format(person)
assert ground[1] > 0.5, "the bridge deck faces up: {}".format(ground)
assert 0.0 <= roughness <= 1.0, roughness
# The plate passes through untouched, and outside the range the passes are black.
assert sample(group, "rgba.red", PERSON, 1006) == sample(read, "rgba.red", PERSON, 1006)
assert sample(group, "albedo.red", PERSON, LAST + 1) == 0.0

# show puts a pass in rgb; a one-channel pass shows grey.
group["show"].setValue("albedo")
assert abs(sample(group, "rgba.green", PERSON, 1006) - albedo[1]) < 1e-6
group["show"].setValue("roughness")
assert sample(group, "rgba.red", PERSON, 1006) == sample(group, "rgba.blue", PERSON, 1006) == roughness
group["show"].setValue("Plate")

# Same parameters: Compute has nothing left to do.
job = plateaovs_nuke.compute(group, wait=True)
assert job.error is None and job.result["written"] == 0, job.result

# "this frame" on a computed frame: it is kept, nothing runs.
nuke.frame(1006)
group["range_current"].execute()
assert (group["range_first"].value(), group["range_last"].value()) == (1006, 1006)
job = plateaovs_nuke.compute(group, wait=True)
assert job.error is None and job.result["written"] == 0, job.result
# On a missing frame it makes just that one, and the shot's other frames stay.
cache = os.path.join(scratch, "cache")
missing = [os.path.join(cache, name) for name in os.listdir(cache) if name.endswith(".1009.exr")]
os.remove(missing[0])
group["range_first"].setValue(1009)
group["range_last"].setValue(1009)
job = plateaovs_nuke.compute(group, wait=True)
assert job.error is None and job.result["written"] == 1, job.result
assert len([name for name in os.listdir(cache) if name.endswith(".exr")]) == LAST - FIRST + 1
assert vector(group, "albedo", PERSON, 1006) == albedo, "a one-frame look left the shot alone"

# The render path: a node between the Read and the node forces a render; one
# frame, from a new source and with fewer passes, so a new cache.
grade = nuke.nodes.Grade(inputs=[read], white=1.2)
group.setInput(0, grade)
assert plateaovs_nuke.aitools_nuke.direct_read(group) is None
group["range_first"].setValue(1006)
group["range_last"].setValue(1006)
group["pass_roughness"].setValue(False)
group["pass_metallic"].setValue(False)
job = plateaovs_nuke.compute(group, wait=True)
assert job.error is None, (job.error, "\n".join(job.log[-20:]))
assert job.result["written"] == 1
assert not os.path.exists(os.path.join(scratch, "cache", "source"))
assert "roughness.red" not in set(group.channels()), "an unticked pass is not copied in"
single = vector(group, "normal", GROUND, 1006)
print("one frame alone: ground normal {} ({})".format([round(v, 2) for v in single], group["status"].value()))
assert single[1] > 0.5, single
print("NUKE SMOKE OK")
