"""PlateAOVs node for Nuke.

A Group that passes its input through and copies in the material passes of
the plate read from a per-node EXR cache: ``albedo`` (linear, light
removed), camera-space ``normal``, ``roughness``, ``metallic`` and, if asked,
``depth_relative``. Pressing Compute runs NVIDIA's Cosmos DiffusionRenderer
inverse renderer in its own GPU Python (``.venv`` in the project root),
outside Nuke's interpreter, and reloads the cache when it finishes. The
*show* knob puts one pass in rgb for a quick look.

The engine subprocess, progress, source render and cache Read are the
shared ones in aitools_nuke (same folder).
"""

import json
import os
import sys

import nuke

# realpath: the plugin may be reached through a symlink.
PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(PLUGIN_DIR)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)
import aitools_nuke  # noqa: E402

TITLE = "PlateAOVs"
NODE_VERSION = 3
SOURCES = ["Auto", "Render input"]
PRECISIONS = ["bf16", "int8"]
# (engine pass, Nuke layer, knob, label, on by default)
PASSES = [
    ("basecolor", "albedo", "pass_albedo", "albedo", True),
    ("normal", "normal", "pass_normal", "normals", True),
    ("roughness", "roughness", "pass_roughness", "roughness", True),
    ("metallic", "metallic", "pass_metallic", "metallic", True),
    ("depth", "depth_relative", "pass_depth", "depth", False),
]
SHOW = ["Plate"] + [label for _, _, _, label, _ in PASSES]
CHANNELS = {"albedo": "rgb", "normal": "rgb"}


def add_layers():
    """Declare the layers so Copy nodes, Shuffles and the viewer can address them."""
    for _, layer, _, _, _ in PASSES:
        names = ("red", "green", "blue") if CHANNELS.get(layer) else ("red",)
        nuke.tcl("add_layer {{{} {}}}".format(layer, " ".join("{}.{}".format(layer, c) for c in names)))


# ----------------------------------------------------------------- creation


def create():
    """Create a PlateAOVs node after the selected node."""
    add_layers()
    selected = nuke.selectedNodes()
    group = nuke.createNode("Group", inpanel=False)
    index = 1
    while nuke.toNode("PlateAOVs{}".format(index)) is not None:
        index += 1
    group.setName("PlateAOVs{}".format(index))
    group["tile_color"].setValue(0x4F9F8FFF)
    # Knobs first: the internal Read's expressions refer to them.
    _add_knobs(group)
    _build_internals(group)
    if len(selected) == 1:
        group.setInput(0, selected[0])
    source = group.input(0)
    if source is not None:
        group["range_first"].setValue(int(source.firstFrame()))
        group["range_last"].setValue(int(source.lastFrame()))
    else:
        group["range_first"].setValue(int(nuke.root().firstFrame()))
        group["range_last"].setValue(int(nuke.root().lastFrame()))
    group.showControlPanel()
    return group


def _build_internals(group):
    with group:
        source = nuke.nodes.Input(name="Source")
        cache = aitools_nuke.add_cache_read()
        # Show every cached frame, not just the last range computed: after a
        # one-frame look the rest of the shot stays visible.
        for knob in ("first", "origfirst"):
            cache[knob].setExpression("min(parent.range_first, root.first_frame)")
        for knob in ("last", "origlast"):
            cache[knob].setExpression("max(parent.range_last, root.last_frame)")
        stream = source
        for _, layer, knob, _, _ in PASSES:
            copy = nuke.nodes.Copy(name="Copy_" + layer, inputs=[stream, cache])
            copy["channels"].setValue(layer)
            # A pass not computed would come in as a black layer.
            copy["disable"].setExpression("!parent.{}".format(knob))
            stream = copy
        views = [stream]
        for _, layer, _, _, _ in PASSES:
            view = nuke.nodes.Shuffle(name="Show_" + layer, inputs=[stream])
            view["in"].setValue(layer)
            if not CHANNELS.get(layer):
                # One channel: grey in the viewer.
                for channel in ("green", "blue"):
                    view[channel].setValue("red")
            view["alpha"].setValue("white")
            views.append(view)
        switch = nuke.nodes.Switch(name="Show", inputs=views)
        switch["which"].setExpression("parent.show")
        nuke.nodes.Output(inputs=[switch])


def _add_knobs(group):
    group.addKnob(nuke.Tab_Knob("plateaovs", "PlateAOVs"))
    group.addKnob(nuke.Text_Knob("passes_label", "passes", ""))
    for engine_pass, layer, knob, label, default in PASSES:
        box = nuke.Boolean_Knob(knob, label)
        box.setValue(default)
        if engine_pass != "basecolor":
            box.clearFlag(nuke.STARTLINE)
        group.addKnob(box)
    group["pass_albedo"].setTooltip("albedo: the surface colour with the lighting taken out, linear.")
    group["pass_normal"].setTooltip("normal: camera-space unit normals, x right, y up, z towards the camera.")
    group["pass_depth"].setTooltip(
        "depth_relative: 0 near to 1 far, rescaled per 57-frame window, so it can drift across a long shot. "
        "A dedicated video depth tool gives steadier depth or metres.")
    show = nuke.Enumeration_Knob("show", "show", SHOW)
    show.setTooltip("Plate: the input with the passes added as layers. A pass: that pass in rgb, for a look.")
    group.addKnob(show)
    source = nuke.Enumeration_Knob("source_mode", "source", SOURCES)
    source.setTooltip(
        "Auto reads the files of a Read connected straight to the input. "
        "Otherwise, or with 'Render input', the input is first rendered to a "
        "temporary EXR sequence so upstream nodes are included."
    )
    group.addKnob(source)
    precision = nuke.Enumeration_Knob("precision", "DiT weights", PRECISIONS)
    precision.setTooltip("bf16: original weights. int8: lower VRAM, weight-only quantization; math stays bf16. "
                         "Changing this replaces the cache after confirmation.")
    group.addKnob(precision)
    low_vram = nuke.Boolean_Knob("low_vram", "low VRAM")
    low_vram.setTooltip("Spatial tokenizer tiles and DiT block offload to RAM. Slower; tiles may change the passes. "
                        "Changing this replaces the cache after confirmation.")
    group.addKnob(low_vram)

    first = nuke.Int_Knob("range_first", "frame range")
    last = nuke.Int_Knob("range_last", "")
    last.clearFlag(nuke.STARTLINE)
    group.addKnob(first)
    group.addKnob(last)
    from_input = nuke.PyScript_Knob(
        "range_from_input", "from input",
        "import plateaovs_nuke; plateaovs_nuke.range_from_input(nuke.thisNode())",
    )
    from_input.clearFlag(nuke.STARTLINE)
    group.addKnob(from_input)
    current = nuke.PyScript_Knob(
        "range_current", "this frame",
        "n = nuke.thisNode(); n['range_first'].setValue(int(nuke.frame())); n['range_last'].setValue(int(nuke.frame()))",
    )
    current.setTooltip("Only the current frame: a quick look (about a minute) before the whole shot. "
                       "Frames already computed stay; the whole shot later redoes only this one, in its window.")
    current.clearFlag(nuke.STARTLINE)
    group.addKnob(current)

    cache = nuke.File_Knob("cache_pattern", "cache")
    cache.setTooltip("EXR sequence written by Compute, or a folder to write it in. Empty uses a folder beside the script.")
    group.addKnob(cache)

    group.addKnob(nuke.Text_Knob("divider_compute", ""))
    group.addKnob(nuke.PyScript_Knob(
        "compute", "Compute",
        "import plateaovs_nuke; plateaovs_nuke.compute(nuke.thisNode())",
    ))
    recompute = nuke.PyScript_Knob(
        "recompute", "Recompute all",
        "import plateaovs_nuke; plateaovs_nuke.compute(nuke.thisNode(), force=True)",
    )
    recompute.clearFlag(nuke.STARTLINE)
    group.addKnob(recompute)
    group.addKnob(nuke.Text_Knob("status", "status", "not computed"))

    group.addKnob(nuke.Text_Knob("divider_info", ""))
    group.addKnob(nuke.Text_Knob(
        "info", "",
        # Text knobs don't wrap: one short line per <br>.
        "Adds <b>albedo</b>, <b>normal</b>, <b>roughness</b>, <b>metallic</b> layers<br>"
        "(and <b>depth_relative</b>), estimated 57 frames at a time.<br>"
        "Normal: bf16 ~19 / int8 ~11.5 GiB reserved.<br>"
        "int8 + low VRAM: ~4.3 GiB, using RAM and block transfers.<br>"
        "8 diffusion steps; a shot shorter than 57 frames still costs one window.<br>"
        "Cosmos DiffusionRenderer, NVIDIA (code Apache-2.0,<br>"
        "weights NVIDIA Open Model License).",
    ))
    version = nuke.Int_Knob("plateaovs_version", "")
    version.setValue(NODE_VERSION)
    version.setVisible(False)
    group.addKnob(version)


def is_plateaovs(node):
    return node is not None and "plateaovs_version" in node.knobs()


def range_from_input(node):
    aitools_nuke.range_from_input(node, TITLE)


def selected_passes(node):
    return [engine_pass for engine_pass, _, knob, _, _ in PASSES if node[knob].value()]


# ------------------------------------------------------------------ compute


def build_command(python, input_pattern, cache_pattern, first, last, passes, force, precision="bf16", low_vram=False):
    command = [
        python, "-m", "plateaovs",
        "--input", input_pattern,
        "--output", cache_pattern,
        "--first", str(first),
        "--last", str(last),
        "--passes",
    ] + list(passes) + ["--precision", precision, "--json-progress"]
    if low_vram:
        command.append("--low-vram")
    if force:
        command.append("--force")
    return command


def _describe_setup(event):
    return "{}x{}, model at {}x{}, {} window(s)".format(
        event["width"], event["height"], event["process_width"], event["process_height"], event["windows"])


def _keeps_cache(cache_pattern, input_pattern, passes, precision, low_vram):
    """Ask before source, passes, precision or tiling replace all frames of a cache."""
    try:
        with open(os.path.join(os.path.dirname(cache_pattern), "plateaovs.json")) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return True
    params = manifest.get("params", {})
    count = len(manifest.get("frames", {}))
    if not count or (sorted(params.get("passes", [])) == sorted(passes)
                     and params.get("input") == input_pattern and params.get("precision", "bf16") == precision
                     and params.get("low_vram", False) == low_vram):
        return True
    return nuke.ask("{}: the cache has {} frame(s) made with another source, passes or memory mode ({}). "
                    "They will all be deleted, also outside the frame range. Continue?".format(
                        TITLE, count, ", ".join(params.get("passes", []))))


def compute(node, force=False, wait=False):
    """Compute the node's cache in the background and reload it when done.

    With ``wait`` the call blocks until the engine exits, which the tests and
    terminal sessions use; the GUI path returns immediately.
    """
    # Upgrade saved nodes: expose the new option without changing their bf16 render.
    if "precision" not in node.knobs():
        node.addKnob(nuke.Enumeration_Knob("precision", "DiT weights", PRECISIONS))
    if "low_vram" not in node.knobs():
        node.addKnob(nuke.Boolean_Knob("low_vram", "low VRAM"))
    node["plateaovs_version"].setValue(NODE_VERSION)
    low_vram = bool(node["low_vram"].value())
    precision = node["precision"].value()
    passes = selected_passes(node)
    if not passes:
        nuke.message("{}: tick at least one pass.".format(TITLE))
        return None
    python = aitools_nuke.check_can_compute(node, TITLE, PROJECT_ROOT, "PLATEAOVS_PYTHON")
    if python is None:
        return None
    first = int(node["range_first"].value())
    last = int(node["range_last"].value())
    cache_pattern = aitools_nuke.cache_pattern(node, "plateaovs")
    input_pattern = aitools_nuke.direct_read(node)
    rendered = input_pattern is None
    source_folder = os.path.join(os.path.dirname(cache_pattern), "source")
    # render_input names its sequence <folder>/<name>.####.exr.
    expected = input_pattern or os.path.join(source_folder, "source.####.exr").replace("\\", "/")
    if nuke.GUI and not _keeps_cache(cache_pattern, expected, passes, precision, low_vram):
        return None
    if rendered:
        input_pattern = aitools_nuke.render_for_compute(
            node, TITLE, "Source", "rgb", [(first, last)], source_folder, "source")
        if input_pattern is None:
            return None
    # A re-rendered source has new file times: the engine redoes those frames itself.
    command = build_command(python, input_pattern, cache_pattern, first, last, passes, force, precision, low_vram)
    job = aitools_nuke.EngineJob(
        node, command, PROJECT_ROOT, TITLE,
        rendered=[os.path.dirname(input_pattern)] if rendered else [],
        describe_setup=_describe_setup,
    )
    return job.start_on(node, wait)
