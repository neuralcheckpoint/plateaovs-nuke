"""Command line entry point: ``python -m plateaovs``.

Writes the plate's G-buffer passes (albedo, normals, roughness, metallic,
depth) as an EXR sequence (see pipeline.py). With ``--json-progress`` every
event is printed as one JSON object per line.
"""

import argparse
import sys

from aitools_engine.events import run_cli

from . import pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(prog="plateaovs", description=__doc__)
    parser.add_argument("--input", required=True, help="plate pattern, e.g. plate.####.exr")
    parser.add_argument("--output", required=True, help="cache pattern, e.g. cache/aovs.####.exr")
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--passes", nargs="+", choices=pipeline.PASSES, default=list(pipeline.PASSES))
    parser.add_argument("--window", type=int, default=57, choices=(1, 57),
                        help="frames the model sees at once: 57, or 1 (each frame alone, flickers)")
    parser.add_argument("--overlap", type=int, default=8, help="frames cross-faded between windows")
    parser.add_argument("--steps", type=int, default=8, help="diffusion steps per pass")
    parser.add_argument("--precision", choices=["bf16", "int8"], default="bf16",
                        help="DiT weights: bf16 (default) or int8 weight-only to save VRAM")
    parser.add_argument("--low-vram", action="store_true",
                        help="spatially tile the tokenizer and offload DiT blocks to RAM (slower, may change passes)")
    parser.add_argument("--transfer", choices=["auto", "linear", "display"], default="auto")
    parser.add_argument("--force", action="store_true", help="rewrite frames already cached")
    parser.add_argument("--json-progress", action="store_true")
    args = parser.parse_args(argv)

    def work(report):
        return pipeline.run(args.input, args.first, args.last, args.output, passes=args.passes,
                            window=args.window, overlap=args.overlap, steps=args.steps,
                            transfer=args.transfer, force=args.force, progress=report, precision=args.precision,
                            low_vram=args.low_vram)

    return run_cli(args.json_progress, work, "Close other GPU apps (ComfyUI, other Nuke sessions) "
                   "or use a shorter window.")


if __name__ == "__main__":
    sys.exit(main())
