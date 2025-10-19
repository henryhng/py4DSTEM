"""Stitch a set of Merlin .mib tiles into a single py4DSTEM DataCube.

Example:
    python test/test_self/stitch_mib_dataset.py \
        --source test/test_self/highresdataaset \
        --scan-shape 16 16 \
        --tiles-per-side 8 \
        --real-pixel-size 5.0 \
        --diffraction-pixel-size 0.5 \
        --output test/test_self/highresdataaset/full_dataset.emd
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from py4DSTEM import DataCube
from py4DSTEM.io import save as save_emd
from py4DSTEM.io.filereaders.read_mib import load_mib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch Merlin .mib tiles into a single DataCube")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing default*.mib tiles and their header.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .emd file for the stitched datacube.",
    )
    parser.add_argument(
        "--scan-shape",
        type=int,
        nargs=2,
        metavar=("SCAN_Y", "SCAN_X"),
        default=(16, 16),
        help="Per-tile scan dimensions (default: 16 16).",
    )
    parser.add_argument(
        "--tiles-per-side",
        type=int,
        default=8,
        help="Number of tiles along each scan axis (default: 8 for 64 tiles).",
    )
    parser.add_argument(
        "--real-pixel-size",
        type=float,
        default=None,
        help="Real-space pixel size in Å; if omitted keeps the incoming value.",
    )
    parser.add_argument(
        "--diffraction-pixel-size",
        type=float,
        default=None,
        help="Diffraction pixel size in 1/Å; if omitted keeps the incoming value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def collect_tiles(source: Path) -> Sequence[Path]:
    def tile_key(path: Path) -> int:
        stem = path.stem.replace("default", "")
        return int(stem) if stem else 0

    paths = sorted(source.glob("default*.mib"), key=tile_key)
    if not paths:
        raise FileNotFoundError(f"No default*.mib files found in {source}")
    return paths


def stitch_tiles(paths: Sequence[Path], scan_shape: tuple[int, int], tiles_per_side: int) -> np.ndarray:
    scan_y, scan_x = scan_shape
    frames = []
    for path in paths:
        cube = load_mib(str(path), mem="RAM", reshape=True, scan=scan_shape)
        frames.append(cube.data.reshape(-1, cube.data.shape[-2], cube.data.shape[-1]))

    stack = np.concatenate(frames, axis=0)
    expected_tiles = tiles_per_side * tiles_per_side
    if len(paths) != expected_tiles:
        raise ValueError("Number of tiles does not match tiles_per_side**2; check your inputs.")

    stitched = stack.reshape(
        tiles_per_side * scan_y,
        tiles_per_side * scan_x,
        frames[0].shape[-2],
        frames[0].shape[-1],
    )
    return stitched


def apply_calibration(
    datacube: DataCube,
    real_pixel_size: float | None,
    diffraction_pixel_size: float | None,
) -> None:
    calibration = datacube.calibration
    if real_pixel_size is not None:
        calibration.set_R_pixel_size(real_pixel_size)
        calibration.set_R_pixel_units("A")
    if diffraction_pixel_size is not None:
        calibration.set_Q_pixel_size(diffraction_pixel_size)
        calibration.set_Q_pixel_units("A^-1")

    if calibration.get_origin_mean() in (None, (None, None)):
        calibration.set_origin((datacube.Q_Nx // 2, datacube.Q_Ny // 2))
    datacube.calibrate()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser()
    output = args.output.expanduser()

    if not source.exists():
        raise FileNotFoundError(f"Source directory {source} not found")

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output file {output} already exists (use --overwrite to replace)")

    tiles = collect_tiles(source)
    stitched = stitch_tiles(tiles, tuple(args.scan_shape), args.tiles_per_side)

    datacube = DataCube(stitched)
    apply_calibration(datacube, args.real_pixel_size, args.diffraction_pixel_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_emd(str(output), datacube, mode='w')
    print(f"Wrote stitched datacube to {output}")


if __name__ == "__main__":
    main()
