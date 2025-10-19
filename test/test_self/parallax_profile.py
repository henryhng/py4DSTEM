"""Utility script to time the reference parallax implementation on a dataset.

Run with:
    uv run python test/test_self/parallax_profile.py --dataset PATH_TO_DATA
and optionally tweak scan cropping or reconstruction knobs via CLI flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from py4DSTEM.io import import_file, read as read_emd
from py4DSTEM import DataCube
from emdfile import Root as EmdRoot
from py4DSTEM.process.phase import Parallax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark py4DSTEM parallax reconstruction.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the dataset file (e.g. default1.mib).",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=200_000.0,
        help="Beam energy in eV passed to Parallax (default: 200000).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Execution device for the reconstruction.",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Crop the scan dimensions to max-scan x max-scan before processing.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Forwarded to Parallax.reconstruct to limit virtual BF batches.",
    )
    parser.add_argument(
        "--alignment-bins",
        type=str,
        default=None,
        help="Comma separated list of bin sizes to pass as alignment_bin_values.",
    )
    parser.add_argument(
        "--no-vectorized-com",
        action="store_true",
        help="Disable the vectorized center-of-mass path used during preprocessing.",
    )
    parser.add_argument(
        "--real-pixel-size",
        type=float,
        default=None,
        help="Real-space pixel size in Å (defaults to 1.0 if missing).",
    )
    parser.add_argument(
        "--diffraction-pixel-size",
        type=float,
        default=None,
        help="Diffraction pixel size in 1/Å (defaults to 1.0 if missing).",
    )
    parser.add_argument(
        "--filetype",
        type=str,
        default=None,
        help="Override automatic filetype detection (e.g. 'mib').",
    )
    parser.add_argument(
        "--scan-shape",
        type=int,
        nargs=2,
        metavar=("SCAN_Y", "SCAN_X"),
        help="Optional scan dimensions passed when loading MIB data.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save aligned BF image and shift magnitude map after reconstruction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store visual outputs (default: dataset directory).",
    )
    parser.add_argument(
        "--compare-devices",
        type=str,
        default=None,
        help="Comma separated list of additional devices (e.g. 'gpu') to benchmark.",
    )
    return parser.parse_args()


def maybe_crop_datacube(datacube, max_scan: int | None):
    if max_scan is None:
        return datacube

    scan_y = min(max_scan, datacube.shape[0])
    scan_x = min(max_scan, datacube.shape[1])
    if scan_y == datacube.shape[0] and scan_x == datacube.shape[1]:
        return datacube

    # Reuse DataCube helper to crop in the real-space dimensions.
    return datacube.crop_R((0, scan_y, 0, scan_x))


def ensure_parallax_calibration(
    datacube,
    real_pixel_size: float | None,
    diffraction_pixel_size: float | None,
):
    """Make sure parallax prerequisites are satisfied (Å / Å^-1 calibrations)."""
    calibration = datacube.calibration
    if calibration is None:
        raise ValueError("Datacube is missing calibration metadata.")

    real_size = real_pixel_size if real_pixel_size is not None else calibration.get_R_pixel_size()
    if real_size is None:
        real_size = 1.0
    calibration.set_R_pixel_size(real_size)
    calibration.set_R_pixel_units("A")

    diff_size = (
        diffraction_pixel_size
        if diffraction_pixel_size is not None
        else calibration.get_Q_pixel_size()
    )
    if diff_size is None:
        diff_size = 1.0
    calibration.set_Q_pixel_size(diff_size)
    calibration.set_Q_pixel_units("A^-1")

    origin = calibration.get_origin_mean()
    if origin is None or origin == (None, None):
        calibration.set_origin((datacube.Q_Nx // 2, datacube.Q_Ny // 2))

    datacube.calibrate()
    return datacube


def run_parallax_reference(
    datacube: DataCube,
    args,
    device: str,
    alignment_bins,
    capture_outputs: bool,
) -> dict:
    if device == "gpu":
        try:
            import cupy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("GPU selected but CuPy is not installed. Install cupy to enable GPU backend.") from exc

    parallax = Parallax(
        energy=args.energy,
        datacube=None,
        verbose=False,
        device=device,
        storage=device,
    )
    parallax.attach_datacube(datacube)

    preprocess_start = perf_counter()
    parallax.preprocess(
        plot_average_bf=False,
        progress_bar=False,
        vectorized_com_calculation=not args.no_vectorized_com,
        store_initial_arrays=True,
    )
    preprocess_time = perf_counter() - preprocess_start

    reconstruct_kwargs = dict(
        plot_aligned_bf=False,
        plot_convergence=False,
        progress_bar=False,
        max_batch_size=args.max_batch_size,
        alignment_bin_values=alignment_bins,
        running_average=True,
        regularize_shifts=False,
    )
    if capture_outputs:
        reconstruct_kwargs["return_shifts_and_aligned_bf"] = True

    reconstruct_start = perf_counter()
    recon_result = parallax.reconstruct(**reconstruct_kwargs)
    reconstruct_time = perf_counter() - reconstruct_start

    aligned_bf = shifts_ang = None
    if capture_outputs:
        shifts_ang, aligned_bf = recon_result

        def _to_numpy(array):
            try:
                import cupy as cp
                if isinstance(array, cp.ndarray):
                    return cp.asnumpy(array)
            except ImportError:  # pragma: no cover - cupy optional
                pass
            return np.asarray(array)

        aligned_bf = _to_numpy(aligned_bf)
        shifts_ang = _to_numpy(shifts_ang)

    total_time = preprocess_time + reconstruct_time
    return {
        "device": device,
        "preprocess_time": preprocess_time,
        "reconstruct_time": reconstruct_time,
        "total_time": total_time,
        "aligned_bf": aligned_bf,
        "shifts_ang": shifts_ang,
    }


def save_visualizations(aligned_bf, shifts_ang, output_dir: Path, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    aligned_path = output_dir / f"aligned_bf_{device}.png"
    plt.figure(figsize=(6, 5))
    plt.imshow(aligned_bf, cmap="magma")
    plt.title(f"Aligned BF ({device})")
    plt.colorbar(label="intensity")
    plt.tight_layout()
    plt.savefig(aligned_path)
    plt.close()

    mag = np.linalg.norm(shifts_ang, axis=-1)
    mag_path = output_dir / f"shift_magnitude_{device}.png"
    plt.figure(figsize=(6, 5))
    plt.imshow(mag, cmap="viridis")
    plt.title(f"Shift magnitude (Å) ({device})")
    plt.colorbar(label="Å")
    plt.tight_layout()
    plt.savefig(mag_path)
    plt.close()

    step = max(1, max(shifts_ang.shape[0], shifts_ang.shape[1]) // 32)
    quiver_path = output_dir / f"shift_field_{device}.png"
    plt.figure(figsize=(6, 5))
    plt.quiver(
        shifts_ang[::step, ::step, 0],
        -shifts_ang[::step, ::step, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    plt.title(f"Shift field ({device})")
    plt.tight_layout()
    plt.savefig(quiver_path)
    plt.close()




def main() -> None:
    args = parse_args()

    dataset_path = args.dataset.expanduser()
    dataset_path_str = str(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")

    extra_kwargs = {}
    if args.filetype is not None:
        extra_kwargs["filetype"] = args.filetype
    if args.scan_shape is not None:
        if args.filetype != "mib":
            raise ValueError("--scan-shape is only supported when --filetype=mib")
        extra_kwargs["scan"] = tuple(args.scan_shape)

    if dataset_path.suffix.lower() == ".emd":
        data_obj = read_emd(dataset_path_str, tree=True)
        if isinstance(data_obj, list):
            if not data_obj:
                raise ValueError(f"No datasets found in {dataset_path}")
            data_obj = data_obj[0]
        if isinstance(data_obj, DataCube):
            datacube = data_obj
        elif isinstance(data_obj, EmdRoot):
            if hasattr(data_obj, 'tree'):
                try:
                    datacube = data_obj.tree('root/datacube')
                except AssertionError as exc:
                    raise ValueError('Could not locate datacube node in EMD file; adjust path in script.') from exc
            else:
                raise TypeError(f'Unsupported EMD root type {type(data_obj)}')
        else:
            raise TypeError(f'Unsupported dataset type {type(data_obj)} loaded from {dataset_path}')
    else:
        datacube = import_file(dataset_path_str, mem="RAM", **extra_kwargs)
    datacube = maybe_crop_datacube(datacube, args.max_scan)
    datacube = ensure_parallax_calibration(
        datacube, args.real_pixel_size, args.diffraction_pixel_size
    )

    alignment_bins = None
    if args.alignment_bins:
        alignment_bins = [int(val) for val in args.alignment_bins.split(",") if val.strip()]

    devices = [args.device]
    if args.compare_devices:
        for dev in (item.strip() for item in args.compare_devices.split(",")):
            if dev and dev not in devices:
                devices.append(dev)

    results = []
    for dev in devices:
        capture_outputs = bool(args.visualize)
        print(f"\n=== Running parallax on device: {dev} ===")
        try:
            stats = run_parallax_reference(datacube, args, dev, alignment_bins, capture_outputs)
        except Exception as exc:
            print(f"[warning] Failed on device '{dev}': {exc}")
            continue
        results.append(stats)
        print(f"Preprocess: {stats['preprocess_time']:.2f} s")
        print(f"Reconstruct: {stats['reconstruct_time']:.2f} s")
        print(f"Total: {stats['total_time']:.2f} s")

        if args.visualize and stats['aligned_bf'] is not None and stats['shifts_ang'] is not None:
            out_dir = args.output_dir or dataset_path.parent
            save_visualizations(stats['aligned_bf'], stats['shifts_ang'], Path(out_dir), dev)
            print(f"Saved visualization outputs to {Path(out_dir).resolve()}")

    if not results:
        print('No successful runs to report.')


if __name__ == "__main__":
    main()
