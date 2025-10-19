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
    parser.add_argument(
        "--safe-gpu",
        action="store_true",
        help="Reduce GPU workload by auto-cropping and lowering batch sizes.",
    )
    parser.add_argument(
        "--safe-gpu-max-scan",
        type=int,
        default=128,
        help="Maximum scan dimension when --safe-gpu is enabled.",
    )
    parser.add_argument(
        "--safe-gpu-max-batch",
        type=int,
        default=32,
        help="Maximum virtual BF batch size when --safe-gpu is enabled.",
    )
    parser.add_argument(
        "--safe-gpu-keep-vectorized-com",
        action="store_true",
        help="Keep the vectorized CoM path when --safe-gpu is enabled (default disables it).",
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
    overrides: dict | None = None,
) -> dict:
    overrides = overrides or {}
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

    vectorized_com = overrides.get(
        "vectorized_com_calculation", not args.no_vectorized_com
    )
    selected_max_batch = (
        overrides["max_batch_size"]
        if overrides.get("max_batch_size") is not None
        else args.max_batch_size
    )

    preprocess_start = perf_counter()
    parallax.preprocess(
        plot_average_bf=False,
        progress_bar=False,
        vectorized_com_calculation=vectorized_com,
        store_initial_arrays=True,
        max_batch_size=selected_max_batch,
    )
    preprocess_time = perf_counter() - preprocess_start

    reconstruct_kwargs = dict(
        plot_aligned_bf=False,
        plot_convergence=False,
        progress_bar=False,
        max_batch_size=selected_max_batch,
        alignment_bin_values=alignment_bins,
        running_average=True,
        regularize_shifts=False,
    )
    if "alignment_bin_values" in overrides:
        reconstruct_kwargs["alignment_bin_values"] = overrides["alignment_bin_values"]
    if "running_average" in overrides and overrides["running_average"] is not None:
        reconstruct_kwargs["running_average"] = overrides["running_average"]
    if "clear_fft_cache" in overrides and overrides["clear_fft_cache"] is not None:
        reconstruct_kwargs["clear_fft_cache"] = overrides["clear_fft_cache"]
    if capture_outputs:
        reconstruct_kwargs["return_shifts_and_aligned_bf"] = True

    reconstruct_start = perf_counter()
    try:
        recon_result = parallax.reconstruct(**reconstruct_kwargs)
        reconstruct_time = perf_counter() - reconstruct_start
    finally:
        if device == "gpu":
            try:
                import cupy as cp
                default_pool = getattr(cp.cuda, 'get_default_memory_pool', None)
                if callable(default_pool):
                    default_pool().free_all_blocks()
                pinned_pool = getattr(cp.cuda, 'get_default_pinned_memory_pool', None)
                if callable(pinned_pool):
                    pinned_pool().free_all_blocks()
            except ImportError:  # pragma: no cover - cupy optional
                pass


    aligned_bf = shifts_ang = None
    if capture_outputs:
        if isinstance(recon_result, tuple) and len(recon_result) == 2:
            shifts_ang, aligned_bf = recon_result
        else:
            # Some Parallax builds return self; fall back to attributes if available.
            shifts_ang = getattr(parallax, 'shifts_ang', None)
            aligned_bf = getattr(parallax, 'aligned_bf', None)
        if shifts_ang is not None and aligned_bf is not None:

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
        else:
            print('[warning] Visualization requested but Parallax did not provide output arrays.')
            capture_outputs = False

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

    datacube = ensure_parallax_calibration(
        datacube, args.real_pixel_size, args.diffraction_pixel_size
    )
    base_datacube = maybe_crop_datacube(datacube, args.max_scan)

    alignment_bins = None
    if args.alignment_bins:
        alignment_bins = [int(val) for val in args.alignment_bins.split(",") if val.strip()]

    devices = [args.device]
    if args.compare_devices:
        for dev in (item.strip() for item in args.compare_devices.split(",")):
            if dev and dev not in devices:
                devices.append(dev)

    if args.safe_gpu and "gpu" not in devices:
        print("[warning] --safe-gpu was requested but no GPU run is scheduled.")

    results = []
    for dev in devices:
        capture_outputs = bool(args.visualize)
        device_datacube = base_datacube
        device_alignment_bins = alignment_bins
        device_overrides = {}

        if dev == "gpu" and args.safe_gpu:
            safe_datacube = maybe_crop_datacube(base_datacube, args.safe_gpu_max_scan)
            if safe_datacube.shape[:2] != base_datacube.shape[:2]:
                device_datacube = safe_datacube
                print(
                    f"[info] GPU safe mode: cropped scan to {device_datacube.shape[0]}x{device_datacube.shape[1]}."
                )
            if args.max_batch_size is None:
                device_overrides["max_batch_size"] = args.safe_gpu_max_batch
                print(f"[info] GPU safe mode: limiting max batch size to {args.safe_gpu_max_batch}.")
            if not args.safe_gpu_keep_vectorized_com and not args.no_vectorized_com:
                device_overrides["vectorized_com_calculation"] = False
                print("[info] GPU safe mode: disabling vectorized CoM.")
            device_overrides["clear_fft_cache"] = True
            if device_alignment_bins and len(device_alignment_bins) > 4:
                device_alignment_bins = device_alignment_bins[:4]
                print("[info] GPU safe mode: trimming alignment bins to first 4 values.")

        print(f"\n=== Running parallax on device: {dev} ===")
        try:
            stats = run_parallax_reference(
                device_datacube,
                args,
                dev,
                device_alignment_bins,
                capture_outputs,
                overrides=device_overrides,
            )
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
