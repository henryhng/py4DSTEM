import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt


def walk_datasets(file: h5py.File):
    for name, obj in file.items():
        yield from _walk(name, obj)


def _walk(prefix: str, obj):
    if isinstance(obj, h5py.Dataset):
        yield prefix, obj
    elif isinstance(obj, h5py.Group):
        for key, value in obj.items():
            yield from _walk(f"{prefix}/{key}", value)


def visualize_dataset(name: str, dataset: h5py.Dataset, max_elements: int):
    data = dataset[...]
    print(f"Dataset: {name}  shape={data.shape}  dtype={data.dtype}")
    flat_size = data.size
    if flat_size == 0:
        print("  [empty dataset]")
        return
    if flat_size > max_elements:
        print(f"  [skipping visualization; elements {flat_size} exceeds limit {max_elements}]")
        return

    if data.ndim == 1:
        plt.figure()
        plt.plot(data)
        plt.title(name)
        return

    display = data
    slice_notes = ""
    while display.ndim > 2:
        display = display[0]
        slice_notes += "[0]"

    if display.ndim == 1:
        fig = plt.figure()
        plt.plot(display)
        plt.title(f"{name}{slice_notes}")
    elif display.ndim == 2:
        fig = plt.figure()
        plt.imshow(display, aspect="auto", cmap="viridis")
        plt.title(f"{name}{slice_notes}")
        plt.colorbar()
        if slice_notes:
            print(f"  [showing first slice {slice_notes}]")
    else:
        return

    safe_name = name.strip('/').replace('/', '_') or 'root'
    suffix = slice_notes.replace('[', '_').replace(']', '')
    outfile = Path(f"{safe_name}{suffix}.png")
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved {outfile}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk an HDF5 file and visualize datasets")
    parser.add_argument("path", type=Path, help="Path to the .h5 file")
    parser.add_argument(
        "--max-elements",
        type=int,
        default=1_000_000,
        help="Skip visualization if dataset has more elements than this threshold",
    )
    args = parser.parse_args()

    h5_path = args.path.expanduser()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        found = False
        for name, dataset in walk_datasets(f):
            found = True
            visualize_dataset(name, dataset, args.max_elements)
        if not found:
            print("No datasets found in file.")

    if plt.get_fignums():
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
