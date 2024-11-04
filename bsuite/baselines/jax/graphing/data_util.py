import concurrent.futures
import json
import os.path
import pathlib
from typing import Any, Dict, Tuple
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import sys
import pandas as pd

KEYS_TO_SAVE = [
    "tags",
    "url",
    "id",
    "name",
    "state",
    "config",
    "created_at",
    "project",
    "entity",
    "path",
    "notes",
    # "read_only",
    # "history_keys",
    # "metadata",
    # "system_metrics",
]
def _run_to_arrow_table(run, save_metadata: bool = True):
    info = {k: getattr(run, k) for k in KEYS_TO_SAVE}
    history = [row for row in run.scan_history()]
    table = pa.Table.from_pylist(history)
    if save_metadata:
        combined_meta = {
            b"wandb": json.dumps(info).encode("utf-8"),
        }
        table = table.replace_schema_metadata(combined_meta)
    return table
def read_wandb_metadata(path: os.PathLike, filename: bool = True) -> Dict[str, Any]:
    parquet_metadata = pq.read_metadata(path)
    wandb_metadata = json.loads(parquet_metadata.metadata[b"wandb"])
    if filename:
        wandb_metadata["filename"] = os.fspath(path)
    return wandb_metadata
def read_wandb_history(files):
    rel = duckdb.read_parquet(files, union_by_name=True, filename=True)
    rel = rel.select("parse_filename(filename, true) as id, COLUMNS(*)")
    return rel
def _write_run_to_parquet(run, path: os.PathLike):
    table = _run_to_arrow_table(run, save_metadata=True)
    pq.write_table(table, path)
def _run_filename(run):
    return f"{run.id}.parquet"
def write_runs_to_parquet(
    runs,
    save_dir: os.PathLike,
    partition_cols: Tuple[str, ...] = (),
    num_workers: int = 16,
    overwrite: bool = False,
):
    runs_to_write = []
    for run in runs:
        if run.state == "finished":  # TODO added the run.finished check
            subpaths = [str(getattr(run, col)) for col in partition_cols]
            run_path = pathlib.Path(save_dir).joinpath(*subpaths)
            filepath = run_path / _run_filename(run)
            if not filepath.exists() or overwrite:
                runs_to_write.append((run, filepath))
    def download(run, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_run_to_parquet(run, os.fspath(path))
    if not runs_to_write:
        print("Skip writing runs.")
        return
    print(f"Writing {len(runs_to_write)} runs to {save_dir}.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        fs = []
        for run, path in runs_to_write:
            fs.append(executor.submit(download, run, path))
        for future in tqdm.tqdm(fs):
            future.result()


