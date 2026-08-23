from pathlib import Path

import torch


def _gen_output_path(
    data_type: str, pipeline_name: str, sample_rate: float, output_dir: Path
) -> Path:
    return output_dir / f"{data_type}_{sample_rate}Hz_{pipeline_name}.pt"


def save_chkpt(chkpt, output_dir) -> None:
    output_path: Path = _gen_output_path(
        chkpt["data_type"], chkpt["pipeline"], chkpt["sample_rate_hz"], output_dir
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(chkpt, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
