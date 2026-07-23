"""Fetchers for the public OCSR sources behind ``glyph data prepare --ocsr``.

Every fetch here touches the network and, for the training corpora, can be
multi-gigabyte. To keep the pipeline safe by default, large downloads are
*refused* unless the caller passes
``allow_large=True`` -- the CLI surfaces this as ``--allow-large``. ``--dry-run``
plans without touching the network, and PubChem-1M additionally supports a cheap
``limit``-based streaming head so a small validation run never pulls the full ~210 MB CSV.

Upstream (pinned in :mod:`glyph.data.layout`):

* PubChem-1M + USPTO-680K + MolScribe recovery CSV: ``CYF200127/MolNexTR``
  (files ``train_pubchem.csv``, ``train_uspto.zip``, ``real.zip``).
* Stereo-200K base images: ``Robert-zwr/Stereo-200k`` (``stereo.zip``).

None of these are re-hosted by this work; they are referenced by manifest.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from glyph.data.layout import (
    MOLNEXTR_REPO,
    MOLNEXTR_REVISION,
    STATUS_MATERIALIZED,
    STATUS_PLANNED,
    STATUS_REFUSED,
    STEREO_REPO,
    STEREO_REVISION,
)

ALLOW_LARGE_HINT = "re-run with --allow-large (or --limit N where supported) or use --dry-run"


@dataclass
class FetchOutcome:
    """Result of a single source fetch/extract."""

    status: str
    detail: str
    path: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)


def _hf_auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def stream_csv_head(
    filename: str,
    out_csv: Path,
    *,
    limit: int,
    repo_id: str = MOLNEXTR_REPO,
    revision: str = MOLNEXTR_REVISION,
) -> int:
    """Stream the header + first ``limit`` rows of an HF-hosted CSV.

    Reads only the leading bytes over HTTP (the connection is closed once enough
    lines are seen), so this never downloads the whole file. Returns the number
    of data rows written.
    """

    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with requests.get(url, headers=_hf_auth_headers(), stream=True, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            lines.append(line)
            if len(lines) >= limit + 1:  # header + ``limit`` data rows
                break
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return max(len(lines) - 1, 0)


def _download_hf_file(
    filename: str,
    *,
    repo_id: str,
    revision: str,
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )


def fetch_pubchem_1m(
    out_csv: Path,
    *,
    limit: int | None = None,
    allow_large: bool = False,
    dry_run: bool = False,
) -> FetchOutcome:
    """Fetch ``train_pubchem.csv`` into ``out_csv`` (full, or a streamed head)."""

    if dry_run:
        plan = (
            f"stream first {limit} rows"
            if limit is not None
            else f"download {MOLNEXTR_REPO}:train_pubchem.csv (~210 MB) @ {MOLNEXTR_REVISION}"
        )
        return FetchOutcome(STATUS_PLANNED, f"would {plan} -> {out_csv}", path=out_csv)

    if limit is not None:
        rows = stream_csv_head("train_pubchem.csv", out_csv, limit=limit)
        return FetchOutcome(
            STATUS_MATERIALIZED,
            f"streamed {rows} rows (limit={limit}) from {MOLNEXTR_REPO}:train_pubchem.csv",
            path=out_csv,
            counts={"rows": rows},
        )

    if not allow_large:
        return FetchOutcome(
            STATUS_REFUSED,
            f"train_pubchem.csv is ~210 MB; {ALLOW_LARGE_HINT}",
            path=out_csv,
        )

    src = _download_hf_file("train_pubchem.csv", repo_id=MOLNEXTR_REPO, revision=MOLNEXTR_REVISION)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_csv)
    return FetchOutcome(
        STATUS_MATERIALIZED,
        f"downloaded {MOLNEXTR_REPO}:train_pubchem.csv @ {MOLNEXTR_REVISION}",
        path=out_csv,
    )


def _extract_zip(archive: Path, dest_root: Path, *, limit: int | None = None) -> int:
    """Extract ``archive`` into ``dest_root``; return the count of members written.

    ``limit`` caps the number of *image* members extracted (any non-image member
    such as a CSV is always extracted so the split stays usable).
    """

    dest_root.mkdir(parents=True, exist_ok=True)
    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    written = 0
    images_written = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename
            # Guard against path traversal in the archive.
            target = (dest_root / name).resolve()
            if not str(target).startswith(str(dest_root.resolve())):
                continue
            is_image = Path(name).suffix.lower() in image_suffixes
            if is_image and limit is not None and images_written >= limit:
                continue
            zf.extract(member, dest_root)
            written += 1
            if is_image:
                images_written += 1
    return written


def fetch_uspto_680k(
    out_dir: Path,
    *,
    limit: int | None = None,
    allow_large: bool = False,
    dry_run: bool = False,
) -> FetchOutcome:
    """Download + extract ``train_uspto.zip`` (images + CSV) into ``out_dir``."""

    if dry_run:
        return FetchOutcome(
            STATUS_PLANNED,
            f"would download {MOLNEXTR_REPO}:train_uspto.zip (~2.1 GB) @ {MOLNEXTR_REVISION} "
            f"and extract -> {out_dir}",
            path=out_dir,
        )
    if not allow_large:
        return FetchOutcome(
            STATUS_REFUSED,
            f"train_uspto.zip is ~2.1 GB of images; {ALLOW_LARGE_HINT}",
            path=out_dir,
        )
    archive = _download_hf_file(
        "train_uspto.zip", repo_id=MOLNEXTR_REPO, revision=MOLNEXTR_REVISION
    )
    written = _extract_zip(archive, out_dir, limit=limit)
    return FetchOutcome(
        STATUS_MATERIALIZED,
        f"extracted {written} members from train_uspto.zip @ {MOLNEXTR_REVISION}",
        path=out_dir,
        counts={"members": written},
    )


def fetch_stereo_200k(
    out_dir: Path,
    *,
    allow_large: bool = False,
    dry_run: bool = False,
) -> FetchOutcome:
    """Download + extract the external ``Robert-zwr/Stereo-200k`` ``stereo.zip``.

    Returns the extraction root; the caller runs the white-crop pass over it.
    """

    if dry_run:
        return FetchOutcome(
            STATUS_PLANNED,
            f"would download {STEREO_REPO}:stereo.zip (~4.3 GB) @ {STEREO_REVISION} "
            f"and extract -> {out_dir}",
            path=out_dir,
        )
    if not allow_large:
        return FetchOutcome(
            STATUS_REFUSED,
            f"stereo.zip is ~4.3 GB (external MolSight/Robert-zwr images); {ALLOW_LARGE_HINT}",
            path=out_dir,
        )
    archive = _download_hf_file("stereo.zip", repo_id=STEREO_REPO, revision=STEREO_REVISION)
    written = _extract_zip(archive, out_dir)
    return FetchOutcome(
        STATUS_MATERIALIZED,
        f"extracted {written} members from {STEREO_REPO}:stereo.zip @ {STEREO_REVISION}",
        path=out_dir,
        counts={"members": written},
    )


def fetch_molscribe_real_uspto_csv(
    dest_csv: Path,
    *,
    allow_large: bool = False,
    dry_run: bool = False,
) -> FetchOutcome:
    """Fetch MolScribe ``real/USPTO.csv`` (from ``real.zip``) for 5719 recovery."""

    if dry_run:
        return FetchOutcome(
            STATUS_PLANNED,
            f"would download {MOLNEXTR_REPO}:real.zip (~267 MB) @ {MOLNEXTR_REVISION} "
            f"and extract real/USPTO.csv -> {dest_csv}",
            path=dest_csv,
        )
    if not allow_large:
        return FetchOutcome(
            STATUS_REFUSED,
            f"real.zip is ~267 MB; {ALLOW_LARGE_HINT}",
            path=dest_csv,
        )
    archive = _download_hf_file("real.zip", repo_id=MOLNEXTR_REPO, revision=MOLNEXTR_REVISION)
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        member = next(
            (n for n in zf.namelist() if n.replace("\\", "/").endswith("real/USPTO.csv")), None
        )
        if member is None:
            return FetchOutcome(
                STATUS_REFUSED,
                "real.zip did not contain real/USPTO.csv",
                path=dest_csv,
            )
        with zf.open(member) as source, dest_csv.open("wb") as handle:
            shutil.copyfileobj(source, handle)
    return FetchOutcome(
        STATUS_MATERIALIZED,
        f"extracted real/USPTO.csv from {MOLNEXTR_REPO}:real.zip @ {MOLNEXTR_REVISION}",
        path=dest_csv,
    )
