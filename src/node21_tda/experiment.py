"""Deney kayıt (registry) ve manifest yardımcıları.

Amaç: Çok sayıda deneyin tekrar üretilebilir ve karşılaştırılabilir olmasını sağlamak.
Her deney çalıştırması ``artifacts/runs/<ID>_<slug>/manifest.json`` yazar; bu dosya
git commit'ini, konfigürasyon hash'ini, girdi dosyalarının hash'ini, seed'i ve
metrikleri kaydeder.

Kullanım::

    from node21_tda.experiment import new_run_dir, write_manifest

    run_dir = new_run_dir("E1.1", "bone_suppression")
    write_manifest(
        run_dir, "E1.1",
        config=cfg, config_path=config_path,
        inputs=[features_path, splits_path],
        seed=42, metrics={"val_roc_auc": 0.8712},
    )
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    """Dosyanın SHA-256 özetini döner; dosya yoksa ``"missing"``."""
    path = Path(path)
    if not path.exists():
        return "missing"
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json(obj: Any) -> str:
    """Nesneyi deterministik (anahtar sıralı) JSON metnine çevirir."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(config: Any) -> str:
    """Konfigürasyonun içerik hash'i (aynı config -> aynı hash)."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def git_state(repo_root: Path | None = None) -> dict[str, Any]:
    """Git commit hash'i, dal adı ve çalışma ağacının kirli olup olmadığını döner.

    git yoksa veya çalıştırılamıyorsa sessizce ``"unknown"`` değerleriyle döner;
    deney akışını asla bloke etmez.
    """
    root = Path(repo_root) if repo_root else PROJECT_ROOT
    state: dict[str, Any] = {"commit": "unknown", "branch": "unknown", "dirty": None}

    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = _run("rev-parse", "HEAD")
    if commit:
        state["commit"] = commit
    branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        state["branch"] = branch
    status = _run("status", "--porcelain")
    if status is not None:
        state["dirty"] = bool(status)
    return state


def new_run_dir(experiment_id: str, slug: str, artifacts_root: Path | None = None) -> Path:
    """``artifacts/runs/<ID>_<slug>/`` dizinini oluşturur ve döner."""
    root = Path(artifacts_root) if artifacts_root else PROJECT_ROOT / "artifacts" / "runs"
    run_dir = root / f"{experiment_id}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(
    run_dir: Path,
    experiment_id: str,
    *,
    config: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
    inputs: Iterable[Path] | None = None,
    seed: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    decision: str | None = None,
    notes: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Tek bir deney çalıştırması için ``manifest.json`` yazar ve yolunu döner."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
    }

    if config is not None:
        manifest["config"] = dict(config)
        manifest["config_sha256"] = config_hash(config)

    if config_path is not None:
        config_path = Path(config_path)
        manifest["config_path"] = str(config_path)
        manifest["config_file_sha256"] = file_sha256(config_path)

    if inputs:
        manifest["inputs"] = {str(Path(p)): file_sha256(Path(p)) for p in inputs}

    if metrics is not None:
        manifest["metrics"] = dict(metrics)
    if decision is not None:
        manifest["decision"] = decision
    if notes is not None:
        manifest["notes"] = notes
    if extra is not None:
        manifest["extra"] = dict(extra)

    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest_path
