"""Read/write fold-level training progress for HGPRGDel CV runs."""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from utils.paths import glob_fold_best_checkpoints

PROGRESS_FILENAME = "progress.json"
TRAIN_LOG_FILENAME = "train.log"
DEFAULT_STALE_MINUTES = 30
BATCH_PROGRESS_MIN_INTERVAL_SEC = 60.0

HealthVerdict = Literal[
    "running_ok",
    "stale",
    "process_dead",
    "likely_crashed",
    "completed",
    "training_complete",
    "interrupted",
    "stopped",
    "crashed",
    "unknown",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def progress_path(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir) / PROGRESS_FILENAME


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class TrainingProgressTracker:
    """Writes ``progress.json`` under ``experiment_dir`` during CV training."""

    def __init__(
        self,
        experiment_dir: str | Path,
        *,
        num_folds: int,
        num_epochs: int,
        cbm: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.path = progress_path(experiment_dir)
        self.num_folds = int(num_folds)
        self.num_epochs = int(num_epochs)
        self.cbm = cbm
        self.batch_size = batch_size
        self.started_at = _utc_now()
        self.completed_folds: list[dict[str, Any]] = []
        self._payload: dict[str, Any] = {
            "status": "running",
            "cbm": cbm,
            "num_folds": self.num_folds,
            "num_epochs": self.num_epochs,
            "batch_size": batch_size,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": self.started_at,
            "updated_at": self.started_at,
            "completed_folds": self.completed_folds,
            "current": None,
        }
        self._flush()

    def _flush(self) -> None:
        self._payload["updated_at"] = _utc_now()
        self._payload["completed_folds"] = self.completed_folds
        _atomic_write(self.path, self._payload)

    def start_fold(self, fold: int, train_metas: set[str], val_metas: set[str]) -> None:
        self._payload["current"] = {
            "fold": fold + 1,
            "epoch": 0,
            "train_metabolites": len(train_metas),
            "val_metabolites": len(val_metas),
            "train_loss": None,
            "train_f1": None,
            "val_loss": None,
            "val_f1": None,
            "best_val_f1": 0.0,
            "epochs_no_improve": 0,
        }
        self._flush()

    def update_epoch(
        self,
        fold: int,
        epoch: int,
        *,
        train_loss: float,
        train_acc: float,
        train_f1: float,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        best_f1: float,
        epochs_no_improve: int,
    ) -> None:
        self._payload["current"] = {
            "fold": fold + 1,
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "train_f1": round(train_f1, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_f1": round(val_f1, 6),
            "best_val_f1": round(best_f1, 6),
            "epochs_no_improve": epochs_no_improve,
        }
        self._flush()

    def update_training_batch(
        self,
        fold: int,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        *,
        phase: str = "train",
        loss_ema: float | None = None,
    ) -> None:
        """Refresh ``progress.json`` during long epochs (for monitoring without notebook)."""
        current = dict(self._payload.get("current") or {})
        batch_n = int(batch_idx) + 1
        total_n = int(total_batches)
        current.update(
            {
                "fold": fold + 1,
                "epoch": int(epoch),
                "batch": batch_n,
                "total_batches": total_n,
                "batch_frac": round(batch_n / total_n, 4) if total_n else None,
                "phase": phase,
            }
        )
        if loss_ema is not None:
            current["batch_loss_ema"] = round(float(loss_ema), 6)
        self._payload["current"] = current
        self._flush()

    def complete_fold(
        self,
        fold: int,
        *,
        best_f1: float,
        epochs_run: int,
        checkpoint: str | None = None,
    ) -> None:
        entry = {
            "fold": fold + 1,
            "best_val_f1": round(best_f1, 6),
            "epochs_run": epochs_run,
        }
        if checkpoint:
            entry["checkpoint"] = checkpoint
        self.completed_folds.append(entry)
        self._payload["current"] = None
        self._flush()

    def mark_interrupted(self, fold: int, epoch: int) -> None:
        self._payload["status"] = "interrupted"
        current = dict(self._payload.get("current") or {})
        current["fold"] = fold + 1
        current["epoch"] = epoch + 1
        self._payload["current"] = current
        self._flush()

    def mark_crashed(self, fold: int, epoch: int, error: BaseException | str) -> None:
        message = str(error)
        if len(message) > 500:
            message = message[:497] + "..."
        self._payload["status"] = "crashed"
        self._payload["error"] = message
        self._payload["crashed_at"] = _utc_now()
        current = dict(self._payload.get("current") or {})
        current["fold"] = fold + 1
        current["epoch"] = epoch + 1
        self._payload["current"] = current
        self._flush()

    def mark_stopped(self, reason: str) -> None:
        self._payload["status"] = "stopped"
        self._payload["stop_reason"] = reason
        self._payload["stopped_at"] = _utc_now()
        self._payload["current"] = None
        self._flush()

    def mark_training_complete(self, mean_f1: float | None = None) -> None:
        self._payload["status"] = "training_complete"
        self._payload["current"] = None
        if mean_f1 is not None:
            self._payload["mean_val_f1"] = round(mean_f1, 6)
        self._flush()

    def mark_completed(self) -> None:
        self._payload["status"] = "completed"
        self._payload["current"] = None
        self._flush()


def mark_experiment_completed(experiment_dir: str | Path) -> None:
    """Set status to ``completed`` after inference, preserving fold history."""
    path = progress_path(experiment_dir)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = infer_progress_from_artifacts(experiment_dir)
    payload["status"] = "completed"
    payload["current"] = None
    payload["updated_at"] = _utc_now()
    _atomic_write(path, payload)


def _load_config(experiment_dir: Path) -> dict[str, Any]:
    cfg_path = experiment_dir / "config.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def infer_progress_from_artifacts(experiment_dir: str | Path) -> dict[str, Any]:
    """Best-effort progress for runs started before ``progress.json`` existed."""
    root = Path(experiment_dir)
    cfg = _load_config(root)
    num_folds = int(cfg.get("num_folds") or 5)
    ckpt_dir = root / "checkpoints"
    completed: list[dict[str, Any]] = []
    if ckpt_dir.is_dir():
        for path in glob_fold_best_checkpoints(ckpt_dir):
            parts = path.stem.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                completed.append(
                    {
                        "fold": int(parts[1]) + 1,
                        "checkpoint": str(path.relative_to(root)),
                    }
                )
    completed.sort(key=lambda x: x["fold"])

    metrics_path = root / "metrics.json"
    mean_f1 = None
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            mean_f1 = metrics.get("mean_f1")
            for row in metrics.get("fold_results") or []:
                fold_no = int(row["fold"])
                if not any(c["fold"] == fold_no for c in completed):
                    completed.append({"fold": fold_no, "best_val_f1": row.get("best_f1")})
            completed.sort(key=lambda x: x["fold"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    predictions = (root / "predictions" / "all_metabolites_predictions.csv").is_file()
    n_done = len(completed)
    if predictions and n_done >= num_folds:
        status = "completed"
    elif n_done >= num_folds:
        status = "training_complete"
    elif n_done > 0:
        status = "running"
    else:
        status = "unknown"

    return {
        "status": status,
        "source": "inferred_from_artifacts",
        "cbm": cfg.get("CBM"),
        "num_folds": num_folds,
        "num_epochs": cfg.get("num_epochs"),
        "completed_folds": completed,
        "current": None,
        "mean_val_f1": mean_f1,
        "has_predictions": predictions,
    }


def load_progress(experiment_dir: str | Path) -> dict[str, Any]:
    path = progress_path(experiment_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return infer_progress_from_artifacts(experiment_dir)


def parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def minutes_since_update(progress: dict[str, Any], *, now: datetime | None = None) -> float | None:
    updated = parse_utc_timestamp(progress.get("updated_at"))
    if updated is None:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return max(0.0, (now - updated).total_seconds() / 60.0)


def is_pid_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _current_fold_started_without_metrics(progress: dict[str, Any]) -> bool:
    current = progress.get("current")
    if not isinstance(current, dict):
        return False
    epoch = current.get("epoch")
    if epoch not in (0, None):
        return False
    metric_keys = ("train_f1", "val_f1", "train_loss", "val_loss")
    return all(current.get(key) is None for key in metric_keys)


def assess_training_health(
    progress: dict[str, Any],
    *,
    stale_minutes: int = DEFAULT_STALE_MINUTES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Cross-check ``progress.json`` against process liveness and update staleness."""
    status = str(progress.get("status") or "unknown")
    pid = progress.get("pid")
    pid_int = int(pid) if pid is not None else None
    pid_alive = is_pid_alive(pid_int)
    stale_for = minutes_since_update(progress, now=now)
    is_stale = stale_for is not None and stale_for >= float(stale_minutes)

    terminal_statuses = {
        "completed",
        "training_complete",
        "interrupted",
        "stopped",
        "crashed",
    }
    if status in terminal_statuses:
        verdict: HealthVerdict = status  # type: ignore[assignment]
        trust_status = status
        if status == "completed":
            message = "Training finished (predictions written)."
        elif status == "training_complete":
            message = "CV training finished; inference may still be pending."
        elif status == "crashed":
            message = "Training process recorded a crash."
        elif status == "stopped":
            message = "Training was stopped manually."
        else:
            message = f"Terminal status: {status}."
        return {
            "verdict": verdict,
            "trust_status": trust_status,
            "message": message,
            "pid": pid_int,
            "pid_alive": pid_alive,
            "minutes_since_update": stale_for,
            "is_stale": is_stale,
            "stale_minutes": stale_minutes,
        }

    if status != "running":
        return {
            "verdict": "unknown",
            "trust_status": status,
            "message": f"Unrecognized status {status!r}; verify process and logs.",
            "pid": pid_int,
            "pid_alive": pid_alive,
            "minutes_since_update": stale_for,
            "is_stale": is_stale,
            "stale_minutes": stale_minutes,
        }

    fold_start_stuck = _current_fold_started_without_metrics(progress)
    if pid_alive is False:
        verdict = "likely_crashed"
        trust_status = "crashed_or_aborted"
        message = (
            "progress.json says running but training PID is gone "
            "(hard crash, kill, or OOM)."
        )
    elif is_stale and pid_alive is True:
        verdict = "stale"
        trust_status = "stalled"
        message = (
            f"No progress update for {stale_for:.0f} min (threshold {stale_minutes} min) "
            "but PID is still alive; may be stuck."
        )
    elif is_stale:
        verdict = "likely_crashed"
        trust_status = "crashed_or_aborted"
        message = (
            f"No progress update for {stale_for:.0f} min and PID unknown; "
            "treat as crashed/stopped, not running."
        )
    elif fold_start_stuck and stale_for is not None and stale_for >= 10:
        verdict = "likely_crashed"
        trust_status = "crashed_or_aborted"
        message = (
            "Fold started (epoch 0, no metrics) and has not advanced; "
            "likely failed at fold startup."
        )
    else:
        verdict = "running_ok"
        trust_status = "running"
        message = "Training appears active."

    return {
        "verdict": verdict,
        "trust_status": trust_status,
        "message": message,
        "pid": pid_int,
        "pid_alive": pid_alive,
        "minutes_since_update": stale_for,
        "is_stale": is_stale,
        "stale_minutes": stale_minutes,
    }


def format_progress_report(
    progress: dict[str, Any],
    experiment_dir: str | Path | None = None,
    *,
    include_health: bool = True,
    stale_minutes: int = DEFAULT_STALE_MINUTES,
) -> str:
    lines: list[str] = []
    if experiment_dir is not None:
        lines.append(f"Experiment: {Path(experiment_dir).resolve()}")

    status = progress.get("status", "unknown")
    cbm = progress.get("cbm") or "?"
    num_folds = int(progress.get("num_folds") or 0)
    num_epochs = progress.get("num_epochs")
    completed = list(progress.get("completed_folds") or [])
    current = progress.get("current")

    lines.append(f"Status: {status}")
    lines.append(f"GEM: {cbm}  |  folds: {len(completed)}/{num_folds}" + (
        f"  |  max epochs/fold: {num_epochs}" if num_epochs is not None else ""
    ))

    if progress.get("source") == "inferred_from_artifacts":
        lines.append("(Progress inferred from checkpoints/metrics; no live progress.json)")

    if completed:
        lines.append("Completed folds:")
        for row in completed:
            f1 = row.get("best_val_f1")
            f1_txt = f"  val F1={f1:.4f}" if isinstance(f1, (int, float)) else ""
            ep = row.get("epochs_run")
            ep_txt = f"  epochs={ep}" if ep is not None else ""
            lines.append(f"  - Fold {row['fold']}/{num_folds}{f1_txt}{ep_txt}")

    if current:
        fold = current.get("fold")
        epoch = current.get("epoch")
        lines.append(
            f"Current: fold {fold}/{num_folds}, epoch {epoch}/{num_epochs or '?'}"
        )
        batch = current.get("batch")
        total_batches = current.get("total_batches")
        if batch is not None and total_batches:
            frac = current.get("batch_frac")
            frac_txt = f" ({frac:.1%})" if isinstance(frac, (int, float)) else ""
            phase = current.get("phase") or "train"
            lines.append(f"  {phase} batch {batch}/{total_batches}{frac_txt}")
            loss_ema = current.get("batch_loss_ema")
            if isinstance(loss_ema, (int, float)):
                lines.append(f"  batch_loss_ema={loss_ema:.4f}")
        val_f1 = current.get("val_f1")
        best = current.get("best_val_f1")
        if val_f1 is not None and best is not None:
            lines.append(
                f"  val F1={val_f1:.4f}  best={best:.4f}  "
                f"no_improve={current.get('epochs_no_improve', 0)}"
            )

    mean_f1 = progress.get("mean_val_f1")
    if isinstance(mean_f1, (int, float)) and completed:
        lines.append(f"Mean val F1 (completed): {mean_f1:.4f}")

    updated = progress.get("updated_at")
    started = progress.get("started_at")
    if started:
        lines.append(f"Started: {started}")
    if updated:
        lines.append(f"Updated: {updated}")

    pid = progress.get("pid")
    host = progress.get("host")
    if pid is not None:
        host_txt = f" on {host}" if host else ""
        lines.append(f"Training PID: {pid}{host_txt}")

    if include_health:
        health = assess_training_health(progress, stale_minutes=stale_minutes)
        lines.append("")
        lines.append("Health check:")
        lines.append(f"  verdict: {health['verdict']}")
        lines.append(f"  trust_status: {health['trust_status']}")
        lines.append(f"  {health['message']}")
        if health["pid_alive"] is not None:
            lines.append(f"  pid_alive: {health['pid_alive']}")
        stale_for = health.get("minutes_since_update")
        if stale_for is not None:
            lines.append(
                f"  minutes_since_update: {stale_for:.1f} (stale after {stale_minutes})"
            )
        error = progress.get("error")
        if error:
            lines.append(f"  error: {error}")
        stop_reason = progress.get("stop_reason")
        if stop_reason:
            lines.append(f"  stop_reason: {stop_reason}")

    return "\n".join(lines)
