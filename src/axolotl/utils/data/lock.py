"""Logic for loading / preparing a dataset once over all processes."""

import time
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.utils.dict import DictDefault

LOCK_FILE_NAME = "datasets_prep.lock"
READY_FILE_NAME = "datasets_ready.flag"
PROCESS_COUNTER_FILE_NAME = "process_counter.txt"
# Persistent completion marker. Presence means the prepared dataset at
# `dataset_prepared_path` is fully built, so every rank may `load_from_disk` concurrently
# WITHOUT the advisory FileLock. Unlike READY_FILE_NAME it is never removed by cleanup().
# Invariant: a `dataset_prepared_path` holds one dataset+config (its hash) -- point
# distinct configs at distinct prepared paths, or delete this marker after changing the
# dataset config, so a stale marker never short-circuits a needed rebuild.
COMPLETE_FILE_NAME = ".axolotl_prepared_complete"


class FileLockLoader:
    """
    Simple class for abstracting single process data loading / processing. The first
    process that creates a lock file does the work; the remaining procesees simply load
    the preprocessed dataset once the first process is done.

    Lock-free fast path: if a prior (e.g. single-process `axolotl preprocess`) build
    persisted COMPLETE_FILE_NAME, every rank loads the prepared dataset lock-free. This
    avoids the shared-FS advisory FileLock, which returns ENOLCK ("No locks available") /
    ESTALE ("Stale file handle") on networked filesystems (Lustre/NFS) under many-rank
    concurrency. axolotl loads datasets BEFORE the process group is initialized, so it
    cannot coordinate with a torch.distributed barrier (as LlamaFactory's
    `main_process_first` does) and falls back to this filesystem lock.
    """

    def __init__(self, cfg: DictDefault):
        self.cfg = cfg
        self.dataset_prepared_path = (
            cfg.dataset_prepared_path or DEFAULT_DATASET_PREPARED_PATH
        )
        self.lock_file_path = Path(self.dataset_prepared_path) / LOCK_FILE_NAME
        self.ready_flag_path = Path(self.dataset_prepared_path) / READY_FILE_NAME
        self.counter_path = Path(self.dataset_prepared_path) / PROCESS_COUNTER_FILE_NAME
        self.complete_flag_path = Path(self.dataset_prepared_path) / COMPLETE_FILE_NAME

    def load(self, load_fn: Callable[[], Any]) -> Any:
        # Fast path: a prior build persisted the completion marker -> the prepared dataset
        # is on disk; all ranks load it concurrently with no shared-FS locking.
        if self.complete_flag_path.exists():
            return load_fn()

        with FileLock(str(self.lock_file_path)):
            self._increment_counter()

            if not self.ready_flag_path.exists():
                result = load_fn()
                self.ready_flag_path.touch()
                self.complete_flag_path.touch()
                return result

            while not self.ready_flag_path.exists():
                time.sleep(1)
            return load_fn()

    def _increment_counter(self):
        """Safely increment the process counter."""
        if self.counter_path.exists():
            counter_content = self.counter_path.read_text().strip()
            count = int(counter_content) if counter_content else 0
        else:
            count = 0
        self.counter_path.write_text(str(count + 1))

    def cleanup(self):
        """Clean up ready flag when last process is done."""
        # Fast-path ranks never took the lock or incremented the counter; taking the lock
        # here would re-introduce the many-rank shared-FS contention the fast path avoids.
        if self.complete_flag_path.exists():
            return
        try:
            with FileLock(str(self.lock_file_path)):
                counter_content = self.counter_path.read_text().strip()
                count = int(counter_content) if counter_content else 0
                count -= 1

                if count <= 0:
                    # Last process cleans everything up
                    self.ready_flag_path.unlink(missing_ok=True)
                    self.counter_path.unlink(missing_ok=True)
                else:
                    # Still have active processes
                    self.counter_path.write_text(str(count))
        except FileNotFoundError:
            # Lock file might have already been deleted by another process
            pass
