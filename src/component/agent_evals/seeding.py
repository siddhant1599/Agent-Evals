"""The only place seeds are set.

``set_seed`` is called once, by ``run.py``, and nothing else in the package calls
``torch.manual_seed`` or ``random.seed``. That single call site is what makes "three seeds, mean
and standard deviation" mean anything: if seeding happened in several places, a run's reported
seed would not describe the run.

It returns a record of what it actually seeded, which goes into ``manifest.json``. If numpy or
torch were not installed, the manifest says so rather than implying a determinism that was never
established.
"""

from __future__ import annotations

import os
import random
from typing import Any

__all__ = ["set_seed", "make_sampling_rng", "MAX_SEED"]

#: Upper bound for a seed, the widest value accepted by every library we seed.
MAX_SEED = 2**32 - 1


def make_sampling_rng(seed: int) -> random.Random:
    """Create a private sampling stream without altering any global generation RNG.

    Args:
        seed: Dataset sampling seed, in [0, MAX_SEED].

    Returns:
        A new independent random generator, reset to the supplied seed on every call.

    Raises:
        ValueError: If the seed is not an integer in range.
    """
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}], got {seed!r}")
    return random.Random(seed)


def set_seed(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """Seed every random source in the process.

    Covers ``random``, ``numpy``, ``torch`` (CPU and CUDA), and the ``PYTHONHASHSEED``
    environment variable. The vLLM sampling seed is passed per request by the serving client
    rather than set globally, because vLLM generates in a separate process.

    Args:
        seed: The seed. Must be in [0, MAX_SEED].
        deterministic: Request bitwise-deterministic torch kernels. Off by default, pending a
            team decision before GRPO training starts. Some 4-bit QLoRA kernels have no
            deterministic implementation and will raise rather than run slowly, so this is
            reported in the return value rather than assumed to have worked.

    Returns:
        A record of what was seeded, for ``manifest.json``. Keys: ``seed``, ``python``,
        ``numpy``, ``torch``, ``torch_cuda``, ``deterministic``. Library entries are the library
        version if it was seeded, or None if it was not installed.

    Raises:
        ValueError: If ``seed`` is outside [0, MAX_SEED].
    """
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")

    record: dict[str, Any] = {
        "seed": seed,
        "python": True,
        "numpy": None,
        "torch": None,
        "torch_cuda": False,
        "deterministic": False,
    }

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        record["numpy"] = np.__version__

    try:
        import torch
    except ImportError:
        return record

    torch.manual_seed(seed)
    record["torch"] = torch.__version__
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        record["torch_cuda"] = True

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        record["deterministic"] = True

    return record
