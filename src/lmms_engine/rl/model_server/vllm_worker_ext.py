"""In-process vLLM worker extension that resolves ``reload_weights``.

Why this exists
---------------
``VLLMChatModelServer._reload_weights`` (``vllm.py``) issues
``self.llm.collective_rpc("reload_weights", kwargs={"weights_path": ...,
"is_checkpoint_format": ...})`` to push freshly-merged policy weights into every
running vLLM worker. Stock vLLM has **no** ``reload_weights`` method on
``Worker`` / ``WorkerWrapperBase`` (verified against vLLM v0.11.0
``vllm/worker/worker_base.py`` -- searching the file finds no such method), so
that RPC only resolves if a worker extension supplies it.

How it wires up
---------------
Passing ``worker_extension_cls`` to ``LLM(...)`` makes vLLM append this class to
the worker's base classes at init (``WorkerWrapperBase.init_worker`` in
``vllm/worker/worker_base.py``), so ``self`` inside these methods is the live
vLLM worker and ``collective_rpc("reload_weights", ...)`` resolves the method by
name via ``execute_method`` -> ``run_method`` -> ``getattr(worker, "reload_weights")``.
The qualname is resolved with ``rsplit(".", 1)``
(``vllm/utils/__init__.py::resolve_obj_by_qualname``, v0.11.0 line 2822), so the
string wired in ``VLLMChatModelServer.__init__`` uses **dot** separators, not a
colon. ``VLLMChatModelServer.__init__`` sets this class as the default
``worker_extension_cls`` (leaving any caller-supplied value untouched).

What it does
------------
Streams ``(name, tensor)`` pairs from the safetensors shards of a merged
Hugging Face checkpoint directory and hands them to the model's ``load_weights``
-- the same sink vLLM's own loader uses and the pattern the upstream RLHF
example uses (``examples/offline_inference/rlhf_utils.py``:
``self.model_runner.model.load_weights(...)``). Tensors are read on CPU; each
parameter's ``weight_loader`` copies them onto the correct device.

Config example (wired automatically -- not called directly)::

    LLM(model=..., worker_extension_cls=
        "lmms_engine.rl.model_server.vllm_worker_ext.WeightReloadWorkerExtension")

Scope: bring-up only. This performs a full in-place ``load_weights`` and does
not re-run ``process_weights_after_loading`` -- correct for an unquantized bf16
model, but a quantized / layout-transformed model would need that follow-up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator


class WeightReloadWorkerExtension:
    """Adds ``reload_weights`` to the in-process vLLM worker.

    Mixed into the worker via ``worker_extension_cls``; keep the public surface
    to a single method so it cannot shadow existing worker attributes (vLLM's
    ``init_worker`` rejects extensions that clash with the worker's own names).
    """

    def reload_weights(
        self,
        weights_path: str,
        is_checkpoint_format: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        """Reload model weights in-place from a Hugging Face checkpoint.

        Args:
            weights_path: A merged HF checkpoint directory (``config.json`` plus
                ``*.safetensors``) or a single ``*.safetensors`` file.
            is_checkpoint_format: Accepted so the ``collective_rpc`` call never
                trips the ``TypeError`` fallback in
                ``VLLMChatModelServer._reload_weights``. The actual file/dir
                shape is auto-detected, so this flag is not branched on.
            **_: Absorbs any extra ``reload_kwargs`` forwarded by the caller.

        Returns:
            A small summary dict for logging.
        """
        model = _resolve_worker_model(self)
        weight_files = _resolve_safetensors_files(weights_path)
        if not weight_files:
            raise FileNotFoundError(
                f"reload_weights found no *.safetensors under {weights_path!r}. Disk weight sync "
                "requires a merged HF checkpoint (rl_config.policy_sync_merge_to_hf=true)."
            )
        model.load_weights(_safetensors_weights_iterator(weight_files))
        return {
            "backend": "vllm_worker_ext",
            "reloaded_files": len(weight_files),
            "weights_path": str(weights_path),
        }


def _resolve_worker_model(worker: Any) -> Any:
    """Return the loaded ``nn.Module`` from a vLLM worker instance."""
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise RuntimeError(
            "reload_weights expected a vLLM worker exposing `model_runner`; "
            f"got {type(worker).__name__} without one."
        )
    get_model = getattr(model_runner, "get_model", None)
    if callable(get_model):
        return get_model()
    model = getattr(model_runner, "model", None)
    if model is None:
        raise RuntimeError("reload_weights could not resolve `model_runner.model` on the vLLM worker.")
    return model


def _resolve_safetensors_files(weights_path: str) -> list[Path]:
    resolved = Path(weights_path).expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"reload_weights checkpoint path does not exist: {resolved}")
    return sorted(resolved.glob("*.safetensors"))


def _safetensors_weights_iterator(weight_files: list[Path]) -> Iterator[tuple[str, Any]]:
    """Stream ``(name, cpu_tensor)`` pairs from safetensors shards.

    Mirrors vLLM's ``model_executor.model_loader.weight_utils.safetensors_weights_iterator``
    but depends only on the ``safetensors`` package so it is stable across vLLM
    versions. ``safe_open`` memory-maps each shard, so peak host memory stays at
    roughly one tensor.
    """
    from safetensors import safe_open

    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                yield name, reader.get_tensor(name)
