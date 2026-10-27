"""Matched SE-LLM endpoint generation around a user-supplied frozen model."""

from contextlib import contextmanager, nullcontext
from importlib import import_module

import torch

from .train import seed_everything


@contextmanager
def alignment_condition(model, condition):
    """Null: zero fuser.attention output before decomposition and fusion.

    History, text embeddings, noise/decomposition and fusion remain in place.
    Hook removal is guaranteed, including when an inference call fails.
    """
    if condition not in {"valid", "null"}:
        raise ValueError("condition must be valid or null")
    handle = None
    if condition == "null":
        if not hasattr(model, "fuser") or not hasattr(model.fuser, "attention"):
            raise TypeError("the model needs the SE-LLM fuser.attention module")
        def zero_alignment(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("alignment output must be a tensor")
            return torch.zeros_like(output)
        handle = model.fuser.attention.register_forward_hook(zero_alignment)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def load_backbone(factory_name, checkpoint_path, dataset, horizon, device):
    """A local factory loads the upstream SE-LLM architecture and user's weights.

    The factory signature is (checkpoint_path, dataset, horizon, device).
    No credential, private location, checkpoint, or architecture config is
    bundled. It returns a torch module with the upstream SE-LLM call signature.
    """
    module_name, separator, function_name = factory_name.partition(":")
    if not separator:
        raise ValueError("factory must use the module:function form")
    factory = getattr(import_module(module_name), function_name)
    model = factory(checkpoint_path=checkpoint_path, dataset=dataset,
                    horizon=horizon, device=device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("factory must return a torch.nn.Module")
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def rollout(model, history, horizon, token_length=96):
    """Same recursive 96-point rollout for both endpoint conditions.

    This adapter targets the archived Qwen-based SE-LLM, whose temporal mark
    arguments are unused. A backbone that requires real time markers needs
    its matching upstream rollout rather than dummy marker substitution.
    """
    chunks = []
    rolling = history
    for _ in range((horizon + token_length - 1)//token_length):
        if chunks:
            rolling = torch.cat([rolling[:, token_length:], chunks[-1]], dim=1)
        context = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if history.device.type == "cuda" else nullcontext())
        with context:
            output = model(rolling, rolling, None, rolling)
        if output.ndim != 3 or output.shape[1] < token_length:
            raise ValueError("backbone must return [B,T,C] with at least one output token")
        chunks.append(output[:, -token_length:].float())
    return torch.cat(chunks, dim=1)[:, :horizon]


def endpoint_pass(model, batches, condition, protocol, device):
    # Reset once per complete endpoint pass, not independently per sample.
    seed_everything(protocol.seed)
    model.eval()
    with alignment_condition(model, condition), torch.inference_mode():
        for batch in batches:
            history = batch["history"].to(device=device, dtype=torch.float32)
            yield rollout(model, history, protocol.horizon, protocol.token_length).cpu()
