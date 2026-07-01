"""LigerCE ops for aero_realtime_omni talker.

Replaces the materialized logits path in
``AeroRealtimeTalkerForConditionalGeneration.forward_sub_talker_finetune``
with per-group ``LigerFusedLinearCrossEntropyLoss`` calls, which saves the
``[B, 15, V]`` logits materialization (2048*15*B fp16/32 memory) and fuses
the linear + shift + CE into one kernel.

The shift semantics match the original HF ``ForCausalLMLoss`` path used by
``code_predictor.forward_finetune``: labels are right-padded with -100 and
sliced by one, so ``hidden_states[:, i]`` predicts ``codec_ids[:, i+2]``
(the last group's label is -100 and gets ignored).
"""

from typing import Tuple

import torch
import torch.nn.functional as F

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )

    _HAS_LIGER = True
except Exception:
    _HAS_LIGER = False


def lce_forward_sub_talker_finetune(
    self,
    codec_ids: torch.LongTensor,
    talker_hidden_states: torch.Tensor,
) -> Tuple[None, torch.Tensor]:
    """LCE replacement for ``forward_sub_talker_finetune``.

    Matches the original method's contract:
        codec_ids: [B, num_code_groups]  (== 16)
        talker_hidden_states: [B, hidden_size]

    Returns:
        (None, loss).  Original returns ``(logits, loss)``; we skip
        materializing the [B, 15, V] logits and return ``None`` in its slot.
        The caller in ``compute_talker_loss`` discards logits (``_, residual_loss = ...``).
    """
    assert len(codec_ids.shape) == 2
    assert len(talker_hidden_states.shape) == 2
    assert codec_ids.shape[0] == talker_hidden_states.shape[0]
    assert talker_hidden_states.shape[1] == self.config.hidden_size
    assert codec_ids.shape[1] == self.config.num_code_groups

    num_code_groups = self.config.num_code_groups

    sub_talker_inputs_embeds = [talker_hidden_states.unsqueeze(1)]
    for i in range(num_code_groups - 1):
        if i == 0:
            sub_talker_inputs_embeds.append(self.get_input_embeddings()(codec_ids[:, :1]))
        else:
            sub_talker_inputs_embeds.append(self.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, i : i + 1]))
    sub_talker_inputs_embeds = torch.cat(sub_talker_inputs_embeds, dim=1)

    inputs_embeds = self.code_predictor.small_to_mtp_projection(sub_talker_inputs_embeds)
    outputs = self.code_predictor.model(inputs_embeds=inputs_embeds)
    hidden_states = outputs.last_hidden_state  # [B, num_code_groups, D]

    # Reproduce HF ForCausalLMLoss shift: labels=codec_ids[:, 1:] gets padded to
    # [B, num_code_groups] with -100 and then sliced [:, 1:], so effectively
    # hidden_states[:, i] predicts codec_ids[:, i+2] for i in [1..14], and
    # hidden_states[:, 15]'s label is -100 (ignored).
    shift_labels = F.pad(codec_ids[:, 2:], (0, 1), value=-100)  # [B, 15]

    if _HAS_LIGER:
        lce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
        total = hidden_states.new_zeros(())
        n_valid = 0
        for i in range(1, num_code_groups):
            h_i = hidden_states[:, i, :].contiguous()
            lbl_i = shift_labels[:, i - 1].contiguous()
            total = total + lce(self.code_predictor.lm_head[i - 1].weight, h_i, lbl_i)
            n_valid += (lbl_i != -100).sum()
        loss = total / n_valid.clamp(min=1)
    else:
        logits = []
        for i in range(1, num_code_groups):
            logits.append(self.code_predictor.lm_head[i - 1](hidden_states[:, i]))
        logits = torch.stack(logits, dim=1)  # [B, 15, V]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

    return None, loss
