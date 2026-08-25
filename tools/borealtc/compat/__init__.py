"""Faithful-compatibility adapters (docs/STAGE0B2..., §3.3).

Adapters here may alter import/API glue only — never the numerical model,
data pipeline, loss, optimizer, scheduler behavior, split, or evaluation
semantics — and are applied from CryoLocoManip code, never by editing the
pinned upstream checkout.
"""
