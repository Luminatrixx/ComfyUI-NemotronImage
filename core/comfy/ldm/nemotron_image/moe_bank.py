from contextlib import contextmanager

import torch

from comfy_aimdo.model_vbar import vbar_fault, vbar_signature_compare, vbar_unpin
from comfy.model_management import current_stream
from comfy.model_prefetch import cleanup_prefetched_modules, pin_modules


class ExpertBank:
    """Execution-local tables for an AIMDO resident set of routed expert weights."""

    def __init__(self, experts, byte_limit):
        self.experts = experts
        self.allocation = getattr(experts, "_v_block", None)
        self.signature = None
        self.pointers = None
        if self.allocation is not None:
            vbar, address, size = self.allocation
            if address - vbar.base_addr + size > min(byte_limit, vbar.get_watermark() * (32 << 20)):
                self.allocation = None

    def populate(self, x):
        projections = [projection for expert in self.experts for projection in (expert.up_proj, expert.down_proj)]
        stream = None
        try:
            stream, _ = pin_modules(projections, x.device, x.dtype)
            if any(projection._prefetch["signature"] is None for projection in projections):
                return None
            tables = []
            for group in (projections[::2], projections[1::2]):
                weights = [projection._v_weight for projection in group]
                tables.append(tuple(torch.tensor(addresses, device=x.device, dtype=torch.int64) for addresses in (
                    [weight._qdata.data_ptr() for weight in weights],
                    [weight._params.scale.data_ptr() for weight in weights],
                )))
            return tuple(tables)
        finally:
            if stream is not None:
                stream.wait_stream(current_stream(x.device))
            cleanup_prefetched_modules(self.experts, projections)

    @contextmanager
    def resident(self, x):
        allocation = self.allocation
        if allocation is None:
            yield None
            return
        signature = vbar_fault(allocation)
        if signature is None:
            self.allocation = None
            yield None
            return
        try:
            if self.pointers is None:
                self.pointers = self.populate(x)
                self.signature = signature
            elif not vbar_signature_compare(signature, self.signature):
                self.pointers = None
            if self.pointers is None:
                self.allocation = None
            yield self.pointers
        finally:
            vbar_unpin(allocation)
