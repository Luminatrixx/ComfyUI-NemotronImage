from contextlib import ExitStack, contextmanager
import logging

import torch

import comfy.model_prefetch as prefetch
from comfy.cli_args import args
from comfy.model_management import current_stream
from comfy.ops import vbar_above_watermark


class CapturedBlocks:
    def __init__(self, start, end, banks):
        self.start = start
        self.end = end
        self.banks = banks
        self.graph = None
        self.failed = False

    def forward(self, x, blocks, cache, capacity, options):
        if self.failed:
            return None
        with ExitStack() as pins:
            if any(pins.enter_context(bank.resident(x)) is None for bank in self.banks):
                self.failed = True
                return None
            if self.graph is None:
                self.input = torch.empty_like(x)
                graph = torch.cuda.CUDAGraph()
                graph_options = {**options, "nemotron_in_graph": True}
                try:
                    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                        out = self.input
                        for i in range(self.start, self.end):
                            out = blocks[i](out, cache, i, 0, capacity, graph_options)
                        self.output = out
                except RuntimeError as error:
                    if "captur" not in str(error).lower():
                        raise
                    logging.info("Nemotron KSampler: CUDA graph unavailable for layers %d-%d: %s", self.start, self.end - 1, error)
                    self.failed = True
                    return None
                self.graph = graph
            self.input.copy_(x)
            self.graph.replay()
            return self.output


class DecodeGraph:
    def __init__(self):
        self.groups = None
        self.warmed = False

    def plan(self, blocks, cache):
        groups = {}
        start = 0
        banks = []
        for i, block in enumerate(blocks):
            if block.kind == "M":
                continue
            if block.kind == "E" and cache[i][1] is not None and cache[i][1].pointers is not None:
                banks.append(cache[i][1])
                continue
            if start < i:
                groups[start] = CapturedBlocks(start, i, banks)
            start, banks = i + 1, []
        if start < len(blocks):
            groups[start] = CapturedBlocks(start, len(blocks), banks)
        return groups

    def forward(self, x, blocks, cache, position, capacity, options):
        planning = self.warmed and self.groups is None
        if planning:
            self.groups = self.plan(blocks, cache)
        index = 0
        while index < len(blocks):
            group = self.groups.get(index) if self.groups is not None else None
            out = group.forward(x, blocks, cache, capacity, options) if group is not None else None
            if out is not None:
                x = out
                index = group.end
            else:
                x = blocks[index](x, cache, index, position, capacity, options)
                index += 1
        self.warmed = True
        if planning:
            captured = [group for group in self.groups.values() if group.graph is not None]
            logging.info("Nemotron KSampler: CUDA graphs active for %d layers in %d segments",
                         sum(group.end - group.start for group in captured), len(captured))
        return x


@contextmanager
def prepare(model, device, options):
    if not options.get("nemotron_bank_bytes", 0):
        yield options
        return
    allocator = torch.cuda.memory.get_allocator_backend()
    # Triton's launcher queries graph-private pointers before cudaMallocAsync maps them.
    capture = not args.disable_cuda_graphs and allocator == "native"
    if not args.disable_cuda_graphs and not capture:
        logging.info("Nemotron KSampler: CUDA graph capture disabled with %s; using GPU-side routing", allocator)
    units = [model.backbone.embeddings, model.backbone.norm_f, model.lm_head, *model.units]
    modules = [module for unit in units for root in (unit if isinstance(unit, tuple) else (unit,))
               for module in root.modules() if hasattr(module, "_v")]
    if not modules or any(vbar_above_watermark(module) or getattr(module, "weight_lowvram_function", None) is not None
                          or getattr(module, "bias_lowvram_function", None) is not None
                          or module.weight_function or module.bias_function for module in modules):
        yield options
        return

    stream = None
    pinned = True
    try:
        stream, _ = prefetch.pin_modules(modules, device, model.dtype)
        if all(module._prefetch["signature"] is not None for module in modules):
            decode_options = {**options, "prefetch_dynamic_vbars": False}
            if capture:
                decode_options["nemotron_decode_graph"] = DecodeGraph()
            yield decode_options
        else:
            if stream is not None:
                stream.wait_stream(current_stream(device))
            prefetch.cleanup_prefetched_modules(model, modules)
            pinned = False
            yield options
    finally:
        if pinned:
            if stream is not None:
                stream.wait_stream(current_stream(device))
            prefetch.cleanup_prefetched_modules(model, modules)
