"""Apply a SEAL steering vector during decoding via forward hooks (no custom modeling,
version-safe). Adds coef*S to the layer-L residual stream at every `\\n\\n` token inside
the <think> block. BATCHED: steering state is per-sequence, so many prompts can be
generated at once (left-padded, so [:, -1] is the current token for every row).

Verified: out.hidden_states[L] == model.model.layers[L-1].output, so we hook layers[L-1].
SEAL vector = H_RT - H_E, so default coef = -1.0 (push toward execution).
"""
import torch


def _split_ids(tok):
    return {i for t, i in tok.get_vocab().items() if "ĊĊ" in t}   # tokens containing "\n\n"


class Steerer:
    def __init__(self, model, tok, vector, layer=20, coef=-1.0):   # SEAL vector = H_RT - H_E
        self.model, self.tok = model, tok
        self.vec = vector.float()
        self.layer, self.coef = layer, coef
        self.split = _split_ids(tok)
        self.tstart = tok.encode("<think>", add_special_tokens=False)[0]
        self.tend = tok.encode("</think>", add_special_tokens=False)[0]
        self.split_t = None                 # tensor of split ids (lazy, on device)
        self.in_think = None                # [B] bool
        self.inject = None                  # [B] bool
        self._handles = []

    def _embed_pre(self, module, args):
        ids = args[0]
        if ids is None or ids.ndim != 2:
            return
        dev = ids.device
        B = ids.shape[0]
        if self.split_t is None or self.split_t.device != dev:
            self.split_t = torch.tensor(sorted(self.split), device=dev)
        if ids.shape[1] > 1:                # prefill: set per-seq think-state, no inject
            in_think = torch.zeros(B, dtype=torch.bool, device=dev)
            for b in range(B):
                row = ids[b].tolist()
                if self.tstart in row:
                    i = row.index(self.tstart)
                    in_think[b] = self.tend not in row[i:]
            self.in_think = in_think
            self.inject = torch.zeros(B, dtype=torch.bool, device=dev)
        else:                               # decode: [B,1]
            toks = ids[:, -1]
            self.inject = torch.isin(toks, self.split_t) & self.in_think
            self.in_think = (self.in_think | (toks == self.tstart)) & (toks != self.tend)

    def _layer_hook(self, module, inp, out):
        if self.inject is not None and bool(self.inject.any()):
            hs = out[0] if isinstance(out, tuple) else out
            mask = self.inject.to(hs.device)
            hs[mask, -1, :] = hs[mask, -1, :] + self.coef * self.vec.to(hs.dtype).to(hs.device)
        return out

    def __enter__(self):
        self.in_think = None
        self.inject = None
        self._handles.append(self.model.model.embed_tokens.register_forward_pre_hook(self._embed_pre))
        self._handles.append(self.model.model.layers[self.layer - 1].register_forward_hook(self._layer_hook))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.inject = None
