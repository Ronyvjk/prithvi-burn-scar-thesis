import sys, torch
src, dst = sys.argv[1], sys.argv[2]
bits = int(sys.argv[3]) if len(sys.argv) > 3 else 8
qmax = 2 ** (bits - 1) - 1
ck = torch.load(src, map_location="cpu", weights_only=False)
ck.pop("optimizer_states", None)
sd = ck["state_dict"]
store, n_total, n_q = {}, 0, 0
for k, w in sd.items():
    if not torch.is_floating_point(w):
        continue
    n_total += w.numel()
    if w.ndim == 2 and w.numel() >= 4096 and k.endswith(".weight"):
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
        q = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)
        sd[k] = (q.float() * scale).to(w.dtype)
        store[k] = (q, scale.squeeze(1).half())
        n_q += w.numel()
torch.save(ck, dst)
torch.save(store, dst.replace(".ckpt", "_int_only.pt"))
size_mb = (n_q * bits / 8 + (n_total - n_q) * 2) / 1e6
print(f"bits={bits}: squeezed {n_q/1e6:.1f} M of {n_total/1e6:.1f} M numbers ({100*n_q/n_total:.1f} %)")
print(f"estimated model size (rest kept in FP16): {size_mb:.0f} MB")