import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ck.pop("optimizer_states", None)
sd = ck["state_dict"]
for k, w in sd.items():
    if torch.is_floating_point(w):
        sd[k] = w.half().float()
torch.save(ck, sys.argv[2])
print("all weights rounded to FP16")