import re, statistics as st
def read(path):
    log = open(path).read()
    parts = re.split(r"===== RESULT (\w+) version (\d+)", log)
    out = {}
    for i in range(1, len(parts), 3):
        m = re.search(r"test/IoU_Burn scar\D*?([0-9]\.[0-9]+)", parts[i+2])
        e = re.search(r"test/Boundary_mIoU\D*?([0-9]\.[0-9]+)", parts[i+2])
        if m and e:
            out.setdefault(parts[i], {})[parts[i+1]] = (float(m.group(1))*100, float(e.group(1))*100)
    return out
L = "/home/kolanchi/thesis/runs/logs/"
r = {}
for f in ["int8_test_2294.out", "lowbit_test_2295.out", "fp16_test_2296.out", "fp16w_test_2297.out", "lowbit300_2298.out"]:
    try: r.update(read(L + f))
    except FileNotFoundError: print("missing", f)
rows = [("FP32", "FP32", "600M FP32"), ("FP16_600M", "FP32", "600M FP16"),
        ("INT8W", "FP32", "600M INT8"), ("INT6W", "FP32", "600M INT6"),
        ("INT4W", "FP32", "600M INT4"), ("INT3W", "FP32", "600M INT3"),
        ("FP32_300M", "FP32_300M", "300M FP32"), ("FP16_300M", "FP32_300M", "300M FP16"), ("INT8W_300M", "FP32_300M", "300M INT8"), ("INT4W_300M", "FP32_300M", "300M INT4"), ("INT3W_300M", "FP32_300M", "300M INT3")]
print(f"{'model':>10} | {'n':>2} | {'IoU burn scar':>14} | {'loss':>13} | {'edge loss':>9}")
for name, base, label in rows:
    if name not in r or base not in r: continue
    vs = [v for v in r[name] if v in r[base]]
    iou = [r[name][v][0] for v in vs]
    loss = [r[base][v][0] - r[name][v][0] for v in vs]
    eloss = [r[base][v][1] - r[name][v][1] for v in vs]
    sd = lambda x: st.stdev(x) if len(x) > 1 else 0
    print(f"{label:>10} | {len(vs):>2} | {st.mean(iou):6.2f} ± {sd(iou):4.2f} | {st.mean(loss):5.2f} ± {sd(loss):4.2f} | {st.mean(eloss):9.2f}")