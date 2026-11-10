import re, statistics as st
log = open("/home/kolanchi/thesis/runs/logs/int8_test_2294.out").read()
parts = re.split(r"===== RESULT (FP32|INT8W) version (\d+)", log)
res = {}
for i in range(1, len(parts), 3):
    kind, v, text = parts[i], parts[i+1], parts[i+2]
    iou = re.search(r"test/IoU_Burn scar\D*?([0-9]\.[0-9]+)", text)
    miou = re.search(r"test/mIoU\D*?([0-9]\.[0-9]+)", text)
    bnd = re.search(r"test/Boundary_mIoU\D*?([0-9]\.[0-9]+)", text)
    if iou and miou and bnd:
        res.setdefault(v, {})[kind] = (float(iou.group(1))*100, float(miou.group(1))*100, float(bnd.group(1))*100)
print(f"{'model':>6} | {'FP32 IoU':>8} | {'INT8 IoU':>8} | {'loss':>6} || {'FP32 edge':>9} | {'INT8 edge':>9}")
d, f, q, db = [], [], [], []
for v, r in res.items():
    if "FP32" in r and "INT8W" in r:
        a, b = r["FP32"], r["INT8W"]
        d.append(a[0]-b[0]); f.append(a[0]); q.append(b[0]); db.append(a[2]-b[2])
        print(f"{v:>6} | {a[0]:8.2f} | {b[0]:8.2f} | {a[0]-b[0]:6.2f} || {a[2]:9.2f} | {b[2]:9.2f}")
print(f"\nFP32 IoU  : {st.mean(f):.2f} ± {st.stdev(f):.2f}")
print(f"INT8 IoU  : {st.mean(q):.2f} ± {st.stdev(q):.2f}")
print(f"INT8 loss : {st.mean(d):.2f} ± {st.stdev(d):.2f} IoU points")
print(f"Edge loss : {st.mean(db):.2f} ± {st.stdev(db):.2f} edge-IoU points")