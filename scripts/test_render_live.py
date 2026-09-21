"""POST all 10 sample cases to the live Render URL."""
import json
import time
import urllib.request

URL = "https://grid-based-project.onrender.com"

data = json.load(open("data/raw/sample_cases.json", encoding="utf-8"))
print(f"Testing POST against {URL}/optimize-energy")
print(f"{'CASE':<12}{'LATENCY':>10}  {'COST':>10}  {'REF':>10}  MATCH")
print("-" * 60)
n_ok = 0
total_t = 0
for c in data["cases"]:
    inp = c["input"]
    ref_cost = c["expected_output"]["total_cost_bdt"]
    ref_types = [d["directive_type"] for d in c["expected_output"]["directive_interpretation"]]
    req = urllib.request.Request(
        URL + "/optimize-energy",
        data=json.dumps(inp).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        dt = (time.time() - t0) * 1000
        total_t += dt
        cost = resp["total_cost_bdt"]
        got_types = [d["directive_type"] for d in resp["directive_interpretation"]]
        ok = got_types == ref_types and abs(cost - ref_cost) < 0.01
        n_ok += int(ok)
        flag = "OK" if ok else "FAIL"
        print(f"{c['id']:<12}{dt:>8.0f}ms  {cost:>10.2f}  {ref_cost:>10.2f}  {flag}")
        if not ok:
            print(f"             expected: {ref_types}")
            print(f"             got     : {got_types}")
    except Exception as e:
        print(f"{c['id']:<12}ERROR: {type(e).__name__}: {e}")
print("-" * 60)
print(f"{n_ok}/{len(data['cases'])} cases match reference exactly")
print(f"Total: {total_t:.0f}ms  (avg {total_t/len(data['cases']):.0f}ms/case)")
