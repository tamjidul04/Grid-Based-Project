"""POST 40 synthetic test cases to the live Render URL.
Captures both response and error details."""
import json
import time
import urllib.request

URL = "https://grid-based-project.onrender.com"

cases = json.load(open("data/synthetic/test_40_cases.json", encoding="utf-8"))["cases"]
n_ok_dir = 0
n_ok_full = 0
total_t = 0

for c in cases:
    inp = c["input"]
    expected = c.get("expected_directives") or []
    req = urllib.request.Request(
        URL + "/optimize-energy",
        data=json.dumps(inp).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        resp_text = urllib.request.urlopen(req, timeout=60).read()
        resp = json.loads(resp_text)
        dt = (time.time() - t0) * 1000
        total_t += dt
        directives = resp["directive_interpretation"]
        got_types = [d["directive_type"] for d in directives]
        exp_types = [d["directive_type"] for d in expected]
        types_match = got_types == exp_types

        adj_match = True
        for gd, ed in zip(directives, expected):
            if ed.get("applies", True) is False:
                continue
            if gd["directive_type"] != ed["directive_type"]:
                adj_match = False; break
            if not gd.get("structured_adjustment"):
                adj_match = False; break
            exp_adj = ed.get("structured_adjustment") or {}
            for k, v in exp_adj.items():
                if gd["structured_adjustment"].get(k) != v:
                    adj_match = False; break
            if not adj_match: break

        dir_match = types_match and adj_match
        n_ok_dir += int(dir_match)
        n_ok_full += int(dir_match)
        status = "OK" if dir_match else "FAIL"
        cost = resp["total_cost_bdt"]
        print(f"{c['id']:<14}{dt:>6.0f}ms {status} {got_types} cost={cost:.0f} {inp['operator_notes'][0][:55]}")
        if not dir_match:
            print(f"             expected: {[(d['directive_type'], d.get('structured_adjustment')) for d in expected]}")
            for gd in directives:
                print(f"             got:      {gd['directive_type']} -> {gd.get('structured_adjustment')}")
    except urllib.error.HTTPError as e:
        dt = (time.time() - t0) * 1000
        total_t += dt
        body = e.read().decode(errors='replace')
        # 422 means the LP found the constraints infeasible — that IS a
        # correct response from a well-formed optimizer (it refuses to lie).
        # We treat it as a separate "infeasible" bucket rather than counting
        # against directive-match.
        is_infeasible = e.code == 422 and "Infeasible" in body
        status = "INFEASIBLE" if is_infeasible else f"HTTP{e.code}"
        print(f"{c['id']:<14}{dt:>6.0f}ms {status}  {inp['operator_notes'][0][:55]}")
    except Exception as e:
        dt = (time.time() - t0) * 1000
        total_t += dt
        print(f"{c['id']:<14}{dt:>6.0f}ms ERROR {type(e).__name__}: {e}")

print("-" * 80)
print(f"Directive-match (full success): {n_ok_full}/{len(cases)}")
print(f"Total: {total_t:.0f}ms  (avg {total_t/len(cases):.0f}ms/case)")
print()
print("Legend: OK = directive + structured_adjustment match")
print("        INFEASIBLE = LP correctly refused (constraints can't be satisfied)")
print("        FAIL = directive type/values don't match expectation")
