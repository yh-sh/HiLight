# check_leak.py
import json, sys, random
adj = {}
for line in open(sys.argv[1], "r", encoding="utf-8"):
    j = json.loads(line); adj[j["iid"]] = {b: w for b,w in j["neighbors"]}

leaks = 0; total = 0
for line in open(sys.argv[2], "r", encoding="utf-8"):  # train_last_label.jsonl
    j = json.loads(line)
    hist, label = j.get("history") or [], j.get("label")
    if not hist or not label: continue
    last = hist[-1]
    # if the label is a top neighbor of the last seen item in the *current* graph, likely leakage
    nb = adj.get(last, {})
    if label in nb: leaks += 1
    total += 1
print(f"label present as neighbor of last-history item in {leaks}/{total} users ({leaks/total:.1%})")