import json, random, datetime
def rand_date():
    d = datetime.date(2020,1,1) + datetime.timedelta(days=random.randint(0,2000))
    return d.isoformat()
rows=[]
for i in range(200):
    pol = f"ZX-{random.randint(10000,99999)}"
    eff = rand_date()
    rows.append({
      "instruction":"Extract policy and effective date. Return exactly one line: Policy=<ID>; Effective=<YYYY-MM-DD>",
      "input":f"Policy: {pol}\nEffective: {eff}",
      "output":f"Policy={pol}; Effective={eff}"
    })
open("train.jsonl","w").write("\n".join(json.dumps(r) for r in rows))

