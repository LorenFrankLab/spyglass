"""List class attributes that tests patch (any receiver, any wrapping).

Run from the repo root: python .claude/docs/plans/v2-module-split/patch_inventory.py
"""
import ast, pathlib, collections
CLASSES={"CurationV2","CurationEvaluation","Sorting","SortingSelection","Recording","SortGroupV2","SorterParameters","QualityMetricParameters","AutoCurationRules"}
hits=collections.Counter(); where=collections.defaultdict(set)
for p in pathlib.Path("tests").rglob("*.py"):
    try: t=ast.parse(p.read_text())
    except Exception: continue
    for n in ast.walk(t):
        if not isinstance(n,ast.Call): continue
        f=n.func
        name=f.attr if isinstance(f,ast.Attribute) else (f.id if isinstance(f,ast.Name) else "")
        if name not in ("setattr","object"): continue  # monkeypatch.setattr / mp.setattr / setattr / patch.object
        if len(n.args)>=2 and isinstance(n.args[1],ast.Constant) and isinstance(n.args[1].value,str):
            tgt=ast.unparse(n.args[0]); attr=n.args[1].value
            for c in CLASSES:
                if tgt==c or tgt.endswith("."+c) or tgt.endswith(c+"()"):
                    hits[(c,attr)]+=1; where[(c,attr)].add(str(p).split("/")[-1])
        elif len(n.args)==1 and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and "." in n.args[0].value:
            parts=n.args[0].value.split(".")
            if len(parts)>=2 and parts[-2] in CLASSES:
                hits[(parts[-2],parts[-1])]+=1; where[(parts[-2],parts[-1])].add(str(p).split("/")[-1])
for (c,a),k in sorted(hits.items()):
    print(f"{c}.{a}: {k}  {sorted(where[(c,a)])}")
