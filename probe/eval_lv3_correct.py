"""Correct level-3 control: remove the API-description JSON from the input."""
from __future__ import annotations
import argparse, json, os, random, re
from . import llm
from .pilot11_apibank import SYSTEM, match, target_call
from .paths import MODELS, DATA, RUNS  # noqa: F401

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data", default=str(DATA / "apibank" / "lv3-api-train.json"))
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()
    items=json.load(open(args.data)); items=[x for x in items if "API-Request:" in x.get("output","")]
    by={}
    for x in items:
        m=re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)",x["output"])
        if m: by.setdefault(m.group(1),[]).append(x)
    test=[]
    for name,xs in by.items():
        if len(xs)<2: continue
        xs=xs[:]; random.Random(args.seed+len(name)).shuffle(xs)
        nt=max(1,int(round(len(xs)*args.test_frac))); test+=xs[:nt]
    random.Random(args.seed+1).shuffle(test)
    tok=llm.load_tokenizer(args.model); base=llm.load_base_model(args.model)
    from peft import PeftModel
    peft=PeftModel.from_pretrained(base,os.path.join(args.run_dir,"adapter")); peft.eval()
    def strip_api(inp):
        lines=inp.splitlines(); keep=[]
        for l in lines:
            if l.strip().startswith('{"apiCode"'):
                continue
            keep.append(l)
        return "\n".join(keep)
    rows=[]
    for x in test:
        target=target_call(x["output"])
        full=x["instruction"]+"\nInput: "+x["input"]+"\nAPI-Request:"
        no=x["instruction"]+"\nInput: "+strip_api(x["input"])+"\nAPI-Request:"
        r={"id":x.get("id")}
        with peft.disable_adapter():
            out=llm.generate_answer(peft,tok,full,SYSTEM,None,max_new_tokens=128); r["base_full"]=match(out,target)
            out=llm.generate_answer(peft,tok,no,SYSTEM,None,max_new_tokens=128); r["base_no"]=match(out,target)
        out=llm.generate_answer(peft,tok,full,SYSTEM,None,max_new_tokens=128); r["lora_full"]=match(out,target)
        out=llm.generate_answer(peft,tok,no,SYSTEM,None,max_new_tokens=128); r["lora_no"]=match(out,target)
        rows.append(r)
    res={c:sum(x[c] for x in rows)/len(rows) for c in ("base_full","base_no","lora_full","lora_no")}
    json.dump({"n":len(rows),"results":res,"rows":rows},open(os.path.join(args.run_dir,"lv3_correct.json"),"w"),indent=2)
    print(res)

if __name__=="__main__": main()
