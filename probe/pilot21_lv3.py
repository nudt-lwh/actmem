"""E6: API-Bank level-3 API-request generalization.

Train on 80% of the 907 level-3 API-request training examples, test on 20%.
These are harder, multi-step planning-style requests. Compare RAG, LoRA-only,
and hybrid.
"""
from __future__ import annotations
import argparse, json, os, random, re
from datetime import datetime
from . import llm
from .pilot11_apibank import SYSTEM, match, strip_descriptions, target_call
from .paths import MODELS, DATA, RUNS  # noqa: F401

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--data", default=str(DATA / "apibank" / "lv3-api-train.json"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args=ap.parse_args()
    run_id="pilot21_lv3_"+datetime.now().strftime("%Y%m%d_%H%M%S")
    rd=os.path.join(args.run_root,run_id); os.makedirs(rd,exist_ok=True)
    json.dump(vars(args),open(os.path.join(rd,"config.json"),"w"),indent=2)
    items=json.load(open(args.data))
    items=[x for x in items if "API-Request:" in x.get("output","")]
    by={}
    for x in items:
        m=re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)",x["output"])
        if m: by.setdefault(m.group(1),[]).append(x)
    train,test=[],[]
    for name,xs in by.items():
        if len(xs)<2: continue
        xs=xs[:]; random.Random(args.seed+len(name)).shuffle(xs)
        nt=max(1,int(round(len(xs)*args.test_frac)))
        test+=xs[:nt]; train+=xs[nt:]
    random.Random(args.seed+1).shuffle(test)
    print("train",len(train),"test",len(test),"apis",len(by))
    examples=[]
    for x in train:
        user=x["instruction"]+"\nInput: "+x["input"]+"\nAPI-Request:"
        examples.append({"question":user,"answer":target_call(x["output"])})
    ad=os.path.join(rd,"adapter")
    llm.train_lora_freeform(args.model,examples,ad,SYSTEM,epochs=args.epochs,lr=5e-5,batch_size=2,grad_accum=2,seed=args.seed)
    from peft import PeftModel
    base=llm.load_base_model(args.model); peft=PeftModel.from_pretrained(base,ad); peft.eval()
    tok=llm.load_tokenizer(args.model)
    def run(cond):
        rows=[]
        for x in test:
            target=target_call(x["output"])
            full=x["instruction"]+"\nInput: "+x["input"]+"\nAPI-Request:"
            none=strip_descriptions(x["instruction"])+"\nInput: "+x["input"]+"\nAPI-Request:"
            if cond=="rag":
                with peft.disable_adapter(): out=llm.generate_answer(peft,tok,full,SYSTEM,None,max_new_tokens=128)
            elif cond=="lora":
                out=llm.generate_answer(peft,tok,none,SYSTEM,None,max_new_tokens=128)
            else:
                out=llm.generate_answer(peft,tok,full,SYSTEM,None,max_new_tokens=128)
            rows.append({"id":x.get("id"),"match":match(out,target)})
        return sum(r["match"] for r in rows)/len(rows)
    res={c:run(c) for c in ("rag","lora","hybrid")}
    summary={"run_id":run_id,"n_train":len(train),"n_test":len(test),"results":res,"finished":datetime.now().isoformat(timespec="seconds")}
    json.dump(summary,open(os.path.join(rd,"summary.json"),"w"),indent=2)
    print(res); print("RUN_DIR="+rd)

if __name__=="__main__": main()
