"""E5: SEAL-style self-generated SFT baseline on API-Bank.

Generate API calls with the base model + full RAG for training tasks; keep only
calls that match the ground truth; train a LoRA adapter on those self-generated
calls (ReSTEM-style supervised selection, no RL). Test RAG / adapter-only /
adapter+RAG. This tests whether ground-truth action labels are necessary.
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
    ap.add_argument("--data_dir", default=str(DATA / "apibank"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_root", default=str(RUNS))
    args=ap.parse_args()
    run_id="pilot20_selfsft_"+datetime.now().strftime("%Y%m%d_%H%M%S")
    rd=os.path.join(args.run_root,run_id); os.makedirs(rd,exist_ok=True)
    json.dump(vars(args),open(os.path.join(rd,"config.json"),"w"),indent=2)
    items=[]
    for f in ("level-1-api.json","level-2-api.json"):
        items+=json.load(open(os.path.join(args.data_dir,f)))
    items=[x for x in items if "API-Request:" in x.get("expected_output","")]
    by={}
    for x in items:
        m=re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)",x["expected_output"])
        if m: by.setdefault(m.group(1),[]).append(x)
    train,test=[],[]
    for name,xs in by.items():
        if len(xs)<2: continue
        xs=xs[:]; random.Random(args.seed+len(name)).shuffle(xs)
        nt=max(1,int(round(len(xs)*args.test_frac)))
        test+=xs[:nt]; train+=xs[nt:]
    random.Random(args.seed+1).shuffle(test)
    tok=llm.load_tokenizer(args.model); base=llm.load_base_model(args.model)
    examples=[]; generated=[]; correct=0
    for x in train:
        target=target_call(x["expected_output"])
        user=x["instruction"]+"\nInput: "+x["input"]+"\nAPI-Request:"
        out=llm.generate_answer(base,tok,user,SYSTEM,None,max_new_tokens=128)
        ok=match(out,target); correct+=int(ok)
        generated.append({"id":x["id"],"output":out,"target":target,"match":ok})
        if ok:
            examples.append({"question":user,"answer":out})
    print(f"train {len(train)}; self-generated correct {correct}/{len(train)}; self-SFT examples {len(examples)}")
    json.dump(generated,open(os.path.join(rd,"generated_train.json"),"w"),indent=2)
    del base; import torch; torch.cuda.empty_cache()
    ad=os.path.join(rd,"adapter")
    llm.train_lora_freeform(args.model,examples,ad,SYSTEM,epochs=args.epochs,lr=5e-5,batch_size=2,grad_accum=2,seed=args.seed)
    from peft import PeftModel
    base=llm.load_base_model(args.model); peft=PeftModel.from_pretrained(base,ad); peft.eval()
    def run(cond):
        rows=[]
        for x in test:
            target=target_call(x["expected_output"])
            full=x["instruction"]+"\nInput: "+x["input"]+"\nAPI-Request:"
            none=strip_descriptions(x["instruction"])+"\nInput: "+x["input"]+"\nAPI-Request:"
            if cond=="rag":
                with peft.disable_adapter(): out=llm.generate_answer(peft,tok,full,SYSTEM,None,max_new_tokens=128)
            elif cond=="lora":
                out=llm.generate_answer(peft,tok,none,SYSTEM,None,max_new_tokens=128)
            else:
                out=llm.generate_answer(peft,tok,full,SYSTEM,None,max_new_tokens=128)
            rows.append({"id":x["id"],"match":match(out,target)})
        return sum(r["match"] for r in rows)/len(rows)
    res={c:run(c) for c in ("rag","lora","hybrid")}
    summary={"run_id":run_id,"n_train":len(train),"self_correct":correct,"n_selfsft":len(examples),"n_test":len(test),"results":res,"finished":datetime.now().isoformat(timespec="seconds")}
    json.dump(summary,open(os.path.join(rd,"summary.json"),"w"),indent=2)
    print(res); print("RUN_DIR="+rd)

if __name__=="__main__": main()
