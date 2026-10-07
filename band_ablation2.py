# ======================================================================================
# SUPERSEDED (Oct 2026). Kept unchanged below as the record of the submitted SPMB 2026 paper.
# Issue: within-stage baseline uses sleep_pipeline_v2._crossstage (leaky) and the duplicate-
# identity bootstrap. Its band-stop embeddings (emb_ablation.pkl) are re-scored correctly by
# revision_2026-10.
# Corrected protocol and results: revision_2026-10/ (see README.md, section 'Correction').
# ======================================================================================
import glob,os,json,datetime,pickle,numpy as np
from scipy.signal import butter,sosfiltfilt
import importlib.util
def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}",flush=True)
def d(r,*p): return os.path.join(r,*p)
def load_epochs(r):
    c={}
    for fp in sorted(glob.glob(d(r,"epochs2","*.npz"))):
        z=np.load(fp,allow_pickle=True); c[os.path.basename(fp)[:6]]=(z["X"].astype("float32"),z["y"])
    return c
BANDS={"delta":(0.5,4),"theta":(4,8),"alpha":(8,12),"sigma":(12,16),"beta":(16,30)}
ROOT="./"; SEED=42; FUSE=1; MINE=10; SF=100.0; B=500
import torch,torch.nn as nn,torch.nn.functional as F
dev="cuda" if torch.cuda.is_available() else "cpu"; log(f"device={dev} fuse={FUSE} B={B}")
cache=load_epochs(ROOT); subs=sorted(cache); rng=np.random.default_rng(SEED); rng.shuffle(subs)
ntr=int(round(len(subs)*0.6)); train,ev=subs[:ntr],subs[ntr:]
assert set(train).isdisjoint(ev),"LEAKAGE"
Xtr=np.concatenate([cache[s][0] for s in train])
mu=Xtr.mean(axis=(0,2),keepdims=True); sd=Xtr.std(axis=(0,2),keepdims=True)+1e-6
def norm(X): return (X-mu)/sd
C=Xtr.shape[1]
class Enc(nn.Module):
    def __init__(s,C,emb):
        super().__init__()
        def blk(i,o,k=7,st=2): return nn.Sequential(nn.Conv1d(i,o,k,st,k//2),nn.BatchNorm1d(o),nn.ELU(),nn.Dropout(0.3))
        s.net=nn.Sequential(blk(C,32),blk(32,64),blk(64,128),blk(128,128),nn.AdaptiveAvgPool1d(1)); s.fc=nn.Linear(128,emb)
    def forward(s,x): z=s.net(x).squeeze(-1); return F.normalize(s.fc(z),dim=1)
enc=Enc(C,128).to(dev); enc.load_state_dict(torch.load(d(ROOT,"encoder.pt"),map_location=dev)); enc.eval()
sp=importlib.util.spec_from_file_location("v2",d(ROOT,"sleep_pipeline_v2.py")); v2=importlib.util.module_from_spec(sp); sp.loader.exec_module(v2)
def bandstop(X,lo,hi): return sosfiltfilt(butter(4,[lo,hi],btype="bandstop",fs=SF,output="sos"),X,axis=-1).astype("float32")
def embed_all(tf):
    ec={}
    with torch.no_grad():
        for s in ev:
            X,y=cache[s]; Xa=tf(X) if tf else X; Z=[]
            for i in range(0,len(Xa),512): Z.append(enc(torch.tensor(norm(Xa[i:i+512].astype("float32"))).to(dev)).cpu().numpy())
            ec[s]=(np.concatenate(Z),y)
    return ec
CACHE="emb_ablation.pkl"
if os.path.exists(CACHE): emb=pickle.load(open(CACHE,"rb")); log("loaded cached embeddings")
else:
    emb={"baseline":embed_all(None)}
    for bn,(lo,hi) in BANDS.items(): emb[bn]=embed_all(lambda X,b=(lo,hi):bandstop(X,*b)); log(f"embedded {bn}")
    pickle.dump(emb,open(CACHE,"wb"))
def wover(ec): return v2._means(v2._crossstage(ec,MINE,fuse=FUSE,rng=np.random.default_rng(SEED)))[0]
rb=np.random.default_rng(SEED); boot={bn:[] for bn in BANDS}
for b in range(B):
    bs=list(rb.choice(ev,len(ev),replace=True))
    sub=lambda nm:{f"{s}#{i}":emb[nm][s] for i,s in enumerate(bs)}
    bw=wover(sub("baseline"))
    for bn in BANDS: boot[bn].append(wover(sub(bn))-bw)
    if (b+1)%50==0: log(f"  bootstrap {b+1}/{B}")
res={}
for bn in BANDS:
    a=np.array(boot[bn]); res[bn]={"dEER_mean":round(float(a.mean()),3),"CI95":[round(float(np.percentile(a,2.5)),3),round(float(np.percentile(a,97.5)),3)],"robust":bool(np.percentile(a,2.5)>0)}
    log(f"{bn}: dEER={res[bn]['dEER_mean']} CI{res[bn]['CI95']} robust={res[bn]['robust']}")
json.dump({"fuse":FUSE,"baseline_within":round(float(wover(emb['baseline'])),3),"bands":res},open("band_ablation2.json","w"),indent=2)
log("DONE -> band_ablation2.json")
