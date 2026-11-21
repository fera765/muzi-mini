"""
muzi-mini / nano — treino REAL do zero de um LM recorrente (constante em
memoria por token, mesma tese do nosso runtime SSM), em numpy puro.
- backward/BPTT analitico + checagem numerica de gradiente (prova de corretude)
- split treino/validacao (nao treina no teste — GR)
- mede QUALIDADE real: perplexidade e acuracia de proximo-caractere na validacao,
  subindo de aleatorio -> treinado. Sem download, sem framework.
Safe p/ python3 -I.
"""
import numpy as np

# ----------------------- corpus real (estruturado, aprendivel) -----------------
NOMES = ["ana","bruno","carla","diego","elena","felipe","gustavo","helena",
         "igor","julia","karina","lucas","mateus","natalia","otavio","paula"]
CIDADES = ["ferraz","suzano","mogi","poa","itaquera","maua","santos","osasco"]
def make_corpus():
    rng = np.random.default_rng(7)
    linhas = []
    for n in NOMES:
        for c in CIDADES:
            linhas.append(f"{n} mora em {c}.")
    rng.shuffle(linhas)
    return linhas

# ----------------------- tokenizer char-level -----------------
class Vocab:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c:i for i,c in enumerate(chars)}
        self.itos = {i:c for c,i in self.stoi.items()}
        self.size = len(chars)
    def enc(self, s): return [self.stoi[c] for c in s]

# ----------------------- modelo RNN (tanh) + BPTT -----------------
class RNNLM:
    def __init__(self, V, H, seed=0):
        r = np.random.default_rng(seed); s = 1.0/np.sqrt(H)
        self.Wxh = (r.standard_normal((H,V))*s).astype(np.float64)
        self.Whh = (r.standard_normal((H,H))*s).astype(np.float64)
        self.Why = (r.standard_normal((V,H))*s).astype(np.float64)
        self.bh  = np.zeros(H); self.by = np.zeros(V)
        self.H, self.V = H, V
    def params(self): return [self.Wxh,self.Whh,self.Why,self.bh,self.by]
    def forward_loss(self, xs, ys, h0=None, cache=False):
        H=self.H; h=np.zeros(H) if h0 is None else h0
        hs={-1:h}; ps={}; loss=0.0; xsl=[]
        for t,(x,y) in enumerate(zip(xs,ys)):
            h=np.tanh(self.Wxh[:,x]+self.Whh@hs[t-1]+self.bh)
            yv=self.Why@h+self.by
            yv-=yv.max(); e=np.exp(yv); p=e/e.sum()
            loss-=np.log(p[y]+1e-12); hs[t]=h; ps[t]=p; xsl.append(x)
        if cache: self._c=(xs,ys,hs,ps)
        return loss/len(xs)
    def backward(self):
        xs,ys,hs,ps=self._c; T=len(xs)
        dWxh=np.zeros_like(self.Wxh); dWhh=np.zeros_like(self.Whh)
        dWhy=np.zeros_like(self.Why); dbh=np.zeros_like(self.bh); dby=np.zeros_like(self.by)
        dhn=np.zeros(self.H)
        for t in reversed(range(T)):
            dy=ps[t].copy(); dy[ys[t]]-=1.0; dy/=T
            dWhy+=np.outer(dy,hs[t]); dby+=dy
            dh=self.Why.T@dy+dhn
            dr=(1-hs[t]**2)*dh
            dbh+=dr; dWxh[:,xs[t]]+=dr; dWhh+=np.outer(dr,hs[t-1])
            dhn=self.Whh.T@dr
        return [dWxh,dWhh,dWhy,dbh,dby]

# ----------------------- checagem numerica de gradiente -----------------
def grad_check(model, xs, ys):
    model.forward_loss(xs,ys,cache=True); an=model.backward()
    worst=0.0
    for pi,(P,G) in enumerate(zip(model.params(),an)):
        it=np.ndindex(P.shape);
        for _ in range(6):
            idx=tuple(np.random.randint(0,s) for s in P.shape)
            old=P[idx]; eps=1e-5
            P[idx]=old+eps; lp=model.forward_loss(xs,ys)
            P[idx]=old-eps; lm=model.forward_loss(xs,ys); P[idx]=old
            num=(lp-lm)/(2*eps); rel=abs(num-G[idx])/(abs(num)+abs(G[idx])+1e-12)
            worst=max(worst,rel)
    return worst

# ----------------------- Adam -----------------
class Adam:
    def __init__(self,params,lr=2e-2):
        self.p=params; self.lr=lr
        self.m=[np.zeros_like(x) for x in params]; self.v=[np.zeros_like(x) for x in params]; self.t=0
    def step(self,grads,clip=5.0):
        self.t+=1
        gn=np.sqrt(sum(float((g*g).sum()) for g in grads))
        if gn>clip:
            grads=[g*(clip/gn) for g in grads]
        for i,(g) in enumerate(grads):
            self.m[i]=0.9*self.m[i]+0.1*g; self.v[i]=0.999*self.v[i]+0.001*g*g
            mh=self.m[i]/(1-0.9**self.t); vh=self.v[i]/(1-0.999**self.t)
            self.p[i]-=self.lr*mh/(np.sqrt(vh)+1e-8)

# ----------------------- avaliacao (val) -----------------
def eval_model(model, lines, vocab):
    tot=0; correct=0; nll=0.0; nchar=0
    for ln in lines:
        ids=vocab.enc(ln+"\n"); xs,ys=ids[:-1],ids[1:]
        h=np.zeros(model.H)
        for x,y in zip(xs,ys):
            h=np.tanh(model.Wxh[:,x]+model.Whh@h+model.bh)
            yv=model.Why@h+model.by; yv-=yv.max(); e=np.exp(yv); p=e/e.sum()
            nll-=np.log(p[y]+1e-12); nchar+=1
            if int(np.argmax(p))==y: correct+=1
            tot+=1
    return correct/tot, np.exp(nll/nchar)

if __name__=="__main__":
    lines=make_corpus()
    text="\n".join(lines)+"\n"; vocab=Vocab(text)
    rng=np.random.default_rng(1); idx=rng.permutation(len(lines))
    split=int(0.85*len(lines))
    train=[lines[i] for i in idx[:split]]; val=[lines[i] for i in idx[split:]]
    print(f"corpus: {len(lines)} frases | treino {len(train)} | val {len(val)} | vocab {vocab.size} chars")

    model=RNNLM(vocab.size, H=96, seed=0)
    # 1) prova de corretude do treino
    s=train[0]; ids=vocab.enc(s+"\n")
    err=grad_check(model, ids[:-1], ids[1:])
    print(f"checagem de gradiente (erro rel. max): {err:.2e}  -> {'OK (backward correto)' if err<1e-4 else 'FALHA'}")

    # 2) qualidade ANTES do treino (aleatorio)
    a0,pp0=eval_model(model,val,vocab)
    print(f"\nANTES  do treino | val acc(proximo char): {a0*100:5.1f}%  | perplexidade {pp0:6.1f}")

    # 3) treino real — com clipping e early stopping (guarda melhor checkpoint)
    opt=Adam(model.params(), lr=1e-2)
    best_acc=-1; best=None; patience=0
    for epoch in range(1,41):
        rng.shuffle(train); L=0.0
        for s in train:
            ids=vocab.enc(s+"\n")
            L+=model.forward_loss(ids[:-1],ids[1:],cache=True)
            opt.step(model.backward())
        a,pp=eval_model(model,val,vocab)
        if a>best_acc:
            best_acc=a; best=[p.copy() for p in model.params()]; patience=0
        else:
            patience+=1
        if epoch%5==0:
            print(f"epoca {epoch:2d} | loss treino {L/len(train):.3f} | val acc {a*100:5.1f}% | ppl {pp:6.1f}"
                  + ("  <- melhor" if a>=best_acc else ""))
        if patience>=8:
            print(f"early stopping na epoca {epoch} (sem melhora ha 8 epocas)"); break
    # restaura melhor checkpoint
    for p,b in zip(model.params(),best): p[:]=b

    # 4) qualidade DEPOIS + amostra gerada
    a1,pp1=eval_model(model,val,vocab)
    print(f"\nDEPOIS do treino | val acc(proximo char): {a1*100:5.1f}%  | perplexidade {pp1:6.1f}")
    print(f"ganho de qualidade medido: {a0*100:.1f}% -> {a1*100:.1f}% (val, dados nao vistos)")
    # gera a partir de 'mateus'
    h=np.zeros(model.H); out="mateus"
    for x in vocab.enc(out): h=np.tanh(model.Wxh[:,x]+model.Whh@h+model.bh)
    for _ in range(25):
        yv=model.Why@h+model.by; nx=int(np.argmax(yv)); out+=vocab.itos[nx]
        if vocab.itos[nx]=="\n": break
        h=np.tanh(model.Wxh[:,nx]+model.Whh@h+model.bh)
    print(f"amostra gerada: {out!r}")
