import numpy as np

def boot(d,groups,seed):
    rng=np.random.default_rng(seed);u=sorted(set(groups));n=len(u);g=np.array(groups)
    sums=np.array([d[g==x].sum(0) for x in u]);counts=np.array([(g==x).sum() for x in u]);draws=[]
    for _ in range(100):
        w=rng.multinomial(n,np.full(n,1/n),size=1000);draws.append((w@sums)/(w@counts)[:,None])
    return np.quantile(np.concatenate(draws),[.025,.975],axis=0)
