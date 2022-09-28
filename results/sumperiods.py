import numpy as np

n_cpus = 4
for n in range(n_cpus):
    with open(f"periodsgrid_{n}.npy", "rb") as f:
        pg = np.load(f)
    if n==0:
        pg_total = np.zeros_like(pg)
    pgc = np.copy(pg)
    pgc[np.isnan(pgc)] = 0
    pg_total += pgc

pg_total[pg_total==0] = np.nan

with open("periodsgrid_total.npy", "wb") as g:
    np.save(g, pg_total)
