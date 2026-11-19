"""
Replication of Sec. V (Empirical validation)
Figs. 1a, 1b (Gaussian), 3a, 3b (Laplace)

Setup exactly as paper:
  * sensitivity profiles  lambda_i ∝ i (linear), i^2 (quadratic), e^i (exponential), uniform, one-hot
  * Gaussian: lambda normalised so ||lambda||_2 = 1;  Laplace: ||lambda||_1 = 1   (Fig. 4: ||lambda||_2 = 1 for both)
  * decoupled coordinates: Delta_p = ||lambda||_p  (Remark 1)
  * delta = 1e-6; Fig. 1a/3a: K = 20, eps in [0.2, 1.2]; Fig. 1b/3b/4: eps = 0.5, K = 1..25
  * metric = theoretical MSE of Table I, in dB

Formulas (Table I):
  Gaussian i.i.d.  MSE = K * Delta_2^2 / mu0^2          i.n.i.d.  MSE = ||lambda||_1^2 / mu0^2       (Thm 4)
  Laplace  i.i.d.  MSE = 2 K Delta_1^2 / eps^2          i.n.i.d.  MSE = 2 ||lambda^(2/3)||_1^3 / eps^2 (Thm 5)
  mu0 = largest mu with Q(eps/mu - mu/2) - e^eps Q(eps/mu + mu/2) <= delta   (problem P2)
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DELTA = 1e-6

# ------------------------------------------------------------------ calibration
def mu0(eps, delta=DELTA):
    f = lambda m: norm.sf(eps/m - m/2) - np.exp(eps)*norm.sf(eps/m + m/2) - delta
    return brentq(f, 1e-6, 100.0, xtol=1e-12)

# ------------------------------------------------------------------ profiles
def profile(name, K):
    i = np.arange(1, K + 1, dtype=float)
    return {"uniform": np.ones(K), "linear": i, "quadratic": i**2,
            "exponential": np.exp(i), "one-hot": np.eye(K)[0]}[name]

# ------------------------------------------------------------------ Table I MSEs
def gauss_mse(lam, eps):
    lam = lam / np.linalg.norm(lam)                       # ||lambda||_2 = 1
    m2 = mu0(eps)**2
    return len(lam) * 1.0 / m2, lam.sum()**2 / m2         # (iid, inid)

def laplace_mse(lam, eps, norm_l2=False):
    lam = lam / (np.linalg.norm(lam) if norm_l2 else lam.sum())
    d1 = lam.sum()
    return 2*len(lam)*d1**2/eps**2, 2*(lam**(2/3)).sum()**3/eps**2

def gauss_sigma_inid(lam, eps):  return np.sqrt(lam * lam.sum()) / mu0(eps)
def laplace_beta_inid(lam, eps): return lam**(1/3) * (lam**(2/3)).sum() / eps

dB = lambda x: 10*np.log10(x)
PROFILES = ["linear", "quadratic", "exponential", "one-hot"]
STYLE = {"linear": "o", "quadratic": "D", "exponential": "s", "one-hot": "^"}

def main():
    print("=== MSE reduction at K=20 (constant over eps)  [paper value in brackets] ===")
    paper_g = {"linear": 1.145, "quadratic": 2.442, "exponential": 9.658, "one-hot": 13.01}
    paper_l = {"linear": 0.546, "quadratic": 1.39,  "exponential": 7.609}
    for p in PROFILES:
        gi, gn = gauss_mse(profile(p, 20), 0.5); li, ln = laplace_mse(profile(p, 20), 0.5)
        print(f"{p:12s} Gaussian {dB(gi/gn):6.3f} dB [{paper_g[p]}]   "
              f"Laplace {dB(li/ln):6.3f} dB [{paper_l.get(p, '-')}]")

    gi25, _ = gauss_mse(profile("linear", 25), 0.5)
    _, ge = gauss_mse(profile("exponential", 60), 0.5)
    _, le = laplace_mse(profile("exponential", 60), 0.5)
    _, lo = laplace_mse(profile("one-hot", 60), 0.5)
    print(f"\nGaussian exp-profile saturation (eps=0.5): {dB(ge):.3f} dB  [paper 21.477]")
    print(f"Laplace  exp-profile saturation (eps=0.5): {dB(le):.3f} dB  [paper 14.243], "
          f"{dB(le)-dB(lo):.2f} dB above one-hot [paper 5.4]")
    for K in (20, 25):
        g = [gauss_mse(profile(p, K), 0.5)[1] for p in ("linear", "quadratic")]
        print(f"K={K}: i.n.i.d. Gaussian linear {dB(g[0]):.3f} dB, quadratic {dB(g[1]):.3f} dB")

    # Fig. 4 crossover: i.i.d. (uniform) Laplace vs Gaussian, ||lambda||_2 = 1
    cross = next(K for K in range(1, 26)
                 if laplace_mse(np.ones(K), 0.5, norm_l2=True)[0] > gauss_mse(np.ones(K), 0.5)[0])
    print(f"Fig. 4: uniform Laplace worse than Gaussian from K = {cross}  [paper K >= 9]")
    worse = [K for K in range(1, 26) if laplace_mse(profile("exponential", K), 0.5, norm_l2=True)[1]
             > gauss_mse(profile("exponential", K), 0.5)[1]]
    print(f"Fig. 4: K where exp-profile i.n.i.d. Laplace is worse than Gaussian: {worse or 'none'}  [paper: none]")

    print("\n=== Monte Carlo check (K=20, eps=0.5, exponential profile, 200k draws) ===")
    rng = np.random.default_rng(0)
    lam = profile("exponential", 20)
    lg = lam / np.linalg.norm(lam); s = gauss_sigma_inid(lg, 0.5)
    print(f"Gaussian  privacy constraint sum(l^2/s^2)={np.sum(lg**2/s**2):.6f} vs mu0^2={mu0(0.5)**2:.6f}")
    print(f"Gaussian  MSE empirical {(rng.normal(0, s, (200000, 20))**2).sum(1).mean():.4f} "
          f"vs theory {gauss_mse(lam, 0.5)[1]:.4f}")
    ll = lam / lam.sum(); b = laplace_beta_inid(ll, 0.5)
    print(f"Laplace   privacy constraint sum(l/b)={np.sum(ll/b):.6f} vs eps=0.5")
    print(f"Laplace   MSE empirical {(rng.laplace(0, b, (200000, 20))**2).sum(1).mean():.4f} "
          f"vs theory {laplace_mse(lam, 0.5)[1]:.4f}")

    # ---------------------------------------------------------- figures
    eps_grid = np.linspace(0.2, 1.2, 21); Ks = np.arange(1, 26)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for row, (fn, title) in enumerate([(gauss_mse, "Fig. 1"), (laplace_mse, "Fig. 3")]):
        a = ax[row, 0]
        a.plot(eps_grid, [dB(fn(profile("uniform", 20), e)[0]) for e in eps_grid], "-o", ms=4, label="i.i.d.")
        for p in PROFILES:
            a.plot(eps_grid, [dB(fn(profile(p, 20), e)[1]) for e in eps_grid], "--" + STYLE[p], ms=4,
                   label=f"i.n.i.d., {p}")
        a.set(xlabel=r"$\epsilon$", ylabel="MSE (dB)", title=f"{title}a — " + ("Gaussian" if row == 0 else "Laplace") + ", K = 20"); a.legend(fontsize=8)
        a = ax[row, 1]
        a.plot(Ks, [dB(fn(profile("uniform", K), 0.5)[0]) for K in Ks], "-o", ms=4, label="i.i.d.")
        for p in PROFILES:
            a.plot(Ks, [dB(fn(profile(p, K), 0.5)[1]) for K in Ks], "--" + STYLE[p], ms=4,
                   label=f"i.n.i.d., {p}")
        a.set(xlabel="K", ylabel="MSE (dB)", title=f"{title}b — " + ("Gaussian" if row == 0 else "Laplace") + r", $\epsilon$ = 0.5"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("sec5_fig1_fig3.png", dpi=150)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(Ks, [dB(gauss_mse(np.ones(K), .5)[0]) for K in Ks], "-x", label="Gaussian, uniform")
    plt.plot(Ks, [dB(laplace_mse(np.ones(K), .5, True)[0]) for K in Ks], "-o", label="Laplace, uniform")
    plt.plot(Ks, [dB(gauss_mse(profile("exponential", K), .5)[1]) for K in Ks], ":x",
             label=r"i.n.i.d. Gaussian, $\lambda_i \propto e^i$")
    plt.plot(Ks, [dB(laplace_mse(profile("exponential", K), .5, True)[1]) for K in Ks], ":o",
             label=r"i.n.i.d. Laplace, $\lambda_i \propto e^i$")
    plt.xlabel("K"); plt.ylabel("MSE (dB)"); plt.title(r"Fig. 4: $\epsilon$=0.5, $\delta=10^{-6}$, $\Delta_2$=1")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("sec5_fig4.png", dpi=150)
    print("\nSaved sec5_fig1_fig3.png and sec5_fig4.png")

if __name__ == "__main__":
    main()
