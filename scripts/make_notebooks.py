import sys
from pathlib import Path
import nbformat as nbf

OUT = Path(sys.argv[1])

SETUP = '''# Setup: works both locally (inside the cloned repository) and on Google Colab.
from pathlib import Path
import subprocess

REPO_URL = "https://github.com/albgirelli/gonzo-viewer"
if Path("../data/regions.csv").exists():
    DATA = Path("../data")
else:
    if not Path("gonzo-viewer").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL], check=True)
    DATA = Path("gonzo-viewer/data")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})'''


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    return book


def badge(name):
    return (f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/albgirelli/gonzo-viewer/blob/main/notebooks/{name})")


explore = nb([
    f"""md:# 1 · Exploring the Gonzo tracer data

{badge("01_explore_data.ipynb")}

A tracer (gadobutrol, 0.25 mmol) was injected into the spinal CSF of a healthy volunteer.
MRI was acquired before injection and about 4, 24, 48 and 70 h after. This notebook shows
what is in the tables and volumes. Everything here takes seconds to run.""",
    SETUP,
    "md:## Sessions and blood samples",
    """sessions = pd.read_csv(DATA / "sessions.csv")
blood = pd.read_csv(DATA / "blood_plasma.csv")
sessions""",
    "md:## Regions\nOne row per region: FreeSurfer brain regions, CSF next to each region, and some **summary regions** (whole brain, all CSF, lobes, …). `conc_ses-XX` is the mean concentration (mmol/L) and `amount_umol_ses-XX` the amount of tracer (µmol) in the region.",
    """regions = pd.read_csv(DATA / "regions.csv")
summary = regions[regions.kind == "aggregate"].set_index("name")
summary.filter(regex="volume|conc_").round(3)""",
    """hours = sessions.hours_after_injection.values
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name in ["Cerebral cortex", "Cerebral white matter", "Deep grey matter", "Cerebellum", "Brainstem"]:
    ax[0].plot(hours, summary.loc[name].filter(like="conc_").values, "o-", label=name)
for name in ["CSF: subarachnoid space over the cerebrum", "CSF: around cerebellum & brainstem", "CSF: ventricles"]:
    ax[1].plot(hours, summary.loc[name].filter(like="conc_").values, "o-", label=name.replace("CSF: ", ""))
ax[0].set_title("Brain tissue"); ax[1].set_title("CSF")
for a in ax:
    a.set_xlabel("hours after injection"); a.set_ylabel("mmol/L"); a.legend(fontsize=8)
plt.tight_layout()""",
    "md:## Where is the tracer? A mass balance\n0.25 mmol = 250 µmol was injected. How much of it is inside the skull at each time?",
    """parts = ["All brain tissue", "All intracranial CSF", "Other intracranial (dura, vessels, unlabelled)"]
amounts = summary.loc[parts].filter(like="amount_umol").T
amounts.index = hours
ax = amounts.plot.area(figsize=(7, 4), alpha=0.7)
ax.axhline(250, ls="--", c="k"); ax.text(1, 255, "injected: 250 µmol"); ax.set_ylim(0, 275)
ax.set_xlabel("hours after injection"); ax.set_ylabel("µmol")
ax2 = ax.twinx()
ax2.plot(blood.hours_after_injection, blood.plasma_concentration_mmol_per_L * 1000, "rs--", label="blood plasma")
ax2.set_ylabel("plasma concentration (µmol/L)", color="r");""",
    "md:## How deep does it go?\nMean concentration against distance from the brain surface (negative: CSF outside the brain; positive: inside tissue).",
    """depth = pd.read_csv(DATA / "depth_profiles.csv")
for h, g in depth[depth.hours_after_injection > 0].groupby("hours_after_injection"):
    plt.plot(g.depth_mm, g.mean_concentration_mmol_per_L, label=f"{h:.0f} h")
plt.axvline(0, c="k", lw=0.5); plt.xlabel("distance from brain surface (mm)"); plt.ylabel("mmol/L"); plt.legend();""",
    "md:## 3D volumes\n`volumes/gonzo_2mm.npz` (and a 1 mm version) contains the concentration at every time (NaN outside the head), FreeSurfer labels, a T1-weighted anatomical image and DTI mean diffusivity. The `affine` maps voxel indices to scanner coordinates in mm. Axes are (right, anterior, superior).",
    """vol = np.load(DATA / "volumes/gonzo_2mm.npz")
print({k: vol[k].shape for k in vol.files})
c = vol["concentration"].astype(float)
z = c.shape[3] // 2 + 5
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for k, a in enumerate(ax):
    a.imshow(vol["t1w"][:, :, z].T, cmap="gray", origin="lower")
    im = a.imshow(np.ma.masked_less(c[k + 1, :, :, z].T, 0.01), cmap="inferno", vmin=0, vmax=0.25, origin="lower", alpha=0.85)
    a.set_title(f"{hours[k + 1]:.0f} h"); a.axis("off")
fig.colorbar(im, ax=ax, shrink=0.8, label="mmol/L");""",
    """md:## Where next?
* Fit a simple model to the regional curves: see `02_compartment_model.ipynb`.
* Is diffusion enough to explain the depth profiles? See `03_diffusion_depth.ipynb`.
* Simulate on the 3D brain mesh: see `04_mesh_simulation.ipynb`.
* Or do something else entirely: clustering of regional curves, a statistical model of the peak time, …""",
])

compartment = nb([
    f"""md:# 2 · A compartment model

{badge("02_compartment_model.ipynb")}

The simplest possible model: treat each part of the system as a well-mixed box and let tracer move
between boxes at constant rates.

```
 spinal CSF  --k1-->  intracranial CSF  --k3-->  brain
                            |           <--k4--
                            k2
                            v
                         blood / out
```

$$S' = -k_1 S,\\quad C' = k_1 S - (k_2 + k_3) C + k_4 B,\\quad B' = k_3 C - k_4 B,\\qquad S(0) = 250\\,\\mu\\text{{mol}}.$$

We fit the four rates to the measured amounts of tracer in the intracranial CSF and in the brain.""",
    SETUP,
    """from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

regions = pd.read_csv(DATA / "regions.csv").set_index("name")
hours = pd.read_csv(DATA / "sessions.csv").hours_after_injection.values
csf = regions.loc["All intracranial CSF"].filter(like="amount_umol").values.astype(float)
brain = regions.loc["All brain tissue"].filter(like="amount_umol").values.astype(float)
print("CSF  (µmol):", csf.round(1)); print("brain(µmol):", brain.round(1))""",
    """def model(t, y, k1, k2, k3, k4):
    S, C, B = y
    return [-k1 * S, k1 * S - (k2 + k3) * C + k4 * B, k3 * C - k4 * B]

def simulate(k, t_eval):
    sol = solve_ivp(model, (0, t_eval[-1]), [250, 0, 0], t_eval=t_eval, args=tuple(k), rtol=1e-8)
    return sol.y

def residuals(logk):
    y = simulate(np.exp(logk), hours)
    return np.concatenate([y[1] - csf, y[2] - brain])

fit = least_squares(residuals, np.log([0.05, 0.05, 0.05, 0.05]))
k = np.exp(fit.x)
for name, v in zip(["k1 spinal→cranial", "k2 CSF→out", "k3 CSF→brain", "k4 brain→CSF"], k):
    print(f"{name:20s} {v:.4f} /h   (half-life {np.log(2) / v:5.1f} h)")""",
    """t = np.linspace(0, 80, 400)
y = simulate(k, t)
plt.plot(t, y[1], "C0", label="CSF (model)"); plt.plot(hours, csf, "C0o", label="CSF (data)")
plt.plot(t, y[2], "C1", label="brain (model)"); plt.plot(hours, brain, "C1o", label="brain (data)")
plt.plot(t, y[0], "C2--", label="spinal CSF (model, not measured)")
plt.xlabel("hours after injection"); plt.ylabel("µmol"); plt.legend();""",
    """md:## Questions to push on
* **Identifiability.** 4 parameters, 8 data points. Are the rates well determined? Try a profile likelihood, bootstrap, or a Bayesian fit (e.g. `emcee`).
* **Use the blood data** (`blood_plasma.csv`): add a blood compartment with renal clearance. Does it constrain the model?
* **More boxes.** Split the brain into cortex / white matter / deep grey matter (see `regions.csv`). Who feeds whom?
* **Predict.** Leave the 48 h scan out, fit, and predict it. How good is the prediction?
* **What if?** Halve the clearance rate (impaired clearance, e.g. ageing) or change the dose: what changes, and when?""",
])

diffusion = nb([
    f"""md:# 3 · Is diffusion enough?

{badge("03_diffusion_depth.ipynb")}

If the tracer only *diffuses* into the brain from the CSF around it, the concentration at depth $x$
below the surface should follow
$$\\partial_t c = D\\,\\partial_{{xx}} c,\\qquad c(0,t) = \\alpha\\, c_{{\\text{{CSF}}}}(t),$$
where $c_{{\\text{{CSF}}}}(t)$ is measured just outside the brain and $\\alpha$ accounts for the fact that
the tracer can only access the extracellular space (roughly 20% of tissue).

Molecular diffusion of gadobutrol in brain tissue is expected to be about
$D \\approx 1.2\\times10^{{-4}}$ mm²/s (free diffusion $3.8\\times10^{{-4}}$ mm²/s slowed by tortuosity).
Let us fit $D$ and $\\alpha$ to the measured depth profiles.""",
    SETUP,
    """from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

depth = pd.read_csv(DATA / "depth_profiles.csv")
P = depth.pivot(index="depth_mm", columns="hours_after_injection", values="mean_concentration_mmol_per_L")
hours = P.columns.values
c_csf = P.loc[-1].values            # CSF concentration 1 mm outside the brain
obs_depth = np.arange(1, 16)
obs = P.loc[obs_depth].values       # tissue concentration at 1..15 mm depth
x = np.arange(0, 40.01, 0.5)        # computational grid (mm), no-flux at 40 mm""",
    """def simulate(D, alpha, k=0.0):
    \"\"\"Diffusion (+ optional first-order clearance k, 1/h) into a half-space; D in mm²/s.\"\"\"
    dx = x[1] - x[0]
    bc = lambda t: alpha * np.interp(t, hours, c_csf)
    def rhs(t, c):
        full = np.concatenate([[bc(t)], c, [c[-1]]])
        return D * 3600 * (full[2:] - 2 * full[1:-1] + full[:-2]) / dx**2 - k * c
    sol = solve_ivp(rhs, (0, hours[-1]), np.zeros(len(x) - 1), t_eval=hours, method="BDF")
    c = np.vstack([[bc(t) for t in hours], sol.y])
    return np.array([np.interp(obs_depth, x, c[:, j]) for j in range(len(hours))]).T

def fit(fixed_D=None):
    def res(p):
        D = fixed_D if fixed_D else np.exp(p[1])
        return (simulate(D, np.exp(p[0])) - obs)[:, 1:].ravel()
    p0 = [np.log(0.5)] + ([] if fixed_D else [np.log(1.2e-4)])
    r = least_squares(res, p0)
    return np.exp(r.x), r.cost

(alpha_m, ), cost_m = fit(fixed_D=1.2e-4)
(alpha_f, D_f), cost_f = fit()
print(f"molecular D = 1.2e-4 mm²/s : alpha = {alpha_m:.2f}, misfit = {cost_m:.2e}")
print(f"fitted    D = {D_f:.1e} mm²/s : alpha = {alpha_f:.2f}, misfit = {cost_f:.2e}  ({D_f / 1.2e-4:.1f}× molecular)")""",
    """fig, ax = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
pred_m, pred_f = simulate(1.2e-4, alpha_m), simulate(D_f, alpha_f)
for j, a in enumerate(ax):
    a.plot(obs_depth, obs[:, j + 1], "ko", ms=4, label="data")
    a.plot(obs_depth, pred_m[:, j + 1], "C0", label="D = 1.2e-4 (molecular)")
    a.plot(obs_depth, pred_f[:, j + 1], "C3", label=f"D = {D_f:.1e} (fitted)")
    a.set_title(f"{hours[j + 1]:.0f} h"); a.set_xlabel("depth (mm)")
ax[0].set_ylabel("mmol/L"); ax[0].legend(fontsize=8);""",
    """md:## Discuss and extend
* The fitted $D$ is several times larger than molecular diffusion. Is that evidence of **flow** (convection, glymphatic transport), or of something the 1D model ignores: brain folding and curvature, partial volume at the surface, tracer arriving from the ventricles, white matter vs grey matter, …?
* Add **clearance** (`k > 0`) or a **velocity** term $-v\\,\\partial_x c$. Which model does the data prefer? Use AIC/BIC, cross-validation, or Bayesian model comparison.
* Repeat the analysis **region by region** (from the 3D volumes). Does the apparent $D$ differ between regions?
* How **uncertain** is $D$? What is the smallest flow velocity this data could detect?""",
])

mesh = nb([
    f"""md:# 4 · Simulating tracer transport on the brain mesh

{badge("04_mesh_simulation.ipynb")}

The tetrahedral mesh (≈100 000 vertices) covers the brain. Each vertex carries the measured tracer
concentration at each time, and the **boundary concentration** (the concentration in the CSF next to
the brain surface). This lets us run a *data-driven* simulation: impose the measured CSF concentration
on the surface, solve a PDE inside, and compare with the measured interior.

Here: pure diffusion $\\partial_t c = D\\,\\Delta c$ with P1 finite elements (scikit-fem) and backward Euler.
Runs in about 30 s.""",
    "!pip install -q scikit-fem 2>/dev/null || true",
    SETUP,
    """import scipy.sparse.linalg as spla
from skfem import MeshTet, Basis, ElementTetP1, asm
from skfem.models.poisson import laplace, mass

m = np.load(DATA / "mesh/gonzo_mesh.npz")
print({k: m[k].shape for k in m.files})
mesh = MeshTet(np.ascontiguousarray(m["points"].T, dtype=float), np.ascontiguousarray(m["tetrahedra"].T))
basis = Basis(mesh, ElementTetP1())
K, M = asm(laplace, basis), asm(mass, basis)
boundary = mesh.boundary_nodes()
interior = np.setdiff1d(np.arange(mesh.nvertices), boundary)
hours, c_data, c_bdry = m["hours"], m["concentration"], m["boundary_concentration"]
# subdomains (per cell): 1 cortical grey matter, 2 white matter, 3 subcortical grey matter""",
    """def simulate(D, alpha=1.0, dt=0.5):
    \"\"\"D in mm²/s; boundary value alpha * measured CSF concentration. Returns solution at the scan times.\"\"\"
    A = (M + dt * D * 3600 * K).tocsr()
    lu = spla.splu(A[interior][:, interior].tocsc())
    A_ib = A[interior][:, boundary]
    c, out = np.zeros(mesh.nvertices), {0: np.zeros(mesh.nvertices)}
    for t in np.arange(dt, hours[-1] + dt / 2, dt):
        # linear interpolation in time between scans (hat-function weights)
        w = np.maximum(0, 1 - np.abs(np.interp(t, hours, np.arange(len(hours))) - np.arange(len(hours))))
        cb = alpha * (w @ c_bdry[:, boundary])
        c[interior] = lu.solve((M @ c)[interior] - A_ib @ cb)
        c[boundary] = cb
        for k, h in enumerate(hours):
            if abs(t - h) < dt / 2:
                out[k] = c.copy()
    return np.array([out.get(k, c) for k in range(len(hours))])

sim = simulate(D=1.2e-4)""",
    """def amount(c):
    \"\"\"Total tracer in the brain in µmol (concentration in mmol/L × volume in mL).\"\"\"
    return (M @ c).sum() / 1000

plt.plot(hours, [amount(c) for c in c_data], "ko-", label="data")
plt.plot(hours, [amount(c) for c in sim], "C0s-", label="diffusion, alpha = 1")
plt.xlabel("hours after injection"); plt.ylabel("tracer in brain (µmol)"); plt.legend();""",
    """md:The simulation puts far too much tracer into the brain. **Why?** Think about what the boundary
condition means: the CSF concentration is a concentration *in fluid*, while an MRI voxel in tissue reports
tracer per *tissue* volume, and only the extracellular space (≈ 20%) is accessible.

Try `simulate(D, alpha)` with other values, compare region by region, and look at the solution in ParaView:""",
    """import meshio
cells = [("tetra", m["tetrahedra"])]
meshio.write("simulation.vtu", meshio.Mesh(m["points"], cells, point_data={
    **{f"data_{h:.0f}h": c_data[k] for k, h in enumerate(hours)},
    **{f"sim_{h:.0f}h": sim[k] for k, h in enumerate(hours)},
}, cell_data={"subdomain": [m["subdomains"]]}))
print("wrote simulation.vtu (open in ParaView)")""",
    """md:## Ideas
* **Estimate** $D$ and $\\alpha$ (and maybe a clearance rate) from the data: an inverse problem with a PDE constraint.
* Use the **diffusion tensor** (`m["diffusion_tensor"]`, per cell) to make diffusion anisotropic. Does it matter?
* Different parameters in **grey and white matter** (`m["subdomains"]`).
* Add a **velocity field** (e.g. from a Darcy model) and see whether advection improves the fit.
* Mesh refinement, time step, and other numerical questions: how accurate is the simulation itself?""",
])

for name, book in [("01_explore_data.ipynb", explore), ("02_compartment_model.ipynb", compartment),
                   ("03_diffusion_depth.ipynb", diffusion), ("04_mesh_simulation.ipynb", mesh)]:
    nbf.write(book, OUT / name)
    print("wrote", name)
