"""Build the lightweight subset of the Gonzo dataset used by the website and notebooks.

Input: the unzipped Zenodo record 10.5281/zenodo.14266867 (mri-processed.zip, mesh-data.zip and
mri_dataset/{timetable.tsv,blood_concentrations.csv} from mri-dataset.zip).

    python scripts/build_subset.py /path/to/gonzo_raw

Outputs (relative to the repository root):
    data/*.csv                     tables (regions, time series, depth profiles, blood, sessions)
    data/volumes/gonzo_{1,2}mm.npz downsampled 4D concentration + anatomy + labels + DTI (tensor, MD, FA)
    data/mesh/gonzo_mesh.npz       tetrahedral brain mesh with concentrations (FEM-ready)
    docs/data/*                    the same volumes in the compact format read by the web viewer
"""

import gzip
import json
import sys
from pathlib import Path

import meshio
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

import labels as L

REPO = Path(__file__).resolve().parents[1]
SESSIONS = ["ses-01", "ses-02", "ses-03", "ses-04", "ses-05"]
SESSION_LABELS = ["Baseline", "4 h", "24 h", "48 h", "70 h"]
WEB_MM = 1  # grid spacing used by the browser viewer


def nanblock(a, b):
    """Mean over b×b×b blocks ignoring NaN; all-NaN blocks stay NaN."""
    s = a.reshape(a.shape[0] // b, b, a.shape[1] // b, b, a.shape[2] // b, b)
    valid = np.isfinite(s)
    total = np.where(valid, s, 0).sum(axis=(1, 3, 5), dtype=np.float64)
    count = valid.sum(axis=(1, 3, 5))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total / count).astype(np.float32)


def blockmode(a, b):
    """Most frequent value in each b×b×b block."""
    n = [d // b for d in a.shape]
    s = (
        a.reshape(n[0], b, n[1], b, n[2], b)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(-1, b**3)
    )
    s = np.sort(s, axis=1)
    pos = np.arange(s.shape[1])
    start = np.ones_like(s, dtype=bool)
    start[:, 1:] = s[:, 1:] != s[:, :-1]
    first = np.maximum.accumulate(np.where(start, pos, 0), axis=1)
    j = np.argmax(pos - first, axis=1)
    return s[np.arange(len(s)), j].reshape(n)


def pooled_mean(means, counts, ks):
    """Mean over the union of regions ks, from per-region means and voxel counts (regions without data are skipped)."""
    m, n = np.asarray(means)[ks], np.asarray(counts)[ks]
    ok = np.isfinite(m) & (n > 0)
    return float((m[ok] * n[ok]).sum() / n[ok].sum()) if ok.any() else np.nan


def tensor_eig(comp, chunk=2_000_000):
    """Fractional anisotropy and principal eigenvector of DTI tensors.

    comp: six arrays (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz), NaN where there is no tensor.
    Returns fa (same shape, NaN where missing) and v1 (shape + (3,), unit vectors, NaN where missing).
    """
    ok = np.logical_and.reduce([np.isfinite(c) for c in comp])
    flat = np.stack([c[ok] for c in comp], axis=1).astype(np.float64)
    fa = np.full(ok.shape, np.nan, np.float32)
    v1 = np.full(ok.shape + (3,), np.nan, np.float32)
    fa_ok = np.empty(len(flat), np.float32)
    v1_ok = np.empty((len(flat), 3), np.float32)
    for a in range(0, len(flat), chunk):
        xx, xy, xz, yy, yz, zz = flat[a : a + chunk].T
        T = np.stack(
            [
                np.stack([xx, xy, xz], -1),
                np.stack([xy, yy, yz], -1),
                np.stack([xz, yz, zz], -1),
            ],
            -2,
        )
        w, v = np.linalg.eigh(T)  # ascending eigenvalues
        w = np.clip(w, 0, None)
        m = w.mean(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            fa_ok[a : a + chunk] = np.sqrt(1.5 * ((w - m) ** 2).sum(1) / (w**2).sum(1))
        v1_ok[a : a + chunk] = v[:, :, 2]
    fa[ok] = fa_ok
    v1[ok] = v1_ok
    return fa, v1


def main(raw):
    raw = Path(raw)
    sub = raw / "mri_processed_data/sub-01"
    load = lambda p: nib.load(sub / p)

    # --- timing -------------------------------------------------------------------------
    tt = pd.read_csv(raw / "mri_dataset/timetable.tsv", sep="\t")
    ll = (
        tt[tt.sequence_label == "looklocker"]
        .set_index("session")
        .acquisition_relative_injection
    )
    scan_hours = [ll[s] / 3600 for s in SESSIONS]
    model_hours = [0.0] + scan_hours[
        1:
    ]  # baseline concentration is zero by construction
    sessions = pd.DataFrame(
        {
            "session": SESSIONS,
            "label": SESSION_LABELS,
            "hours_after_injection": np.round(model_hours, 2),
            "scan_time_hours": np.round(scan_hours, 2),
        }
    )

    blood = pd.read_csv(raw / "mri_dataset/blood_concentrations.csv")
    blood = pd.DataFrame(
        {
            "hours_after_injection": blood["Timedelta (min)"] / 60,
            "plasma_concentration_mmol_per_L": blood[
                "Concentration (mmol Gd per liter fluid)"
            ].values,
        }
    )
    # The ~24 h (1515 min) and ~40 h (2430 min) sample values are swapped in the source file.
    i24, i40 = (
        blood.index[blood.hours_after_injection.round(2) == 25.25][0],
        blood.index[blood.hours_after_injection == 40.5][0],
    )
    col = "plasma_concentration_mmol_per_L"
    blood.loc[[i24, i40], col] = blood.loc[[i40, i24], col].values

    # --- segmentation: one label volume covering the whole intracranial space -------------
    ref = load("segmentations/sub-01_seg-intracranial_binary.nii.gz")
    ic = np.asarray(ref.dataobj) > 0
    lo = np.argwhere(ic).min(0)
    hi = np.argwhere(ic).max(0) + 1
    size = (
        (hi - lo + 3) // 4
    ) * 4  # divisible by 4 so both 1 mm and 2 mm grids are exact
    hi = lo + size
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    affine0 = ref.affine
    ic = ic[crop]

    seg = np.asarray(
        load("segmentations/sub-01_seg-aparc+aseg_refined.nii.gz").dataobj
    )[crop].astype(np.int32)
    csfnear = np.asarray(
        load("segmentations/sub-01_seg-csf-aparc+aseg.nii.gz").dataobj
    )[crop].astype(np.int32)
    csf = (
        np.asarray(load("segmentations/sub-01_seg-csf_binary.nii.gz").dataobj)[crop] > 0
    )
    lab = np.zeros(seg.shape, np.int32)
    lab[ic] = L.OTHER_INTRACRANIAL
    tissue = (seg > 0) & ~csf & ~np.isin(seg, list(L.VENTRICLES))
    lab[tissue] = seg[tissue]
    lab[csf] = L.CSF_OFFSET + csfnear[csf]
    del seg, csfnear

    # --- DTI: diffusion tensor, mean diffusivity, fractional anisotropy (brain tissue only) --
    dti = load("dwi/sub-01_ses-01_dDTI_cleaned.nii.gz")
    tensor = []  # Dxx, Dxy, Dxz, Dyy, Dyz, Dzz in mm²/s
    for k in range(6):
        d = np.asarray(dti.dataobj[crop + (0, k)], dtype=np.float32)
        d[~tissue] = np.nan
        tensor.append(d)
    md = (tensor[0] + tensor[3] + tensor[5]) / 3
    fa, _ = tensor_eig(tensor)

    t1 = np.asarray(load("registered/sub-01_ses-01_T1w_registered.nii.gz").dataobj)[
        crop
    ].astype(np.float32)

    # --- regional statistics at full resolution ------------------------------------------
    ids, inv = np.unique(lab, return_inverse=True)
    inv = inv.reshape(lab.shape)
    voxel_ml = abs(np.linalg.det(affine0[:3, :3])) / 1000
    volume_ml = np.bincount(inv.ravel(), minlength=len(ids)) * voxel_ml

    def region_means(field):
        f = field.ravel()
        ok = np.isfinite(f)
        s = np.bincount(inv.ravel()[ok], weights=f[ok], minlength=len(ids))
        n = np.bincount(inv.ravel()[ok], minlength=len(ids))
        with np.errstate(invalid="ignore", divide="ignore"):
            return s / n, n

    md_mean, md_n = region_means(md)
    fa_mean, fa_n = region_means(fa)
    del fa
    conc_means, conc_full = [], []
    for ses in SESSIONS:
        c = np.asarray(
            load(f"concentrations/sub-01_{ses}_concentration.nii.gz").dataobj,
            dtype=np.float32,
        )[crop]
        c[~ic] = np.nan
        mean, _ = region_means(c)
        conc_means.append(mean)
        conc_full.append(c)

    # Aggregate regions (whole brain, lobes, CSF compartments) defined by sets of labels.
    groups = {lab_id: L.group(lab_id) for lab_id in ids if lab_id}
    tissue_ids = [i for i in ids if 0 < i < L.OTHER_INTRACRANIAL]
    aggregates = {
        20001: ("All brain tissue", tissue_ids),
        20002: ("Cerebral cortex", [i for i in tissue_ids if 1000 <= i < 3000]),
        **{
            20003 + k: (
                f"Cortex: {lobe}",
                [i for i in tissue_ids if groups[i] == f"Cortex: {lobe}"],
            )
            for k, lobe in enumerate(L.LOBES)
        },
        20010: ("Cerebral white matter", [i for i in tissue_ids if i in L.WHITE]),
        20011: ("Deep grey matter", [i for i in tissue_ids if i in L.DEEP_GM]),
        20012: ("Cerebellum", [i for i in tissue_ids if i in L.CEREBELLUM]),
        20013: ("Brainstem", [i for i in tissue_ids if i in L.BRAINSTEM]),
        20020: ("All intracranial CSF", [i for i in ids if i >= L.CSF_OFFSET]),
        20021: (
            "CSF: ventricles",
            [i for i in ids if groups.get(i) == "CSF: ventricles"],
        ),
        20022: (
            "CSF: subarachnoid space over the cerebrum",
            [i for i in ids if str(groups.get(i, "")).startswith("CSF: next to")],
        ),
        20023: (
            "CSF: around cerebellum & brainstem",
            [i for i in ids if groups.get(i) == "CSF: around cerebellum & brainstem"],
        ),
        20030: (
            "Other intracranial (dura, vessels, unlabelled)",
            [L.OTHER_INTRACRANIAL],
        ),
        20040: ("Whole intracranial space", [i for i in ids if i]),
    }

    rows = []
    pos = {v: k for k, v in enumerate(ids)}
    for lab_id in ids:
        if lab_id == 0:
            continue
        k = pos[lab_id]
        rows.append(
            dict(
                label=int(lab_id),
                name=L.name(lab_id),
                group=groups[lab_id],
                kind="CSF"
                if lab_id >= L.CSF_OFFSET
                else "tissue"
                if lab_id < L.OTHER_INTRACRANIAL
                else "other",
                volume_ml=volume_ml[k],
                md=md_mean[k],
                fa=fa_mean[k],
                mean=[m[k] for m in conc_means],
            )
        )
    for agg_id, (agg_name, members) in aggregates.items():
        ks = [pos[m] for m in members]
        v = volume_ml[ks]
        w = v / v.sum()
        rows.append(
            dict(
                label=agg_id,
                name=agg_name,
                group="Summary regions",
                kind="aggregate",
                volume_ml=v.sum(),
                md=pooled_mean(md_mean, md_n, ks),
                fa=pooled_mean(fa_mean, fa_n, ks),
                mean=[np.nansum(m[ks] * w) for m in conc_means],
            )
        )

    regions = pd.DataFrame(
        [
            {
                "label": r["label"],
                "name": r["name"],
                "group": r["group"],
                "kind": r["kind"],
                "volume_ml": round(r["volume_ml"], 3),
                "mean_diffusivity_mm2_per_s": r["md"],
                "fractional_anisotropy": r["fa"],
                **{f"conc_{s}": r["mean"][t] for t, s in enumerate(SESSIONS)},
                **{
                    f"amount_umol_{s}": r["mean"][t] * r["volume_ml"]
                    for t, s in enumerate(SESSIONS)
                },
            }
            for r in rows
        ]
    )
    long = pd.DataFrame(
        [
            {
                "label": r["label"],
                "name": r["name"],
                "group": r["group"],
                "kind": r["kind"],
                "session": s,
                "hours_after_injection": model_hours[t],
                "volume_ml": r["volume_ml"],
                "mean_concentration_mmol_per_L": r["mean"][t],
                "amount_umol": r["mean"][t] * r["volume_ml"],
            }
            for r in rows
            for t, s in enumerate(SESSIONS)
        ]
    )

    # --- downsampled volumes ---------------------------------------------------------------
    def grid_affine(b):
        scale = np.diag([b, b, b, 1.0])
        scale[:3, 3] = lo + (b - 1) / 2
        return affine0 @ scale

    out_vol = REPO / "data/volumes"
    out_vol.mkdir(parents=True, exist_ok=True)
    t1_lo, t1_hi = np.percentile(t1[ic], [1, 99.5])
    grids, dti_grids = {}, {}
    for b, mm in [(2, 1), (4, 2)]:
        conc_b = np.stack([nanblock(c, b) for c in conc_full])
        lab_b = blockmode(lab, b).astype(np.int16)
        t1_b = np.clip(
            (nanblock(t1, b) - t1_lo) / (t1_hi - t1_lo) * 255, 0, 255
        ).astype(np.uint8)
        tensor_b = [
            nanblock(c, b) for c in tensor
        ]  # average the tensor, then decompose
        for c in tensor_b:
            c[lab_b == 0] = np.nan  # outside the head, as for the concentration
        md_b = (tensor_b[0] + tensor_b[3] + tensor_b[5]) / 3
        fa_b, v1_b = tensor_eig(tensor_b)
        conc_b[:, lab_b == 0] = np.nan
        grids[mm] = (conc_b, lab_b, t1_b, grid_affine(b))
        dti_grids[mm] = (md_b, fa_b, v1_b)
        np.savez_compressed(
            out_vol / f"gonzo_{mm}mm.npz",
            concentration=conc_b.astype(np.float16),
            labels=lab_b,
            t1w=t1_b,
            mean_diffusivity=md_b.astype(np.float16),
            fractional_anisotropy=fa_b.astype(np.float16),
            principal_direction=v1_b.astype(np.float16),
            diffusion_tensor=np.stack(tensor_b, -1).astype(np.float16),
            affine=grid_affine(b),
            hours=np.array(model_hours),
            sessions=np.array(SESSIONS),
        )
        print(f"{mm} mm grid", lab_b.shape)
    del tensor

    # --- depth profiles (1 mm grid) --------------------------------------------------------
    conc1, lab1, _, aff1 = grids[1]
    is_tissue = (lab1 > 0) & (lab1 < L.OTHER_INTRACRANIAL)
    is_csf = lab1 >= L.CSF_OFFSET
    is_vent = np.isin(lab1 - L.CSF_OFFSET, list(L.VENTRICLES | {31, 63})) & is_csf
    inside = ndimage.binary_fill_holes(is_tissue | is_vent)
    inside = ndimage.binary_opening(inside, iterations=1)
    depth = ndimage.distance_transform_edt(inside) - ndimage.distance_transform_edt(
        ~inside
    )
    bins = [d for d in range(-6, 31) if d != 0]  # the distance transform never yields 0
    depth_rows = []
    for t, s in enumerate(SESSIONS):
        for d0 in bins:
            sel = (np.abs(depth - d0) <= 0.5) & (
                (is_tissue & (depth > 0)) | (is_csf & ~is_vent & (depth < 0))
            )
            vals = conc1[t][sel]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 50:
                continue
            depth_rows.append(
                dict(
                    depth_mm=int(d0),
                    session=s,
                    hours_after_injection=model_hours[t],
                    mean_concentration_mmol_per_L=vals.mean() if len(vals) else np.nan,
                    median_concentration_mmol_per_L=np.median(vals)
                    if len(vals)
                    else np.nan,
                    n_voxels=len(vals),
                )
            )
    depth_df = pd.DataFrame(depth_rows)

    # --- tables ----------------------------------------------------------------------------
    out = REPO / "data"
    sessions.to_csv(out / "sessions.csv", index=False)
    blood.to_csv(out / "blood_plasma.csv", index=False, float_format="%.6g")
    regions.to_csv(out / "regions.csv", index=False, float_format="%.6g")
    long.to_csv(out / "region_timeseries.csv", index=False, float_format="%.6g")
    depth_df.to_csv(out / "depth_profiles.csv", index=False, float_format="%.6g")

    # --- web viewer files ------------------------------------------------------
    web = REPO / "docs/data"
    web.mkdir(parents=True, exist_ok=True)
    conc2, lab2, t12, aff2 = grids[WEB_MM]
    cmin, cmax = -0.5, float(np.nanmax(conc2))
    scale = (cmax - cmin) / 65534

    def write_gz(name, arr):
        with gzip.open(web / name, "wb", compresslevel=9) as f:
            f.write(np.ascontiguousarray(arr).tobytes())

    write_gz("t1.u8.gz", t12)
    write_gz("labels.u16.gz", lab2.astype(np.uint16))
    conc_files = []
    for t, s in enumerate(SESSIONS):
        q = np.where(
            np.isfinite(conc2[t]),
            np.round((np.clip(conc2[t], cmin, cmax) - cmin) / scale) + 1,
            0,
        )
        write_gz(f"conc_{s}.u16.gz", q.astype(np.uint16))
        conc_files.append(f"conc_{s}.u16.gz")
    # DTI as bytes: 0 = no data; MD and FA quantised to 1..255; direction as |v1| RGB (R: x, G: y, B: z).
    md2, fa2, v12 = dti_grids[WEB_MM]
    md_max = 3e-3
    md_scale = md_max / 254
    write_gz(
        "md.u8.gz",
        np.where(
            np.isfinite(md2), np.round(np.clip(md2, 0, md_max) / md_scale) + 1, 0
        ).astype(np.uint8),
    )
    write_gz(
        "fa.u8.gz",
        np.where(np.isfinite(fa2), np.round(np.clip(fa2, 0, 1) * 254) + 1, 0).astype(
            np.uint8
        ),
    )
    write_gz(
        "dir.u8.gz",
        np.where(np.isfinite(v12), np.round(np.abs(v12) * 255), 0).astype(np.uint8),
    )
    centre = [int(v) for v in np.array(lab2.shape) // 2]
    meta = dict(
        shape=list(lab2.shape),
        spacing_mm=[WEB_MM] * 3,
        affine=aff2.round(4).tolist(),
        sessions=[
            dict(id=s, label=SESSION_LABELS[t], hours=round(model_hours[t], 2))
            for t, s in enumerate(SESSIONS)
        ],
        t1=dict(file="t1.u8.gz"),
        labels=dict(file="labels.u16.gz"),
        conc=dict(files=conc_files, scale=scale, offset=cmin - scale, unit="mmol/L"),
        dti=dict(
            md=dict(file="md.u8.gz", scale=md_scale, offset=-md_scale, unit="mm²/s"),
            fa=dict(file="fa.u8.gz", scale=1 / 254, offset=-1 / 254),
            dir=dict(file="dir.u8.gz", channels=3),
        ),
        initial_cursor=centre,
    )
    (web / "meta.json").write_text(json.dumps(meta))

    def clean(v):
        return None if not np.isfinite(v) else round(float(v), 5)

    present = set(np.unique(lab2).tolist())
    web_regions = [
        dict(
            label=int(r.label),
            name=r["name"],
            group=r.group,
            volume_ml=round(r.volume_ml, 2),
            md=clean(r.mean_diffusivity_mm2_per_s * 1e3),
            fa=clean(r.fractional_anisotropy),
            mean=[clean(r[f"conc_{s}"]) for s in SESSIONS],
            amount_umol=[clean(r[f"amount_umol_{s}"]) for s in SESSIONS],
        )
        for _, r in regions.iterrows()
        if r.kind == "aggregate" or r.label in present
    ]
    web_regions.sort(
        key=lambda r: (r["group"] != "Summary regions", r["group"], r["name"])
    )
    (web / "regions.json").write_text(
        json.dumps(
            dict(
                regions=web_regions,
                default_selection=[20002, 20010, 20012, 20021, 20022],
                blood=dict(
                    hours=blood.hours_after_injection.round(2).tolist(),
                    conc=blood.plasma_concentration_mmol_per_L.round(6).tolist(),
                ),
            )
        )
    )
    piv = depth_df.pivot(
        index="session", columns="depth_mm", values="mean_concentration_mmol_per_L"
    ).loc[SESSIONS[1:]]
    (web / "depth.json").write_text(
        json.dumps(
            dict(
                depth_mm=[int(d) for d in piv.columns],
                sessions=SESSION_LABELS[1:],
                mean=[[clean(v) for v in row] for row in piv.values],
            )
        )
    )

    # --- FEM mesh ---------------------------------------------------------------------------
    mesh = meshio.read(sub / "modeling/resolution32/data.vtk")
    times = ["0", "15716", "86001", "172280", "251641"]
    out_mesh = REPO / "data/mesh"
    out_mesh.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_mesh / "gonzo_mesh.npz",
        points=mesh.points.astype(np.float32),
        tetrahedra=mesh.cells_dict["tetra"].astype(np.int32),
        subdomains=mesh.cell_data["subdomains"][0].ravel().astype(np.int16),
        parcellations=mesh.cell_data["parcellations"][0].ravel().astype(np.int16),
        mean_diffusivity=mesh.cell_data["md"][0].ravel().astype(np.float32),
        fractional_anisotropy=mesh.cell_data["fa"][0].ravel().astype(np.float32),
        diffusion_tensor=mesh.cell_data["dt"][0].reshape(-1, 3, 3).astype(np.float32),
        concentration=np.stack(
            [mesh.point_data[f"concentration-{t}"].ravel() for t in times]
        ).astype(np.float32),
        boundary_concentration=np.stack(
            [mesh.point_data[f"boundary-concentration-{t}"].ravel() for t in times]
        ).astype(np.float32),
        hours=np.array([int(t) / 3600 for t in times]),
    )
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
