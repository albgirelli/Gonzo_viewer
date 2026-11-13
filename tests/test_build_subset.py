"""Tests for the DTI processing in scripts/build_subset.py and the web viewer data files.

Run from the repository root:  python -m pytest tests
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_subset import nanblock, pooled_mean, tensor_eig  # noqa: E402

WEB = REPO / "docs/data"


def tensors(*mats):
    """Six component arrays (xx, xy, xz, yy, yz, zz) from a list of 3x3 matrices."""
    m = np.array(mats, dtype=np.float32)
    return [m[:, i, j] for i, j in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]]


# --- tensor_eig ---------------------------------------------------------------------------


def test_isotropic_tensor_has_zero_fa():
    fa, _ = tensor_eig(tensors(1e-3 * np.eye(3)))
    assert fa[0] == pytest.approx(0, abs=1e-6)


def test_line_tensor_has_unit_fa_and_points_along_the_line():
    fa, v1 = tensor_eig(tensors(np.diag([0, 0, 1e-3])))
    assert fa[0] == pytest.approx(1, abs=1e-6)
    np.testing.assert_allclose(np.abs(v1[0]), [0, 0, 1], atol=1e-6)


def test_known_fa_and_rotated_direction():
    lam = np.array([1.7e-3, 0.3e-3, 0.3e-3])  # typical white matter
    md = lam.mean()
    expected = np.sqrt(1.5 * ((lam - md) ** 2).sum() / (lam**2).sum())
    c, s = np.cos(0.4), np.sin(0.4)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    fa, v1 = tensor_eig(tensors(R @ np.diag(lam) @ R.T))
    assert fa[0] == pytest.approx(expected, rel=1e-5)
    np.testing.assert_allclose(np.abs(v1[0]), np.abs(R[:, 0]), atol=1e-5)


def test_missing_tensor_gives_nan_and_keeps_shape():
    comp = [np.array([[1e-3, np.nan]], np.float32)] + [np.zeros((1, 2), np.float32)] * 2
    comp += [
        np.array([[1e-3, 1e-3]], np.float32),
        np.zeros((1, 2), np.float32),
        np.array([[1e-3, 1e-3]], np.float32),
    ]
    fa, v1 = tensor_eig(comp)
    assert fa.shape == (1, 2) and v1.shape == (1, 2, 3)
    assert np.isfinite(fa[0, 0]) and np.isnan(fa[0, 1]) and np.isnan(v1[0, 1]).all()


def test_chunking_does_not_change_the_result():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 3, 3))
    comp = tensors(*(a @ a.T * 1e-3 for a in A))
    fa1, v1 = tensor_eig(comp)
    fa2, v2 = tensor_eig(comp, chunk=7)
    np.testing.assert_allclose(fa1, fa2)
    np.testing.assert_allclose(v1, v2)


# --- averaging ----------------------------------------------------------------------------


def test_pooled_mean_skips_regions_without_data():
    means = np.array([1.0, np.nan, 3.0])
    counts = np.array([1, 0, 3])
    assert pooled_mean(means, counts, [0, 1, 2]) == pytest.approx((1 * 1 + 3 * 3) / 4)
    assert np.isnan(pooled_mean(means, counts, [1]))


def test_nanblock_ignores_nan():
    a = np.full((2, 2, 2), np.nan, np.float32)
    a[0, 0, 0], a[1, 1, 1] = 1.0, 3.0
    assert nanblock(a, 2)[0, 0, 0] == pytest.approx(2.0)
    assert np.isnan(nanblock(np.full((2, 2, 2), np.nan, np.float32), 2)[0, 0, 0])


# --- web viewer data files ----------------------------------------------------------------


@pytest.fixture(scope="module")
def meta():
    return json.loads((WEB / "meta.json").read_text())


def load_u8(name):
    with gzip.open(WEB / name) as f:
        return np.frombuffer(f.read(), np.uint8)


def test_web_dti_files_match_meta(meta):
    n = int(np.prod(meta["shape"]))
    d = meta["dti"]
    md, fa, direction = (
        load_u8(d["md"]["file"]),
        load_u8(d["fa"]["file"]),
        load_u8(d["dir"]["file"]),
    )
    assert md.size == fa.size == n
    assert direction.size == n * d["dir"]["channels"]
    # Same voxels have data in MD and FA (0 = no data), and only there a direction.
    assert np.array_equal(md > 0, fa > 0)
    assert not direction.reshape(n, 3)[fa == 0].any()


def test_web_dti_values_are_physiological(meta):
    d = meta["dti"]
    md = load_u8(d["md"]["file"]).astype(float)
    fa = load_u8(d["fa"]["file"]).astype(float)
    has = md > 0
    md_val = md[has] * d["md"]["scale"] + d["md"]["offset"]
    fa_val = fa[has] * d["fa"]["scale"] + d["fa"]["offset"]
    assert 0.6e-3 < np.median(md_val) < 1.2e-3  # brain tissue, mm²/s
    assert 0.1 < np.median(fa_val) < 0.5
    assert fa_val.min() >= -1e-9 and fa_val.max() <= 1 + 1e-9
    with gzip.open(WEB / meta["labels"]["file"]) as f:
        labels = np.frombuffer(f.read(), np.uint16)
    assert (labels[has] > 0).all()  # nothing outside the head
    # DTI is measured in brain tissue; at 1 mm a few edge voxels are labelled CSF (partial volume).
    assert ((labels[has] < 9999).mean()) > 0.9


def test_web_regions_have_dti_for_tissue_only():
    regions = json.loads((WEB / "regions.json").read_text())["regions"]
    by_name = {r["name"]: r for r in regions}
    wm, cortex = by_name["Cerebral white matter"], by_name["Cerebral cortex"]
    assert wm["fa"] > cortex["fa"]  # white matter is more anisotropic
    # An aggregate containing CSF must average only voxels with DTI (not be diluted by CSF volume).
    assert by_name["Whole intracranial space"]["md"] == pytest.approx(
        by_name["All brain tissue"]["md"], rel=0.02
    )
    assert by_name["All intracranial CSF"]["md"] is None
