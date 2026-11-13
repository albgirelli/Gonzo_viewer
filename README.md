# Gonzo dataset viewer

An interactive, browser-based view of the **Gonzo dataset**: MRI of a contrast agent (gadobutrol) injected into
the cerebrospinal fluid of a healthy human volunteer, imaged before injection and about 4, 24, 48 and 70 hours
after. The repository also contains a lightweight subset of the data (≈50 MB) and starter notebooks for
teaching, quick exploration and modelling.

* **Website** (dataset description, slice viewer with DTI maps, regional curves): `https://albgirelli.github.io/gonzo-viewer/`
* **Original dataset**: Riseth et al., Zenodo, [10.5281/zenodo.14266867](https://doi.org/10.5281/zenodo.14266867)
* **Data descriptor**: [medRxiv 10.1101/2025.07.23.25331971](https://doi.org/10.1101/2025.07.23.25331971)
* **Processing pipeline**: [github.com/jorgenriseth/gonzo](https://github.com/jorgenriseth/gonzo)


## Contents

```
data/
  sessions.csv            scan times
  blood_plasma.csv        tracer in blood plasma
  regions.csv             per-region volume, mean diffusivity, concentration and amount at each time
  region_timeseries.csv   same, long format
  depth_profiles.csv      concentration vs distance from the brain surface
  volumes/gonzo_{1,2}mm.npz  4D concentration, labels, T1w anatomy, DTI (tensor, MD, FA, direction)
  mesh/gonzo_mesh.npz     tetrahedral brain mesh with data (FEM ready)
docs/                     static website (GitHub Pages)
notebooks/                starter notebooks
scripts/                  how the subset was built from the original record
```

## Rebuilding the subset

Download `mri-processed.zip` and `mesh-data.zip` (plus `mri_dataset/timetable.tsv` and
`blood_concentrations.csv` from `mri-dataset.zip`) from Zenodo, unzip them into one folder and run

```
python scripts/build_subset.py /path/to/that/folder
python scripts/make_notebooks.py notebooks
```

`build_subset.py` swaps the plasma values of the ~25 h and ~40 h blood samples, which are transposed in the
source file.

## Credits and licence

Data: J. Riseth, T. Koch, S. L. Lian, T. H. Storås, L. T. Zikatanov, L. M. Valnes, K. Nordengen,
K.-A. Mardal, *The Gonzo dataset: human brain MRI data of CSF tracer evolution over 72 h for
data-integrated simulations*, Zenodo, [10.5281/zenodo.14266867](https://doi.org/10.5281/zenodo.14266867),
licensed CC BY 4.0. The files in `data/` and `docs/data/` are derived from it (cropped, downsampled,
region-averaged) under the same licence. For research, use the original record.
