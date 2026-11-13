"""FreeSurfer label names and the anatomical groupings used in the lightweight subset."""

DESIKAN = [
    "unknown", "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "corpuscallosum",
    "cuneus", "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate",
    "lateraloccipital", "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal",
    "parahippocampal", "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis",
    "pericalcarine", "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal", "superiorparietal",
    "superiortemporal", "supramarginal", "frontalpole", "temporalpole", "transversetemporal", "insula",
]

LOBES = {
    "frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal",
                "precentral", "paracentral", "frontalpole"],
    "parietal": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
    "temporal": ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                 "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "cingulate": ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate",
                  "isthmuscingulate"],
    "insula": ["insula"],
}
LOBE_OF = {name: lobe for lobe, names in LOBES.items() for name in names}

ASEG = {
    2: "Left cerebral white matter", 41: "Right cerebral white matter",
    4: "Left lateral ventricle", 43: "Right lateral ventricle",
    5: "Left inferior lateral ventricle", 44: "Right inferior lateral ventricle",
    14: "3rd ventricle", 15: "4th ventricle", 24: "CSF (FreeSurfer)", 72: "5th ventricle",
    7: "Left cerebellum white matter", 46: "Right cerebellum white matter",
    8: "Left cerebellum cortex", 47: "Right cerebellum cortex",
    10: "Left thalamus", 49: "Right thalamus", 11: "Left caudate", 50: "Right caudate",
    12: "Left putamen", 51: "Right putamen", 13: "Left pallidum", 52: "Right pallidum",
    16: "Brainstem", 17: "Left hippocampus", 53: "Right hippocampus",
    18: "Left amygdala", 54: "Right amygdala", 26: "Left accumbens", 58: "Right accumbens",
    28: "Left ventral DC", 60: "Right ventral DC", 30: "Left vessel", 62: "Right vessel",
    31: "Left choroid plexus", 63: "Right choroid plexus", 77: "White matter hypointensities",
    85: "Optic chiasm", 251: "Corpus callosum (posterior)", 252: "Corpus callosum (mid-posterior)",
    253: "Corpus callosum (central)", 254: "Corpus callosum (mid-anterior)",
    255: "Corpus callosum (anterior)",
}

VENTRICLES = {4, 5, 14, 15, 24, 43, 44, 72}
WHITE = {2, 41, 77, 251, 252, 253, 254, 255}
DEEP_GM = {10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60}
CEREBELLUM = {7, 8, 46, 47}
BRAINSTEM = {16}

CSF_OFFSET = 10000   # CSF voxel label = CSF_OFFSET + label of the nearest brain region
OTHER_INTRACRANIAL = 9999


def tissue_name(label):
    if 1000 <= label < 3000:
        hemi = "Left" if label < 2000 else "Right"
        return f"{hemi} {DESIKAN[label % 1000]} (cortex)"
    return ASEG.get(label, f"label {label}")


def tissue_group(label):
    if 1000 <= label < 3000:
        return f"Cortex: {LOBE_OF.get(DESIKAN[label % 1000], 'other')}"
    if label in WHITE:
        return "White matter"
    if label in DEEP_GM:
        return "Deep grey matter"
    if label in CEREBELLUM:
        return "Cerebellum"
    if label in BRAINSTEM:
        return "Brainstem"
    return "Other tissue"


def csf_group(nearest):
    """Group a CSF voxel by the brain region it touches."""
    if nearest in VENTRICLES or nearest in {31, 63}:
        return "CSF: ventricles"
    if nearest in CEREBELLUM or nearest in BRAINSTEM:
        return "CSF: around cerebellum & brainstem"
    if 1000 <= nearest < 3000:
        return f"CSF: next to {LOBE_OF.get(DESIKAN[nearest % 1000], 'other')} cortex"
    return "CSF: deep / basal"


def name(label):
    if label == OTHER_INTRACRANIAL:
        return "Other intracranial (dura, vessels, unlabelled)"
    if label >= CSF_OFFSET:
        return f"CSF next to {tissue_name(label - CSF_OFFSET)}"
    return tissue_name(label)


def group(label):
    if label == OTHER_INTRACRANIAL:
        return "Other intracranial"
    if label >= CSF_OFFSET:
        return csf_group(label - CSF_OFFSET)
    return tissue_group(label)
