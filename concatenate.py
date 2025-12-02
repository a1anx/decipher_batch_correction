import pandas as pd
import scanpy as sc
import os

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

adata_nonshifted = sc.read("adata/adata_shiftFalse_alpha0.0.h5ad")
#alpha_vec = [0.001, 0.005, 0.01, 0.05]
adata_shifted_001 = sc.read("adata/adata_shiftTrue_alpha0.001.h5ad")
adata_shifted_005 = sc.read("adata/adata_shiftTrue_alpha0.005.h5ad")
adata_shifted_01 = sc.read("adata/adata_shiftTrue_alpha0.01.h5ad")
adata_shifted_05 = sc.read("adata/adata_shiftTrue_alpha0.05.h5ad")
adata_shifted_1 = sc.read("adata/adata_shiftTrue_alpha0.1.h5ad")
adata_shifted_2 = sc.read("adata/adata_shiftTrue_alpha0.2.h5ad")
adata_shifted_5 = sc.read("adata/adata_shiftTrue_alpha0.5.h5ad")

adata_nonshifted.obs["shift"] = "False"
adata_shifted_001.obs["shift"] = "Alpha 0.001"
adata_shifted_005.obs["shift"] = "Alpha 0.005"
adata_shifted_01.obs["shift"] = "Alpha 0.01"
adata_shifted_05.obs["shift"] = "Alpha 0.05"
adata_shifted_1.obs["shift"] = "Alpha 0.1"
adata_shifted_2.obs["shift"] = "Alpha 0.2"
adata_shifted_5.obs["shift"] = "Alpha 0.5"

import anndata as ad

def concat_adata(adata_array, output_name):
    adata_combined = ad.concat(
        adata_array,
        axis=0,                       # concatenate cells
        join="outer",                 # union of var names (or use "inner")
        label=None,                   # since we already added .obs["patient"]
        merge="same"                  # requires identical obsm keys
    )
    adata_combined.write(f"adata/adata_combined_{output_name}.h5ad")

if __name__ == "__main__":
    # concat_adata(
    #     adata_array = [adata_nonshifted, adata_shifted_001, adata_shifted_005, adata_shifted_01],
    #     output_name="1"
    # )
    concat_adata(
        adata_array = [adata_nonshifted, adata_shifted_01, adata_shifted_05, adata_shifted_1],
        output_name="2"
    )