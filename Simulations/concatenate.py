import pandas as pd
import scanpy as sc
import os

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

import numpy as np
from sklearn.cluster import KMeans


adata_nonshifted = sc.read("newadata/adata_shiftFalse_alpha0.h5ad")
adata_shifted1 = sc.read("newadata/adata_shiftTrue_alpha1.h5ad")
adata_shifted2 = sc.read("newadata/adata_shiftTrue_alpha2.h5ad")
adata_shifted001 = sc.read("newadata/adata_shiftTrue_alpha0.01.h5ad")
adata_shifted005 = sc.read("newadata/adata_shiftTrue_alpha0.05.h5ad")
adata_shifted01 = sc.read("newadata/adata_shiftTrue_alpha0.1.h5ad")
adata_shifted02 = sc.read("newadata/adata_shiftTrue_alpha0.2.h5ad")
adata_shifted05 = sc.read("newadata/adata_shiftTrue_alpha0.5.h5ad")
#alpha_vec = [0.001, 0.005, 0.01, 0.05]
#[0.0001, 0.0005]
# adata_shifted_m0001 = sc.read("adata/adata_shiftTrue_alpha-0.0001.h5ad")
# adata_shifted_0001 = sc.read("adata/adata_shiftTrue_alpha0.0001.h5ad")
# adata_shifted_0005 = sc.read("adata/adata_shiftTrue_alpha0.0005.h5ad")
# adata_shifted_001 = sc.read("adata/adata_shiftTrue_alpha0.001.h5ad")
# adata_shifted_005 = sc.read("adata/adata_shiftTrue_alpha0.005.h5ad")
# adata_shifted_01 = sc.read("adata/adata_shiftTrue_alpha0.01.h5ad")
# adata_shifted_05 = sc.read("adata/adata_shiftTrue_alpha0.05.h5ad")
# adata_shifted_1 = sc.read("adata/adata_shiftTrue_alpha0.1.h5ad")
# adata_shifted_2 = sc.read("adata/adata_shiftTrue_alpha0.2.h5ad")
# adata_shifted_5 = sc.read("newadata/adata_shiftTrue_alpha0.5.h5ad")

adata_nonshifted.obs["shift"] = "Non-shifted"
adata_shifted1.obs["shift"] = "Alpha 1"
adata_shifted2.obs["shift"] = "Alpha 2"
adata_shifted001.obs["shift"] = "Alpha 0.01"
adata_shifted005.obs["shift"] = "Alpha 0.05"
adata_shifted01.obs["shift"] = "Alpha 0.1"
adata_shifted02.obs["shift"] = "Alpha 0.2"
adata_shifted05.obs["shift"] = "Alpha 0.5"
# adata_shifted_m0001.obs["shift"] = "Alpha -0.0001"
# adata_shifted_0001.obs["shift"] = "Alpha 0.0001"
# adata_shifted_0005.obs["shift"] = "Alpha 0.0005"
# adata_shifted_001.obs["shift"] = "Alpha 0.001"
# adata_shifted_005.obs["shift"] = "Alpha 0.005"
# adata_shifted_01.obs["shift"] = "Alpha 0.01"
# adata_shifted_05.obs["shift"] = "Alpha 0.05"
# adata_shifted_1.obs["shift"] = "Alpha 0.1"
# adata_shifted_2.obs["shift"] = "Alpha 0.2"
# adata_shifted_5.obs["shift"] = "Alpha 0.5"


    # ari_all = pd.concat([ari_true, ari_shifted], ignore_index=True)
    # ari_all.to_csv(os.path.join(out_folder, f"ari_by_method_true_vs_shifted{mag}.csv"), index=False)



import anndata as ad

def concat_adata(
    adata_array, 
    output_name,
    k_clusters = 20,
    ):
    adata_combined = ad.concat(
        adata_array,
        axis=0,                       # concatenate cells
        join="outer",                 # union of var names (or use "inner")
        label=None,                   # since we already added .obs["shift"]
        merge="same"                  # requires identical obsm keys
    )
    k_means = KMeans(k_clusters)
    latent_z = adata_combined.obsm["latent_z"]
    k_means.fit(latent_z)
    adata_combined.obs["cluster_latent"] = k_means.labels_
    # for each cluster, order the other clusters by distance of their kmeans center
    cluster_centers = k_means.cluster_centers_
    cluster_rank = np.argsort(
        np.linalg.norm(cluster_centers[:, None] - cluster_centers[None, :], axis=2)
    )
    adata_combined.uns["cluster_rank"] = cluster_rank[:, 1:]
    adata_combined.write(f"newadata/adata_combined_{output_name}.h5ad")
    return adata_combined


if __name__ == "__main__":
    adata_array = [adata_nonshifted, adata_shifted01, adata_shifted02, adata_shifted05, adata_shifted1]
    # adata_combined = ad.concat(
    #     adata_array,
    #     axis=0,                       # concatenate cells
    #     join="outer",                 # union of var names (or use "inner")
    #     label=None,                   # since we already added .obs["patient"]
    #     merge="same"                  # requires identical obsm keys
    # )
    # print("combined obsm:", adata_combined.obsm.keys())
    # latent = adata_combined.obsm['latent_z']
    # print("Any NaNs?", np.isnan(latent).any())
    # print("Number of NaNs:", np.isnan(latent).sum())
    concat_adata(
        adata_array = adata_array,
        output_name="norm_3"
    )
    # concat_adata(
    #     adata_array = [adata_nonshifted, adata_shifted_001, adata_shifted_005, adata_shifted_01],
    #     output_name="c2"
    # )
    # concat_adata(
    #     adata_array = [adata_nonshifted, adata_shifted_01, adata_shifted_05, adata_shifted_1],
    #     output_name="c3"
    # )
    