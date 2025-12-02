import pandas as pd
import scanpy as sc
import os

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

from figures import plot_embedding


def plot_concat_figure(
    adata, 
    out_folder="figures_delta_shift",
    output_name = ""
    ):
    sc.pl.embedding(
        adata,
        basis="X_default_umap",
        color=["cluster_latent", "latent_t", "branch_id", "shift"],
        ncols=4,
        show=False,
    )
    plt.savefig(
        os.path.join(out_folder, f"umap_combined_{output_name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    # Decipher visible embedding
    sc.pl.embedding(
        adata,
        basis="decipher_decipher_v",
        color=["cluster_latent", "latent_t", "branch_id", "shift"],
        ncols=4,
        show=False,
    )
    plt.savefig(
        os.path.join(out_folder, f"decipher_v_combined_{output_name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()
    

if __name__ == "__main__":
    figsize = [2.5, 2]
    # plot_concat_figure(
    #     adata = sc.read("adata/adata_combined_1.h5ad"),
    #     output_name = "1")
    plot_concat_figure(
        adata = sc.read("adata/adata_combined_2.h5ad"),
        output_name = "2")
    
    
