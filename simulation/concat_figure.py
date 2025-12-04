import pandas as pd
import scanpy as sc
import os

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

from figures import plot_embedding

def umap_norm(
    adata,
    out_folder="umap_figures",
    output_name = "norm3",
    seed=0,
):
    os.makedirs(out_folder, exist_ok=True)
    adata_temp = adata.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    # optionally: sc.pp.highly_variable_genes, sc.pp.scale, etc.

    # ---------- 2) UMAP on normalized data ----------
    sc.pp.neighbors(adata_temp, random_state=seed)
    sc.tl.umap(adata_temp, random_state=seed)
    adata_temp.obsm["X_norm_umap"] = adata_temp.obsm["X_umap"]
    sc.pl.embedding(
        adata_temp,
        basis="X_norm_umap",
        color=["cluster_latent", "latent_t", "branch_id", "shift"],
        ncols=4,
        show=False,
    )
    plt.savefig(
        os.path.join(out_folder, f"norm_combined_umap_{output_name}.png"),
        bbox_inches="tight",
    )
    
    

def plot_concat_figure(
    adata, 
    out_folder="new_figures",
    output_name = ""
    ):
    # normalized Umap
    sc.pl.embedding(
        adata,
        basis="X_norm_umap",
        color=["cluster_latent", "latent_t", "branch_id", "shift"],
        ncols=4,
        show=False,
    )
    plt.savefig(
        os.path.join(out_folder, f"norm_umap_combined_{output_name}.png"),
        bbox_inches="tight",
    )
    plt.close()
    
    # original 
    sc.pl.embedding(
        adata,
        basis="X_default_umap",
        color=["cluster_latent", "latent_t", "branch_id", "shift"],
        ncols=4,
        show=False,
    )
    plt.savefig(
        os.path.join(out_folder, f"umap_combined_{output_name}.png"),
        bbox_inches="tight",
    )
    plt.close()
    
    # Decipher visible embedding
    # sc.pl.embedding(
    #     adata,
    #     basis="decipher_decipher_v",
    #     color=["cluster_latent", "latent_t", "branch_id", "shift"],
    #     ncols=4,
    #     show=False,
    # )
    # plt.savefig(
    #     os.path.join(out_folder, f"decipher_v_combined_{output_name}.png"),
    #     bbox_inches="tight",
    # )
    # plt.close()
    

if __name__ == "__main__":
    figsize = [2.5, 2]
    umap_norm(
        adata = sc.read("newadata/adata_combined_norm_3.h5ad"))
    # plot_concat_figure(
    #     adata = sc.read("newadata/adata_combined_norm_2.h5ad"),
    #     output_name = "new2")
    # plot_concat_figure(
    #     adata = sc.read("adata/adata_combined_c2.h5ad"),
    #     output_name = "c2")
    # plot_concat_figure(
    #     adata = sc.read("adata/adata_combined_c3.h5ad"),
    #     output_name = "c3")
    

