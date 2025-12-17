# decipher_batch_correction
Implementing batch correction for Azizi Lab's Decipher method

# Important folders
decipher_batch_correction/decipher-batch-correction
- decipher-bc includes core implementation models 
- decipher-bc-concat houses our implementation of simple-concatenation Decipher BC. Foundational models can be found in /Core_Implementation. Visualizations of running simple concatenation on simulated data can be found in /Visualizations
- simulated-data-analysis contains the results from running attention-based Decipher-BC on simulated data. This corresponds to the results that we showed for the final presentation

decipher_batch_correction/Bone_Marrow_Hematopoiesis: Many folders here are artifacts from tweaking model architecture. Most weren’t used for the final report, but here’s the parts that do matter:
- Preprocessing turned the original .h5ad dataset into a smaller subset
- Initial_Training contained results from running attention-based Decipher BC on the 20K subset
- Beta_1.0_AttentionHeads_2_Training houses the training and results on the version of attention-based Decipher BC where beta = 1.0 and number of attention heads = 2
- AttentionHeads_2_Training shows the training and results on the version of attention-based Decipher BC where beta = 0.1 and number of attention heads = 4
- Decipher_BC_Concat_Training shows the application of simple concatenation Decipher BC on the 20k BoneMarrowMap dataset
- Subset_by_cell_type shows the results of subsetting into exclusively Erythoid cells and the downstream analysis. Visualizations can be found in /Small Subset Training/Visualizations

# simulation
Contains all files relevant to simulated data generation
- concat_adata : concatenated adata with both alpha and delta shifts, used for Native Decipher vs Decipher-BC comparison (Fig. 3). Generated using shift_titration.py
- titration: concatenated adata from running shift_titration.py saved in "adata" folder and the metrics and visualizagion plots saved in the "results" folder
- params.py : fixed parameters for simulated data generation (seed, n_samples, etc)
- shift_titration : used functions from simulations.py to simulate data for each of the shifts and repeats for multiple magnitudes, concatenates them and calculates metrics for each concatenated datasets, as well as generate visualizations (Fig. 2)
- simulations.py : core functions used for simulated data generation that are used for shift_titration
