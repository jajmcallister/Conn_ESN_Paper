
# 30 instantiations of 2 C.ELEGANS subnetworks of connectome and random and CFG
# 30 instantiations of 9 larva DROSOPHILA subnetworks of connectome and random and CFG
# 30 instantiations of 28 adult DROSOPHILA subnetworks of connectome and random and CFG
using JLD2, Statistics

# These are the c elegans reservoirs (2 - male and hermaphrodite, each with 30 instantiations of connectome and random and CFG)
eleg_conn_subnetworks30 = JLD2.load_object("loadpath")
eleg_rand_subnetworks30 = JLD2.load_object("loadpath")
eleg_cfg_subnetworks30 = JLD2.load_object("loadpath")
# These are larval drosophila reservoirs (9 subnetworks of connectome and random and CFG)
conn_subnetworks30 = JLD2.load_object("loadpath")
rand_subnetworks30 = JLD2.load_object("loadpath")
cfg_subnetworks30 = JLD2.load_object("loadpath")
# These are adult hemibrain reservoirs (28 subnetworks of connectome and random and CFG)
adult_conn_subnetworks30 = JLD2.load_object("loadpath")
adult_rand_subnetworks30 = JLD2.load_object("loadpath")
adult_cfg_subnetworks30 = JLD2.load_object("loadpath")

# combining reservoirs
# SO NOTE THAT THE LARVA DROSOPHILA ESNS ARE IN ID'S 3-11 
# AND THE ADULT DROSOPHILA ESNS ARE IN IDS 12-39, 
#SO WHEN PLOTTING OR ANALYSING, BE CAREFUL TO ONLY LOOK AT THE RELEVANT ONES
conn_ESNs = vcat(eleg_conn_subnetworks30, conn_subnetworks30, adult_conn_subnetworks30)
rand_ESNs = vcat(eleg_rand_subnetworks30, rand_subnetworks30, adult_rand_subnetworks30)
cfg_ESNs = vcat(eleg_cfg_subnetworks30, cfg_subnetworks30, adult_cfg_subnetworks30)



######################################
# load optimised parameters
input_conn = JLD2.load_object("load_path")
input_er = JLD2.load_object("load_path")
input_cfg = JLD2.load_object("load_path")
reg_conn = JLD2.load_object("load_path")
reg_er = JLD2.load_object("load_path")
reg_cfg = JLD2.load_object("load_path")
leak_conn = JLD2.load_object("load_path")
leak_er = JLD2.load_object("load_path")
leak_cfg = JLD2.load_object("load_path")

# load optimised performances
conn_opt_performances = JLD2.load_object("load_path")
er_opt_performances = JLD2.load_object("load_path")
cfg_opt_performances = JLD2.load_object("load_path")

# load the total weighted task variances for the above networks
weighted_tv_conn = JLD2.load_object("load_path")
weighted_tv_rand = JLD2.load_object("load_path")
weighted_tv_cfg = JLD2.load_object("load_path")

# PCs for cumulative variance plots
conn_evals = JLD2.load_object("load_path")
rand_evals = JLD2.load_object("load_path")
cfg_evals = JLD2.load_object("load_path")

# neural correlation 
conn_corrs = JLD2.load_object("load_path")
rand_corrs = JLD2.load_object("load_path")
cfg_corrs = JLD2.load_object("load_path")



