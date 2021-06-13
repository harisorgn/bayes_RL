using Turing
using DataFrames
using ReverseDiff
using Serialization
using FillArrays
using MLDataUtils
using Distributed

addprocs(14)

@everywhere using Turing

include("read_abt.jl")
@everywhere include("task_types.jl")
@everywhere  include("lapse.jl")
@everywhere  include("softmax.jl")

Turing.setadbackend(:reversediff)

file_v = [["./abt/ER17_vfx_trials.csv", "./abt/ER17_2vs1_trials.csv"],
			["./abt/SS2_vfx_pre_vs_post_trials.csv", "./abt/SS2_2vs1_trials.csv"]]

cb_file_v = [["./abt/ER17_vfx_counterbalance.csv", "./abt/ER17_2vs1_counterbalance.csv"],
			["./abt/SS2_vfx_pre_vs_post_counterbalance.csv", "./abt/SS2_2vs1_counterbalance.csv"]]

group_d = Dict("V" => 1, "VFX_0" => 1, "VFX_3" => 2, "1" => 1, "2" => 1)

(choice_m, data) = read_data(file_v, cb_file_v, group_d)

folds = kfolds(collect(1:data.n_subjects), k = nworkers())

fold_choice_v = map(x -> choice_m[x,:], folds.train_indices)
fold_data_v = map(x -> ABT_t(data.n_sessions, 
							length(x), 
							data.n_groups, 
							data.avail_actions_m[x,:], 
							data.group_m[x,:], 
							data.R_m[x,:], 
							data.trial_m[x,:]),
					folds.train_indices)

serialize("folds_FG.jls", folds)

chn = pmap((fold_choice, fold_data) -> sample(softmax_model(fold_choice, fold_data), NUTS(1000, 0.65), 2000), 
			fold_choice_v, fold_data_v)

serialize("chn_softmax_FG_CV.jls", chn)

chn = pmap((fold_choice, fold_data) -> sample(lapse_model(fold_choice, fold_data), NUTS(1000, 0.65), 2000), 
			fold_choice_v, fold_data_v)

serialize("chn_lapse_FG_CV.jls", chn)
