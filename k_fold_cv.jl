using Turing
using Serialization
using MLDataUtils
using Random

include("task_types.jl")
include("read_abt.jl")
include("softmax.jl")
include("lapse.jl")

function K_fold_CV(choice_m, data, folds, chains, predict_f)

	train_choice_v = map(x -> choice_m[x,:], 
						folds.train_indices)

	train_data_v = map(x -> ABT_t(data.n_sessions, 
								length(x), 
								data.n_groups, 
								data.avail_actions_m[x,:], 
								data.group_m[x,:], 
								data.R_m[x,:], 
								data.trial_m[x,:]),
						folds.train_indices)

	val_choice_v = map(x -> choice_m[x,:], 
						collect.(folds.val_indices))

	val_data_v = map(x -> ABT_t(data.n_sessions, 
								length(x), 
								data.n_groups, 
								data.avail_actions_m[x,:], 
								data.group_m[x,:], 
								data.R_m[x,:], 
								data.trial_m[x,:]),
						collect.(folds.val_indices))

	
	elpd_v = map((val_choice_m, val_data, chain) -> predict_f(val_choice_m, val_data, chain), 
				val_choice_v, val_data_v, chains)

	return (sum(elpd_v), elpd_v)
end

chains = deserialize("chn_lapse_FG_CV.jls")

file_v = [["./abt/ER17_FG7142_trials.csv", "./abt/ER17_2vs1_trials.csv"],
			["./abt/SS2_FG7142_trials.csv", "./abt/SS2_2vs1_trials.csv"]]

cb_file_v = [["./abt/ER17_FG7142_counterbalance.csv", "./abt/ER17_2vs1_counterbalance.csv"],
			["./abt/SS2_FG7142_counterbalance.csv", "./abt/SS2_2vs1_counterbalance.csv"]]

group_d = Dict("V" => 1, "FG_0" => 1, "FG_3" => 2, "1" => 1, "2" => 1)

(choice_m, data) = read_data(file_v, cb_file_v, group_d)

folds = kfolds(collect(1:data.n_subjects), k = 14)

K_fold_CV(choice_m, data, folds, chains, predict_lapse)

