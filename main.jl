using Turing
using DataFrames
using ReverseDiff
using Serialization
using FillArrays
using MLDataUtils
using Distributed

include("read_abt.jl")
include("task_types.jl")
include("softmax.jl")
include("ε_greedy.jl")
include("ε_soft.jl")

Turing.setadbackend(:reversediff)

file_v = [["./abt/ER17_FG7142_trials.csv", "./abt/ER17_2vs1_trials.csv"],
			["./abt/SS2_FG7142_trials.csv", "./abt/SS2_2vs1_trials.csv"]]

cb_file_v = [["./abt/ER17_FG7142_counterbalance.csv", "./abt/ER17_2vs1_counterbalance.csv"],
			["./abt/SS2_FG7142_counterbalance.csv", "./abt/SS2_2vs1_counterbalance.csv"]]

group_d = Dict("V" => 1, "FG_0" => 1, "FG_3" => 2, "1" => 1, "2" => 1)

(choice_m, data) = read_data(file_v, cb_file_v, group_d)

chn = run_ε_soft(choice_m, data)

serialize("chn_lapse_FG_rec.jls", chn)

