using Turing
using DataFrames
using ReverseDiff
using Serialization
using FillArrays

include("read_abt.jl")
include("task_types.jl")
include("softmax.jl")
include("ε_greedy.jl")
include("ε_soft.jl")
include("wsls.jl")

Turing.setadbackend(:reversediff)

file_v = [["./abt/ER17_FG7142_trials.csv"],
			["./abt/SS2_FG7142_trials.csv"]]

cb_file_v = [["./abt/ER17_FG7142_counterbalance.csv"],
			["./abt/SS2_FG7142_counterbalance.csv"]]

group_d = Dict("V" => 1, "FG_3" => 2)

(choice_m, data) = read_data(file_v, cb_file_v, group_d ; only_test = true)

run_wsls(choice_m, data)