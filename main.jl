using Turing
using Distributions
using Random
using DataFrames
using ReverseDiff
using Serialization
using FillArrays

include("task_types.jl")
include("read_abt.jl")
include("lapse.jl")
include("softmax.jl")

Turing.setadbackend(:reversediff)

file_v = [["./abt/ER17_FG7142_trials.csv"], ["./abt/SS2_FG7142_trials.csv"]]
cb_file_v = [["./abt/ER17_FG7142_counterbalance.csv"], ["./abt/SS2_FG7142_counterbalance.csv"]]

group_d = Dict("V" => 1, "FG_0" => 1, "FG_3" => 2, "1" => 1, "2" => 1)

(choice_m, data) = read_data(file_v, cb_file_v, group_d)

chn = run_softmax(choice_m, data)
serialize("chn_softmax_FG.jls", chn)

chn = run_lapse(choice_m, data)
serialize("chn_lapse_FG.jls", chn)
