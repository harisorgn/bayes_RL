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

file_v = ["./abt/ER17_FG7142_trials.csv", "./abt/ER17_2vs1_trials.csv"]
cb_file_v = ["./abt/ER17_FG7142_counterbalance.csv", "./abt/ER17_2vs1_counterbalance.csv"]

(action_m, data) = read_data(file_v, cb_file_v)

chn = run_softmax(action_m, data)
serialize("chn_softmax_FG.jls", chn)

chn = run_lapse(action_m, data)
serialize("chn_lapse_FG.jls", chn)
