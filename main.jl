using Turing
using Distributions
using Random
using DataFrames
using ReverseDiff
using Serialization
using FillArrays

struct ABT_t
	n_sessions::Int64
	n_subjects::Int64
	n_groups::Int64
	avail_actions_m::Matrix{Array{Int64,1}}
	group_m::Matrix{Int64}
	R_m::Matrix{Array{Float64,1}}
	trial_m::Matrix{Int64}
end

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
