using Turing
using Distributions
using Random
using DataFrames
using ReverseDiff
using Serialization
using FillArrays

struct PRL_t
	n_groups::Int64
	n_subjects::Array{Int64,1}
	group_v::Array{Int64,1}
	trial_v::Array{Int64,1}
	R_m::Matrix{Array{Float64,1}}
end

include("read.jl")
include("lapse.jl")
include("softmax.jl")

Turing.setadbackend(:reversediff)

file = "./data/prl.csv"

(action_m, data) = read_data(file)
	
chn = run_softmax(action_m, data)

serialize("chn_softmax.jls", chn)
