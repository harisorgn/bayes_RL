using Turing
using Distributions
using Random
using DataFrames
using ReverseDiff
using Serialization
using FillArrays

include("read.jl")
include("lapse.jl")
include("softmax.jl")

struct ABT_t
	n_sessions::Int64
	n_subjects::Int64
	n_interv::Int64
	n_trials::Int64
	avail_actions_m::Matrix{Array{Int64,1}}
	trial_m::Matrix{Int64}
	interv_m::Matrix{Int64}
	R_m::Array{Float64, 1}
end

Turing.setadbackend(:reversediff)

chn = run_softmax()
