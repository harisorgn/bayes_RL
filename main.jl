using Turing
using DataFrames
using ReverseDiff
using Serialization
using FillArrays

include("read_data.jl")
include("task_types.jl")
include("softmax.jl")
include("ε_greedy.jl")
include("ε_soft.jl")
include("wsls.jl")

Turing.setadbackend(:reversediff)

group_d = Dict("MJ12" => 1, "MJ13" => 2)

(choice_v, data) = read_PRL("./prl/prl.csv", group_d)

chn = run_softmax_prl(choice_v, data)