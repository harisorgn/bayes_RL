using DataFrames
using Serialization
using PyPlot

include("read_data.jl")
include("task_types.jl")

file_v = [["./abt/ER17_vfx_trials.csv"],
			["./abt/SS2_vfx_pre_vs_post_trials.csv"]]

cb_file_v = [["./abt/ER17_vfx_counterbalance.csv"],
			["./abt/SS2_vfx_pre_vs_post_counterbalance.csv"]]

group_d = Dict("V" => 1, "VFX_3" => 2)

(choice_m, data) = read_ABT(file_v, cb_file_v, group_d ; only_test = true)

ws_veh_v = Array{Float64,1}(undef, data.n_subjects)
ws_drug_v = Array{Float64,1}(undef, data.n_subjects)
ls_veh_v = Array{Float64,1}(undef, data.n_subjects)
ls_drug_v = Array{Float64,1}(undef, data.n_subjects)

for subject = 1 : data.n_subjects

	choice_v = choice_m[subject, 1]
	R_v = data.R_m[subject, 1]

	n_veh_w = sum(map((x,y) -> (x == 0) && (y != 0), choice_v[1:end-1], R_v[1:end-1])) 
	n_drug_w = sum(map((x,y) -> (x == 1) && (y != 0), choice_v[1:end-1], R_v[1:end-1]))
	n_veh_l = sum(map((x,y) -> (x == 0) && (y == 0), choice_v[1:end-1], R_v[1:end-1]))
	n_drug_l = sum(map((x,y) -> (x == 1) && (y == 0), choice_v[1:end-1], R_v[1:end-1]))

	n_veh_ws = sum(map((x,y,z) -> (x == 0) && (y != 0) && (z == x), choice_v[1:end-1], R_v[1:end-1], choice_v[2:end]))
	n_drug_ws = sum(map((x,y,z) -> (x == 1) && (y != 0) && (z == x), choice_v[1:end-1], R_v[1:end-1], choice_v[2:end]))
	n_veh_ls = sum(map((x,y,z) -> (x == 0) && (y == 0) && (z != x), choice_v[1:end-1], R_v[1:end-1], choice_v[2:end]))
	n_drug_ls = sum(map((x,y,z) -> (x == 1) && (y == 0) && (z != x), choice_v[1:end-1], R_v[1:end-1], choice_v[2:end]))

	ws_veh_v[subject] = n_veh_ws / n_veh_w
	ws_drug_v[subject] = n_drug_ws / n_drug_w
	ls_veh_v[subject] = n_veh_ls / n_veh_l
	ls_drug_v[subject] = n_drug_ls / n_drug_l

end

#=
figure()
ax = gca()

scatter(fill(1, length(ws_veh_v)), ws_veh_v)
scatter(fill(2, length(ws_drug_v)), ws_drug_v)

ax.set_xticks([1,2])
ax.set_xticklabels(["1", "2"] ; fontsize = 16)
title("win-stay" ; fontsize = 16)

figure()
ax = gca()

scatter(fill(1, length(ls_veh_v)), ls_veh_v)
scatter(fill(2, length(ls_drug_v)), ls_drug_v)

ax.set_xticks([1,2])
ax.set_xticklabels(["1", "2"] ; fontsize = 16)
title("lose-shift" ; fontsize = 16)
=#

figure()
ax = gca()

scatter(fill(1, length(ws_veh_v)), ws_veh_v - ws_drug_v)
scatter(fill(2, length(ls_veh_v)), ls_veh_v - ls_drug_v)

ax.set_xticks([1,2])
ax.set_xticklabels(["WS", "LS"] ; fontsize = 16)

show()

