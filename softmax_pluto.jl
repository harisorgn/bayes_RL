### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ e34aafc0-b721-11eb-0e8f-1b876bbda906
begin
using ArviZ
using Serialization
using PyPlot
using Random
using Turing
using LinearAlgebra
using DataFrames
end

# ╔═╡ 92d44759-5a63-464d-bc2f-1ba2cc2221ec
include("task_types.jl")

# ╔═╡ 1a862514-dca4-4eb7-9236-e80f2a27eef4
include("softmax.jl")

# ╔═╡ a05e60a2-d069-46db-bf8e-9f2feb06e28b
include("read_abt.jl")

# ╔═╡ 356db041-eba4-4d10-bb81-5c6d38e1a9f8
ArviZ.use_style("arviz-darkgrid")

# ╔═╡ 4d6eed54-2db2-41ce-853e-6b2304b7d5d8
chn = deserialize("./chains/chn_softmax_FG.jls");

# ╔═╡ 0f7b1b7d-d1ae-412d-b4f6-fb0cd7b11c19
begin
	
	file_v = [["./abt/ER17_FG7142_trials.csv", "./abt/ER17_2vs1_trials.csv"],
			["./abt/SS2_FG7142_trials.csv", "./abt/SS2_2vs1_trials.csv"]]

	cb_file_v = [["./abt/ER17_FG7142_counterbalance.csv", 			   "./abt/ER17_2vs1_counterbalance.csv"],
				["./abt/SS2_FG7142_counterbalance.csv", 			"./abt/SS2_2vs1_counterbalance.csv"]]

	group_d = Dict("V" => 1, "FG_0" => 1, "FG_3" => 2, "1" => 1, "2" => 1)
	
	group_name_v = ["Vehicle", "FG_7142"]
	
	(choice_m, data) = read_data(file_v, cb_file_v, group_d; only_test = false)
	
	mdl = softmax_2_model(choice_m, data);	
	
end

# ╔═╡ c8c9d9df-a16c-4078-9fd1-087a2f0733f4
idt = from_mcmcchains(chn;
					observed_data=Dict("choice_m" => choice_m),
					coords = Dict("interv" => group_name_v,
									"subj" => [i for i in 1:data.n_subjects]),
					dims = Dict("μ_β_v" => ["interv"], 
								"σ_β_v" => ["interv"],
								"μ_η_v" => ["interv"], 
								"σ_η_v" => ["interv"],
								"β_norm_m" => ["interv", "subj"],
								"η_norm_m" => ["interv", "subj"],
								"subj_action_v" => ["subj"]),
					library="Turing")

# ╔═╡ b97d7720-86dc-4b7d-9fb8-d72f616a6164
begin
	
plot_energy(idt)
gcf()
	
end

# ╔═╡ 1fb51c9d-a7dd-4459-8f25-8490e972cc88
gelmandiag(chn)

# ╔═╡ 5adb5be9-3a61-4e67-bf70-54da6d18089e
ArviZ.ess(idt)

# ╔═╡ 55d27c13-37fe-49c8-803d-68ff3f0297ce
begin
	
gen = generated_quantities(mdl, MCMCChains.get_sections(chn, :parameters))	
μ_β_v = map(x -> x[1][:], gen[:])
μ_η_v = map(x -> x[2][:], gen[:])

t = (μ_β = [μ_β_v], μ_η = [μ_η_v])
	
idt_trans = from_namedtuple(t; coords = Dict("interv" => group_name_v), 
								dims = Dict("μ_β" =>["interv"], 
											"μ_η" => ["interv"]))
end

# ╔═╡ d63735f5-9961-469a-af84-5a5368690290
let
	
ax = plot_violin(idt_trans ; var_names = ["μ_β"], textsize = 12, quartiles = false,
						rug = true, 
						rug_kwargs = Dict("color" => "white", "alpha" => 0))

ax[1].set_ylabel(L"\mu_\beta"; fontsize = 16)

gcf()
	
end

# ╔═╡ 0d30eaa6-ba92-420a-a869-d1db4384eb3f
let
	
ax = plot_violin(idt_trans ; var_names = ["μ_η"], textsize = 12, quartiles = false,
							rug = true, 
							rug_kwargs = Dict("color" => "white", "alpha" => 0))

ax[1].set_ylabel(L"\mu_\eta"; fontsize = 16)

gcf()
	
end

# ╔═╡ f4370c94-e9a8-4c9f-adb0-dd70106139a5
let

for interv_group in group_name_v[2:end]

	_, ax = plt.subplots(1, 2)
		
	plot_violin(idt_trans.sel(interv = interv_group).posterior["μ_β"] - 		    				idt_trans.sel(interv = group_name_v[1]).posterior["μ_β"];
				textsize = 12, quartiles = false, rug = true, 
				rug_kwargs = Dict("color" => "white", "alpha" => 0), ax = ax[1])

	ax[1].set_title(L"\Delta \mu_{\beta}"; fontsize = 16)

	plot_violin(idt_trans.sel(interv = interv_group).posterior["μ_η"] - 							idt_trans.sel(interv = group_name_v[1]).posterior["μ_η"];
				textsize = 12, quartiles = false, rug = true, 
				rug_kwargs = Dict("color" => "white", "alpha" => 0), ax = ax[2])

	ax[2].set_title(L"\Delta \mu_{\eta}"; fontsize = 16)
end

gcf()
	
end

# ╔═╡ Cell order:
# ╠═e34aafc0-b721-11eb-0e8f-1b876bbda906
# ╠═92d44759-5a63-464d-bc2f-1ba2cc2221ec
# ╠═1a862514-dca4-4eb7-9236-e80f2a27eef4
# ╠═a05e60a2-d069-46db-bf8e-9f2feb06e28b
# ╠═356db041-eba4-4d10-bb81-5c6d38e1a9f8
# ╠═4d6eed54-2db2-41ce-853e-6b2304b7d5d8
# ╠═0f7b1b7d-d1ae-412d-b4f6-fb0cd7b11c19
# ╠═c8c9d9df-a16c-4078-9fd1-087a2f0733f4
# ╠═b97d7720-86dc-4b7d-9fb8-d72f616a6164
# ╠═1fb51c9d-a7dd-4459-8f25-8490e972cc88
# ╠═5adb5be9-3a61-4e67-bf70-54da6d18089e
# ╠═55d27c13-37fe-49c8-803d-68ff3f0297ce
# ╠═d63735f5-9961-469a-af84-5a5368690290
# ╠═0d30eaa6-ba92-420a-a869-d1db4384eb3f
# ╠═f4370c94-e9a8-4c9f-adb0-dd70106139a5
