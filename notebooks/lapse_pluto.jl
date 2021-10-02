### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ e34aafc0-b721-11eb-0e8f-1b876bbda906
using MCMCChains, ArviZ, Serialization, PyPlot, Random, Turing, LinearAlgebra

# ╔═╡ 92d44759-5a63-464d-bc2f-1ba2cc2221ec
include("task_types.jl")

# ╔═╡ 1a862514-dca4-4eb7-9236-e80f2a27eef4
include("lapse.jl")

# ╔═╡ a05e60a2-d069-46db-bf8e-9f2feb06e28b
include("read_abt.jl")

# ╔═╡ 356db041-eba4-4d10-bb81-5c6d38e1a9f8
ArviZ.use_style("arviz-darkgrid")

# ╔═╡ 4d6eed54-2db2-41ce-853e-6b2304b7d5d8
chn = deserialize("./chains/chn_lapse_FG_no2vs1.jls");

# ╔═╡ 0f7b1b7d-d1ae-412d-b4f6-fb0cd7b11c19
begin
	
	file_v = [["./abt/ER17_FG7142_trials.csv"],
			["./abt/SS2_FG7142_trials.csv"]]

	cb_file_v = [["./abt/ER17_FG7142_counterbalance.csv"],
				["./abt/SS2_FG7142_counterbalance.csv"]]

	group_d = Dict("V" => 1, "FG_0" => 1, "FG_3" => 2)
	
	(choice_m, data) = read_data(file_v, cb_file_v, group_d)
	
	mdl = lapse_model(choice_m, data);	
end

# ╔═╡ 89571289-23e5-4deb-8bd9-d22b60683929
loglikelihoods = Turing.pointwise_loglikelihoods(mdl, 
									MCMCChains.get_sections(chn, :parameters));

# ╔═╡ 8142b170-8bfc-4cf8-885d-b72d625bd182
begin

	trial_names = string.(keys(loglikelihoods))
	
	(n_samples, n_chains) = size(loglikelihoods[trial_names[1]])
	loglikelihoods_m = zeros(n_samples, n_chains, data.n_subjects)
	subj_action_v = Array{Array{Int64,1},1}(undef, data.n_subjects)
	
	for subj in 1 : data.n_subjects
		
		subj_name = string("choice_m[", subj)
		
		subj_trial_names = filter(x -> occursin(subj_name, x), trial_names)
		
		subj_loglikelihood_vals = getindex.(Ref(loglikelihoods), subj_trial_names)
		
		loglikelihoods_m[:,:,subj] += reduce(+, subj_loglikelihood_vals)
		
		subj_action_v[subj] = reduce(vcat, choice_m[subj,:])
	end
	
	loglikelihoods_m = permutedims(loglikelihoods_m, (2, 1, 3))
	
	idt = from_mcmcchains(chn;
						log_likelihood=Dict("subj_action_v" => loglikelihoods_m),
						observed_data=Dict("choice_m" => choice_m),
						coords = Dict("interv" => ["V","FG_3"],
									"subj" => [i for i in 1:data.n_subjects]),
						dims = Dict("μ_ε_v" => ["interv"], 
									"σ_ε_v" => ["interv"],
									"μ_η_v" => ["interv"], 
									"σ_η_v" => ["interv"],
									"μ_s_v" => ["interv"], 
									"σ_s_v" => ["interv"],
									"ε_norm_m" => ["interv", "subj"],
									"η_norm_m" => ["interv", "subj"],
									"s_norm_m" => ["interv", "subj"],
									"subj_action_v" => ["subj"]),
						library="Turing")
	
	
end

# ╔═╡ 0c7096c8-9cb8-42a1-9a41-e0eff5884305
begin
	plot_trace(idt; var_names = ["μ_ε_v", "μ_η_v", "μ_s_v"])
	gcf()
end

# ╔═╡ b97d7720-86dc-4b7d-9fb8-d72f616a6164
begin
	plot_energy(idt)
	gcf()
end

# ╔═╡ 51062202-c6b5-4caa-8664-dbc0e97c0039
gelmandiag(chn)

# ╔═╡ e4d0cc3e-906a-4b31-8003-a0e2ce3c5c85
ArviZ.ess(idt)

# ╔═╡ Cell order:
# ╠═e34aafc0-b721-11eb-0e8f-1b876bbda906
# ╠═92d44759-5a63-464d-bc2f-1ba2cc2221ec
# ╠═1a862514-dca4-4eb7-9236-e80f2a27eef4
# ╠═a05e60a2-d069-46db-bf8e-9f2feb06e28b
# ╠═356db041-eba4-4d10-bb81-5c6d38e1a9f8
# ╠═4d6eed54-2db2-41ce-853e-6b2304b7d5d8
# ╠═0f7b1b7d-d1ae-412d-b4f6-fb0cd7b11c19
# ╠═89571289-23e5-4deb-8bd9-d22b60683929
# ╠═8142b170-8bfc-4cf8-885d-b72d625bd182
# ╠═0c7096c8-9cb8-42a1-9a41-e0eff5884305
# ╠═b97d7720-86dc-4b7d-9fb8-d72f616a6164
# ╠═51062202-c6b5-4caa-8664-dbc0e97c0039
# ╠═e4d0cc3e-906a-4b31-8003-a0e2ce3c5c85
