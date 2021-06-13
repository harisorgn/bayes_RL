### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ e34aafc0-b721-11eb-0e8f-1b876bbda906
using MCMCChains, ArviZ, Serialization, PyPlot, Random, Turing, LinearAlgebra

# ╔═╡ 92d44759-5a63-464d-bc2f-1ba2cc2221ec
include("task_types.jl")

# ╔═╡ 1a862514-dca4-4eb7-9236-e80f2a27eef4
include("softmax.jl")

# ╔═╡ a05e60a2-d069-46db-bf8e-9f2feb06e28b
include("read_abt.jl")

# ╔═╡ 356db041-eba4-4d10-bb81-5c6d38e1a9f8
ArviZ.use_style("arviz-darkgrid")

# ╔═╡ 4d6eed54-2db2-41ce-853e-6b2304b7d5d8
chn = deserialize("chn_softmax_FG.jls");

# ╔═╡ 0f7b1b7d-d1ae-412d-b4f6-fb0cd7b11c19
begin
	
	file_v = [["./abt/ER17_FG7142_trials.csv", "./abt/ER17_2vs1_trials.csv"],
			["./abt/SS2_FG7142_trials.csv", "./abt/SS2_2vs1_trials.csv"]]

	cb_file_v = [["./abt/ER17_FG7142_counterbalance.csv",
				"./abt/ER17_2vs1_counterbalance.csv"],
				["./abt/SS2_FG7142_counterbalance.csv",
				"./abt/SS2_2vs1_counterbalance.csv"]]

	group_d = Dict("V" => 1, "FG_0" => 1, "FG_3" => 2, "1" => 1, "2" => 1)

	(choice_m, data) = read_data(file_v, cb_file_v, group_d)
	
	mdl = softmax_model(choice_m, data);	
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
									"subj" => [i for i in 1:data.n_subjects],
									"session" => [i for i in 1:data.n_sessions],
									"trial" => [i for i in 1:30]),
						dims = Dict("μ_β_v" => ["interv"], 
									"σ_β_v" => ["interv"],
									"μ_η_v" => ["interv"], 
									"σ_η_v" => ["interv"],
									"β_norm_m" => ["interv", "subj"],
									"η_norm_m" => ["interv", "subj"],
									"subj_action_v" => ["subj"]),
						library="Turing")
	
	
end

# ╔═╡ 0c7096c8-9cb8-42a1-9a41-e0eff5884305
begin
	plot_trace(idt; var_names = ["μ_β_v", "μ_η_v"])
	gcf()
end

# ╔═╡ b97d7720-86dc-4b7d-9fb8-d72f616a6164
begin
	plot_energy(idt)
	gcf()
end

# ╔═╡ ac16ab02-e131-4ac6-8699-4b6d7f8e69dd
begin
	plot_violin(idt; var_names = ["μ_β_v"])
	gcf()
end

# ╔═╡ 91475576-5073-4af8-b8d4-0e1dbcf323d6
begin
	plot_violin(idt; var_names = ["μ_η_v"])
	gcf()
end

# ╔═╡ 4d971b8a-3171-425e-869d-d85d67bd7d26
loo(idt; var_name = "subj_action_v")

# ╔═╡ 33895959-ae6b-4ef0-a89e-6c218b3ba7dc
waic(idt; var_name = "subj_action_v")

# ╔═╡ 61a97276-48d9-487a-a015-9dce21c9c753
begin
	plot_violin(idt.sel(interv = "V").posterior["μ_β_v"] - 
				idt.sel(interv = "FG_3").posterior["μ_β_v"])
	gcf()
end

# ╔═╡ 1ba90288-edf8-4df7-9db6-55b9270a0cf0
begin
	plot_violin(idt.sel(interv = "V").posterior["μ_η_v"] - 
				idt.sel(interv = "FG_3").posterior["μ_η_v"])
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
# ╠═89571289-23e5-4deb-8bd9-d22b60683929
# ╠═8142b170-8bfc-4cf8-885d-b72d625bd182
# ╠═0c7096c8-9cb8-42a1-9a41-e0eff5884305
# ╠═b97d7720-86dc-4b7d-9fb8-d72f616a6164
# ╠═ac16ab02-e131-4ac6-8699-4b6d7f8e69dd
# ╠═91475576-5073-4af8-b8d4-0e1dbcf323d6
# ╠═4d971b8a-3171-425e-869d-d85d67bd7d26
# ╠═33895959-ae6b-4ef0-a89e-6c218b3ba7dc
# ╠═61a97276-48d9-487a-a015-9dce21c9c753
# ╠═1ba90288-edf8-4df7-9db6-55b9270a0cf0
