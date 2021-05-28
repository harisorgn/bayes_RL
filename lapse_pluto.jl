### A Pluto.jl notebook ###
# v0.14.5

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
chn = deserialize("chn_lapse_FG.jls");

# ╔═╡ 0f7b1b7d-d1ae-412d-b4f6-fb0cd7b11c19
begin
	
	file_v = ["./abt/ER17_FG7142_trials.csv", 
				"./abt/ER17_2vs1_trials.csv"]
	cb_file_v =["./abt/ER17_FG7142_counterbalance.csv", 			
				"./abt/ER17_2vs1_counterbalance.csv"]

	(action_m, data) = read_data(file_v, cb_file_v);
	
	mdl = lapse_model(action_m, data);
	
	prior = sample(mdl, Prior(), 1000; progress=false);
end

# ╔═╡ cffce24f-80d9-49d4-8a37-c33432b5e147
model_predict = lapse_model(missing, data);

# ╔═╡ c2c65caf-8153-4c27-befe-bdc985ffa479
prior_predictive = predict(model_predict, prior)

# ╔═╡ db9c7c58-4ec2-4bad-bd6c-55b47e7a8d23
posterior_predictive = predict(model_predict, chn)

# ╔═╡ 89571289-23e5-4deb-8bd9-d22b60683929
loglikelihoods = Turing.pointwise_loglikelihoods(mdl, 
									MCMCChains.get_sections(chn, :parameters));

# ╔═╡ 8142b170-8bfc-4cf8-885d-b72d625bd182
begin

	trial_names = string.(keys(posterior_predictive))
	
	loglikelihoods_vals = getindex.(Ref(loglikelihoods), trial_names)
	loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (2, 1, 3))

	(n_samples, n_chains) = size(loglikelihoods_vals[1])
	loglikelihoods_m = zeros(n_samples, n_chains, data.n_subjects)
	subj_action_v = Array{Array{Int64,1},1}(undef, data.n_subjects)
	
	for subj in 1 : data.n_subjects
		
		subj_name = string("action_m[", subj)
		
		subj_trial_names = filter(x -> occursin(subj_name, x), trial_names)
		
		subj_loglikelihood_vals = getindex.(Ref(loglikelihoods), subj_trial_names)
		
		loglikelihoods_m[:,:,subj] += reduce(+, subj_loglikelihood_vals)
		
		subj_action_v[subj] = reduce(vcat, action_m[subj,:])
	end
	
	loglikelihoods_m = permutedims(loglikelihoods_m, (2, 1, 3))
	
	idt = from_mcmcchains(chn;
						posterior_predictive=posterior_predictive,
						log_likelihood=Dict("subj_action_v" => loglikelihoods_m),
						prior=prior,
						prior_predictive=prior_predictive,
						observed_data=Dict("action_m" => action_m),
						coords = Dict("interv" => ["V","FG_3","FG_6"],
									"subj" => [i for i in 1:data.n_subjects],
									"session" => [i for i in 1:data.n_sessions],
									"trial" => [i for i in 1:30]),
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

# ╔═╡ ac16ab02-e131-4ac6-8699-4b6d7f8e69dd
begin
	plot_posterior(idt; var_names = ["μ_ε_v", "μ_η_v", "μ_s_v"])
	gcf()
end

# ╔═╡ 4d971b8a-3171-425e-869d-d85d67bd7d26
loo(idt; var_name = "subj_action_v")

# ╔═╡ 98cd7bcc-9564-4e8a-85df-619e1726a238
waic(idt; var_name = "subj_action_v")

# ╔═╡ Cell order:
# ╠═e34aafc0-b721-11eb-0e8f-1b876bbda906
# ╠═92d44759-5a63-464d-bc2f-1ba2cc2221ec
# ╠═1a862514-dca4-4eb7-9236-e80f2a27eef4
# ╠═a05e60a2-d069-46db-bf8e-9f2feb06e28b
# ╠═356db041-eba4-4d10-bb81-5c6d38e1a9f8
# ╠═4d6eed54-2db2-41ce-853e-6b2304b7d5d8
# ╠═0f7b1b7d-d1ae-412d-b4f6-fb0cd7b11c19
# ╠═cffce24f-80d9-49d4-8a37-c33432b5e147
# ╠═c2c65caf-8153-4c27-befe-bdc985ffa479
# ╠═db9c7c58-4ec2-4bad-bd6c-55b47e7a8d23
# ╠═89571289-23e5-4deb-8bd9-d22b60683929
# ╠═8142b170-8bfc-4cf8-885d-b72d625bd182
# ╠═0c7096c8-9cb8-42a1-9a41-e0eff5884305
# ╠═b97d7720-86dc-4b7d-9fb8-d72f616a6164
# ╠═ac16ab02-e131-4ac6-8699-4b6d7f8e69dd
# ╠═4d971b8a-3171-425e-869d-d85d67bd7d26
# ╠═98cd7bcc-9564-4e8a-85df-619e1726a238
