using Turing
using PyPlot
using ArviZ
using Serialization
using LaTeXStrings

include("task_types.jl")
include("read_abt.jl")
include("softmax.jl")
include("lapse.jl")

function plot_trans_violin_softmax(choice_m, data, group_name_v, study_name)
	
	chn = deserialize(string("./chains/chn_softmax_", study_name, ".jls"));

	m = softmax_model(choice_m, data)

	gen = generated_quantities(m, MCMCChains.get_sections(chn, :parameters))

	μ_β_v = map(x -> x[1][1:2], gen[:])
	μ_η_v = map(x -> x[2][1:2], gen[:])

	t = (μ_β = [μ_β_v], μ_η = [μ_η_v])

	idt = from_namedtuple(t; coords = Dict("interv" => group_name_v), 
							dims = Dict("μ_β" =>["interv"], 
										"μ_η" => ["interv"]))

	ax = plot_violin(idt ; var_names = ["μ_β"], textsize = 12, quartiles = false,
							rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0))

	ax[1].set_ylabel(L"\mu_\beta"; fontsize = 16)
	for i = 1 : length(group_name_v)
		ax[i].set_title(group_name_v[i]; fontsize = 16)
	end

	savefig(string("./figures/", "μ_β_soft_", study_name, ".eps"))


	ax = plot_violin(idt ; var_names = ["μ_η"], textsize = 12, quartiles = false,
						rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0))

	ax[1].set_ylabel(L"\mu_\eta"; fontsize = 16)
	for i = 1 : length(group_name_v)
		ax[i].set_title(group_name_v[i]; fontsize = 16)
	end

	savefig(string("./figures/", "μ_η_soft_", study_name, ".eps"))


	for interv_group in group_name_v[2:end]

		_, ax = plt.subplots(1, 2)
		plot_violin(idt.sel(interv = interv_group).posterior["μ_β"] - idt.sel(interv = group_name_v[1]).posterior["μ_β"];
					textsize = 12, quartiles = false, rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0), ax = ax[1])

		ax[1].set_title(L"\Delta \mu_{\beta}"; fontsize = 16)

		plot_violin(idt.sel(interv = interv_group).posterior["μ_η"] - idt.sel(interv = group_name_v[1]).posterior["μ_η"];
					textsize = 12, quartiles = false, rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0), ax = ax[2])

		ax[2].set_title(L"\Delta \mu_{\eta}"; fontsize = 16)
		savefig(string("./figures/", "Δ_soft_", interv_group, ".eps"))
	end
	show()
end

function plot_trans_violin_lapse(choice_m, data, group_name_v, study_name)
	
	chn = deserialize(string("./chains/chn_lapse_", study_name, ".jls"));

	m = lapse_model(choice_m, data)

	gen = generated_quantities(m, MCMCChains.get_sections(chn, :parameters))

	μ_ε_v = map(x -> x[1], gen[:])
	μ_η_v = map(x -> x[2], gen[:])
	μ_s_v = map(x -> x[3], gen[:])

	t = (μ_ε = [μ_ε_v], μ_η = [μ_η_v], μ_s = [μ_s_v])

	idt = from_namedtuple(t; coords = Dict("interv" => group_name_v), 
							dims = Dict("μ_ε" =>["interv"], 
										"μ_η" => ["interv"],
										"μ_s" => ["interv"]))


	ax = plot_violin(idt; var_names = ["μ_ε"], textsize = 12, quartiles = false,
							rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0))

	ax[1].set_ylabel("mean lapse rate [group level]"; fontsize = 16)

	for i = 1 : length(group_name_v)
		ax[i].set_title(group_name_v[i]; fontsize = 16)
	end

	savefig(string("./figures/", "μ_ε_lapse_", study_name, ".eps"))


	ax = plot_violin(idt; var_names = ["μ_η"], textsize = 12, quartiles = false,
							rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0))

	ax[1].set_ylabel("mean learning rate [group level])"; fontsize = 16)

	for i = 1 : length(group_name_v)
		ax[i].set_title(group_name_v[i]; fontsize = 16)
	end

	savefig(string("./figures/", "μ_η_lapse_", study_name, ".eps"))


	ax = plot_violin(idt; var_names = ["μ_s"], textsize = 12, quartiles = false,
							rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0))

	ax[1].set_ylabel("mean sensitivity [group level]"; fontsize = 16)

	for i = 1 : length(group_name_v)
		ax[i].set_title(group_name_v[i]; fontsize = 16)
	end

	savefig(string("./figures/", "μ_s_lapse_", study_name, ".eps"))


	for interv_group in group_name_v[2:end]

		_, ax = plt.subplots(1, 3)
		plot_violin(idt.sel(interv = interv_group).posterior["μ_ε"] - idt.sel(interv = group_name_v[1]).posterior["μ_ε"];
					textsize = 12, quartiles = false, rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0), ax = ax[1])
		ax[1].set_title("Difference in mean lapse rate"; fontsize = 16)

		plot_violin(idt.sel(interv = interv_group).posterior["μ_η"] - idt.sel(interv = group_name_v[1]).posterior["μ_η"];
					textsize = 12, quartiles = false, rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0), ax = ax[2])
		ax[2].set_title("Difference mean learning rate"; fontsize = 16)

		plot_violin(idt.sel(interv = interv_group).posterior["μ_s"] - idt.sel(interv = group_name_v[1]).posterior["μ_s"];
					textsize = 12, quartiles = false, rug = true, rug_kwargs = Dict("color" => "white", "alpha" => 0), ax = ax[3])
		ax[3].set_title("Difference in mean sensitivity"; fontsize = 16)

		savefig(string("./figures/", "Δ_lapse_", interv_group, ".eps"))
	end
	show()
end

ArviZ.use_style("arviz-whitegrid")

file_v = [["./abt/ER17_FG7142_trials.csv", "./abt/ER17_2vs1_trials.csv"],
			["./abt/SS2_FG7142_trials.csv", "./abt/SS2_2vs1_trials.csv"]]

cb_file_v = [["./abt/ER17_FG7142_counterbalance.csv", "./abt/ER17_2vs1_counterbalance.csv"],
			["./abt/SS2_FG7142_counterbalance.csv", "./abt/SS2_2vs1_counterbalance.csv"]]

group_d = Dict("V" => 1, "FG_0" => 1, "FG_3" => 2, "1" => 1, "2" => 1)

(choice_m, data) = read_data(file_v, cb_file_v, group_d)

plot_trans_violin_softmax(choice_m, data, ["Vehicle", "FG_7142"], "FG")
plot_trans_violin_lapse(choice_m, data, ["Vehicle", "FG_7142"], "FG")
