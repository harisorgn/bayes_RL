
P_softmax(β, r_v) = exp.(β * r_v) ./ sum(exp.(β * r_v))
	
function P_ε_soft(ε, r_v)

	n_actions = length(r_v)

	if !all(x -> x == r_v[1], r_v)
		
		return [ε / n_actions for _ = 1:n_actions] + (1.0 - ε) * P_softmax(1.0, r_v)
	else
		return [1.0 / n_actions for _ = 1:n_actions]
	end
end

@model function ε_soft_2_model(choice_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}
	
	if choice_m === missing
		choice_m = Matrix{Array{Int64,1}}(undef, data.n_subjects, data.n_sessions)

		for subject = 1 : data.n_subjects
			for session = 1 : data.n_sessions
				choice_m[subject, session] = [-1 for _ = 1 : data.trial_m[subject, session]]
			end
		end
	end
	
	μ_ε_v ~ filldist(Normal(0,1), data.n_groups)
	σ_ε_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	μ_η_v ~ filldist(Normal(0,1), data.n_groups)
	σ_η_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	ε_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)
	η_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)

	ε_m = cdf.(Normal(0,1), μ_ε_v .+ ε_norm_m .* σ_ε_v)
	η_m = cdf.(Normal(0,1), μ_η_v .+ η_norm_m .* σ_η_v)

	for subject = 1 : data.n_subjects

		r_v = zeros(T, Int(data.n_avail_actions_per_week * data.n_sessions / data.n_sessions_per_week))

		for session = 1 : data.n_sessions

			avail_actions_v = data.avail_actions_m[subject, session]

			g = data.group_m[subject, session]

			ε = ε_m[g, subject]
			η = η_m[g, subject]

			for trial = 1 : data.trial_m[subject, session]

				P_v = P_ε_soft(ε, r_v[avail_actions_v])

				choice_m[subject, session][trial] ~ Binomial(1, P_v[2])

				action = avail_actions_v[choice_m[subject, session][trial] + 1]

				r_v[action] += η * (data.R_m[subject, session][trial] - r_v[action])
			end
		end
	end

	return (cdf.(Normal(0,1), μ_ε_v), cdf.(Normal(0,1), μ_η_v))
	#return (choice_m, μ_ε_v, σ_ε_v, μ_η_v, σ_η_v, μ_s_v, σ_s_v)
end

@model function ε_soft_3_model(choice_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}
	
	if choice_m === missing
		choice_m = Matrix{Array{Int64,1}}(undef, data.n_subjects, data.n_sessions)

		for subject = 1 : data.n_subjects
			for session = 1 : data.n_sessions
				choice_m[subject, session] = [-1 for _ = 1 : data.trial_m[subject, session]]
			end
		end
	end

	s_upper = 30.0 

	μ_ε_v ~ filldist(Normal(0,1), data.n_groups)
	σ_ε_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	μ_η_v ~ filldist(Normal(0,1), data.n_groups)
	σ_η_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	μ_s_v ~ filldist(Normal(0,1), data.n_groups)
	σ_s_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	ε_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)
	η_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)
	s_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)

	ε_m = cdf.(Normal(0,1), μ_ε_v .+ ε_norm_m .* σ_ε_v)
	η_m = cdf.(Normal(0,1), μ_η_v .+ η_norm_m .* σ_η_v)
	s_m = cdf.(Normal(0,1), μ_s_v .+ s_norm_m .* σ_s_v) * s_upper

	for subject = 1 : data.n_subjects

		r_v = zeros(T, Int(data.n_avail_actions_per_week * data.n_sessions / data.n_sessions_per_week))

		for session = 1 : data.n_sessions

			avail_actions_v = data.avail_actions_m[subject, session]

			g = data.group_m[subject, session]

			ε = ε_m[g, subject]
			η = η_m[g, subject]
			s = s_m[g, subject]

			for trial = 1 : data.trial_m[subject, session]

				P_v = P_ε_soft(ε, r_v[avail_actions_v])

				choice_m[subject, session][trial] ~ Binomial(1, P_v[2])

				action = avail_actions_v[choice_m[subject, session][trial] + 1]

				r_v[action] += η * (s * data.R_m[subject, session][trial] - r_v[action])
			end
		end
	end

	return (cdf.(Normal(0,1), μ_ε_v), cdf.(Normal(0,1), μ_η_v), cdf.(Normal(0,1), μ_s_v) * s_upper)
	#return (choice_m, μ_ε_v, σ_ε_v, μ_η_v, σ_η_v, μ_s_v, σ_s_v)
end

function predict_ε_soft_3(choice_m, data, chn)

	rng = MersenneTwister()

	(n_samples, n_groups, n_chains) = size(group(chn, :μ_ε_v).value)

	s_upper = 30.0 
	n_MC = 100
	l = 0.0

	for c = 1 : n_chains
		for s = 1 : n_samples 

			μ_ε_v = group(chn, :μ_ε_v).value[s, :, c]
			σ_ε_v = group(chn, :σ_ε_v).value[s, :, c]
			μ_η_v = group(chn, :μ_η_v).value[s, :, c]
			σ_η_v = group(chn, :σ_η_v).value[s, :, c]
			μ_s_v = group(chn, :μ_s_v).value[s, :, c]
			σ_s_v = group(chn, :σ_s_v).value[s, :, c]

			l_s = 0.0

			for k = 1 : n_MC
				ll_MC = 0.0
				for subject = 1 : data.n_subjects

					r_v = zeros(Int(data.n_avail_actions_per_week * data.n_sessions / data.n_sessions_per_week))

					for session = 1 : data.n_sessions

						avail_actions_v = data.avail_actions_m[subject, session]

						g = data.group_m[subject, session]

						ε_norm = rand(rng, Normal(0,1))
						η_norm = rand(rng, Normal(0,1))
						s_norm = rand(rng, Normal(0,1))

						ε = cdf(Normal(0,1), μ_ε_v[g] + ε_norm * σ_ε_v[g])
						η = cdf(Normal(0,1), μ_η_v[g] + η_norm * σ_η_v[g])
						s = cdf(Normal(0,1), μ_s_v[g] + η_norm * σ_η_v[g]) * s_upper

						for trial = 1 : data.trial_m[subject, session]
							
							P_v = P_ε_soft(ε, r_v[avail_actions_v])

							ll_MC += logpdf(Binomial(1, P_v[2]), choice_m[subject, session][trial])

							action = avail_actions_v[choice_m[subject, session][trial] + 1]
							
							r_v[action] += η * (s * data.R_m[subject, session][trial] - r_v[action])
						end
					end
				end
				l_s += exp(ll_MC)
			end

			l += l_s / n_MC
		end
	end
	return log(l / (n_samples * n_chains))
end

run_ε_soft(choice_m, data::ABT_t) = sample(ε_soft_3_model(choice_m, data), NUTS(1000, 0.65), MCMCThreads(), 2000, 4)