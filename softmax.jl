@model function softmax_prl_model(choice_v, data::PRL_t, ::Type{T} = Float64) where {T <: Real}

	if choice_v === missing
		choice_v = Array{Array{Int64,1},1}(undef, data.n_subjects)

		for subject = 1 : data.n_subjects
			choice_v[subject] = [-1 for _ = 1 : data.trial_v[subject]]
		end
	end

	β_upper = 10.0 
	
	μ_β_v ~ filldist(Normal(0,1), data.n_groups)
	σ_β_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	μ_η_v ~ filldist(Normal(0,1), data.n_groups)
	σ_η_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	β_norm_v ~ filldist(Normal(0,1), data.n_subjects)
	η_norm_v ~ filldist(Normal(0,1), data.n_subjects)

	for subject = 1 : data.n_subjects

		r_v = zeros(T, 2)

		g = data.group_v[subject]

		β = cdf(Normal(0,1), μ_β_v[g] + β_norm_v[subject] * σ_β_v[g]) * β_upper
		η = cdf(Normal(0,1), μ_η_v[g] + η_norm_v[subject] * σ_η_v[g])

		for trial = 1 : data.trial_v[subject]
		
			choice_v[subject][trial] ~ BinomialLogit(1, β * (r_v[2] - r_v[1]))

			action = choice_v[subject][trial] + 1
			
			r_v[action] += η * (data.R_v[subject][trial] - r_v[action])
		end
	end

	return (cdf.(Normal(0,1), μ_β_v) * β_upper, cdf.(Normal(0,1), μ_η_v))
end

@model function softmax_2_model(choice_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}

	if choice_m === missing
		choice_m = Matrix{Array{Int64,1}}(undef, data.n_subjects, data.n_sessions)

		for subject = 1 : data.n_subjects
			for session = 1 : data.n_sessions
				choice_m[subject, session] = [-1 for _ = 1 : data.trial_m[subject, session]]
			end
		end
	end

	β_upper = 10.0 
	
	μ_β_v ~ filldist(Normal(0,1), data.n_groups)
	σ_β_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	μ_η_v ~ filldist(Normal(0,1), data.n_groups)
	σ_η_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	β_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)
	η_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)

	β_m = cdf.(Normal(0,1), μ_β_v .+ β_norm_m .* σ_β_v) .* β_upper
	η_m = cdf.(Normal(0,1), μ_η_v .+ η_norm_m .* σ_η_v)

	for subject = 1 : data.n_subjects

		r_v = zeros(T, Int(data.n_avail_actions_per_week * data.n_sessions / data.n_sessions_per_week))

		for session = 1 : data.n_sessions

			avail_actions_v = data.avail_actions_m[subject, session]

			g = data.group_m[subject, session]

			β = β_m[g, subject]
			η = η_m[g, subject]

			for trial = 1 : data.trial_m[subject, session]
			
				choice_m[subject, session][trial] ~ BinomialLogit(1, β * (r_v[avail_actions_v[2]] - r_v[avail_actions_v[1]]))

				action = avail_actions_v[choice_m[subject, session][trial] + 1]
				
				r_v[action] += η * (data.R_m[subject, session][trial] - r_v[action])
			end
		end
	end

	return (cdf.(Normal(0,1), μ_β_v) * β_upper, cdf.(Normal(0,1), μ_η_v))
end

@model function softmax_3_model(choice_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}

	if choice_m === missing
		choice_m = Matrix{Array{Int64,1}}(undef, data.n_subjects, data.n_sessions)

		for subject = 1 : data.n_subjects
			for session = 1 : data.n_sessions
				choice_m[subject, session] = [-1 for _ = 1 : data.trial_m[subject, session]]
			end
		end
	end

	β_upper = 10.0 
	s_upper = 30.0

	μ_β_v ~ filldist(Normal(0,1), data.n_groups)
	σ_β_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	μ_η_v ~ filldist(Normal(0,1), data.n_groups)
	σ_η_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	μ_s_v ~ filldist(Normal(0,1), data.n_groups)
	σ_s_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	β_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)
	η_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)
	s_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)

	β_m = cdf.(Normal(0,1), μ_β_v .+ β_norm_m .* σ_β_v) .* β_upper
	η_m = cdf.(Normal(0,1), μ_η_v .+ η_norm_m .* σ_η_v)
	s_m = cdf.(Normal(0,1), μ_s_v .+ s_norm_m .* σ_s_v) .* s_upper

	for subject = 1 : data.n_subjects

		r_v = zeros(T, Int(data.n_avail_actions_per_week * data.n_sessions / data.n_sessions_per_week))

		for session = 1 : data.n_sessions

			avail_actions_v = data.avail_actions_m[subject, session]

			g = data.group_m[subject, session]

			β = β_m[g, subject]
			η = η_m[g, subject]
			s = s_m[g, subject]

			for trial = 1 : data.trial_m[subject, session]
			
				choice_m[subject, session][trial] ~ BinomialLogit(1, β * (r_v[avail_actions_v[2]] - r_v[avail_actions_v[1]]))

				action = avail_actions_v[choice_m[subject, session][trial] + 1]
				
				r_v[action] += η * (s * data.R_m[subject, session][trial] - r_v[action])
			end
		end
	end

	return (cdf.(Normal(0,1), μ_β_v) * β_upper, cdf.(Normal(0,1), μ_η_v), cdf.(Normal(0,1), μ_s_v) * s_upper)
end

function predict_softmax_2(choice_m, data, chn)

	rng = MersenneTwister()

	(n_samples, n_groups, n_chains) = size(group(chn, :μ_β_v).value)

	β_upper = 10.0 
	n_MC = 100
	l = 0.0

	for c = 1 : n_chains
		for s = 1 : n_samples 

			μ_β_v = group(chn, :μ_β_v).value[s, :, c]
			σ_β_v = group(chn, :σ_β_v).value[s, :, c]
			μ_η_v = group(chn, :μ_η_v).value[s, :, c]
			σ_η_v = group(chn, :σ_η_v).value[s, :, c]

			l_s = 0.0

			for k = 1 : n_MC
				ll_MC = 0.0
				for subject = 1 : data.n_subjects

					r_v = zeros(Int(data.n_avail_actions_per_week * data.n_sessions / data.n_sessions_per_week))

					for session = 1 : data.n_sessions

						avail_actions_v = data.avail_actions_m[subject, session]

						g = data.group_m[subject, session]

						β_norm = rand(rng, Normal(0,1))
						η_norm = rand(rng, Normal(0,1))

						β = cdf(Normal(0,1), μ_β_v[g] + β_norm * σ_β_v[g]) * β_upper
						η = cdf(Normal(0,1), μ_η_v[g] + η_norm * σ_η_v[g])

						for trial = 1 : data.trial_m[subject, session]
						
							ll_MC += logpdf(BinomialLogit(1, β * (r_v[avail_actions_v[2]] - r_v[avail_actions_v[1]])), 
											choice_m[subject, session][trial])

							action = avail_actions_v[choice_m[subject, session][trial] + 1]
							
							r_v[action] += η * (data.R_m[subject, session][trial] - r_v[action])
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

run_softmax(choice_m, data::ABT_t) = sample(softmax_2_model(choice_m, data), 
											NUTS(1000, 0.65), MCMCThreads(), 2000, 4)

run_softmax_prl(choice_v, data::PRL_t) = sample(softmax_prl_model(choice_v, data), 
												NUTS(1000, 0.65), MCMCThreads(), 2000, 4)
