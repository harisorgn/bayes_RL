
@model function softmax_model(choice_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}

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

		r_v = zeros(T, Int(3*data.n_sessions / 5))

		for session = 1 : data.n_sessions

			avail_actions_v = data.avail_actions_m[subject, session]

			group = data.group_m[subject, session]

			β = β_m[group, subject]
			η = η_m[group, subject]

			for trial = 1 : data.trial_m[subject, session]

				choice_m[subject, session][trial] ~ BinomialLogit(1, β * (r_v[avail_actions_v[2]] - r_v[avail_actions_v[1]]))

				action = avail_actions_v[choice_m[subject, session][trial] + 1]
				
				r_v[action] += η * (data.R_m[subject, session][trial] - r_v[action])
			end
		end
	end

	return choice_m
end

run_softmax(choice_m, data::ABT_t) = sample(softmax_model(choice_m, data), NUTS(1000, 0.65), MCMCThreads(), 2000, 4)

