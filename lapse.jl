
P_softmax(β, r_v) = exp.(β * r_v) ./ sum(exp.(β * r_v))
	
function P_lapse(ε, r_v)

	n_actions = length(r_v)

	if !all(x -> x == r_v[1], r_v)

		return [ε / n_actions for _ = 1:n_actions] + (1.0 - ε) * P_softmax(1.0, r_v)
	else
		return [1.0 / n_actions for _ = 1:n_actions]
	end
end

@model function lapse_model(choice_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}
	
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

		r_v = zeros(T, Int(3*data.n_sessions / 5))

		for session = 1 : data.n_sessions

			avail_actions_v = data.avail_actions_m[subject, session]

			group = data.group_m[subject, session]

			ε = ε_m[group, subject]
			η = η_m[group, subject]
			s = s_m[group, subject]

			for trial = 1 : data.trial_m[subject, session]

				P_v = P_lapse(ε, r_v[avail_actions_v])

				choice_m[subject, session][trial] ~ Binomial(1, P_v[2])

				action = avail_actions_v[choice_m[subject, session][trial] + 1]

				r_v[action] += η * (s * data.R_m[subject, session][trial] - r_v[action])
			end
		end
	end

	return choice_m
end

run_lapse(choice_m, data::ABT_t) = sample(lapse_model(choice_m, data), NUTS(1000, 0.65), MCMCThreads(), 2000, 4)

