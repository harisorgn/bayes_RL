
P_softmax(β, r_v) = exp.(β * r_v) ./ sum(exp.(β * r_v))
	
function P_lapse(ε, r_v)

	n_actions = length(r_v)

	if !all(x -> x == r_v[1], r_v)

		return [ε / n_actions for _ = 1:n_actions] + (1.0 - ε) * P_softmax(1.0, r_v)
	else
		return [1.0 / n_actions for _ = 1:n_actions]
	end
end

@model function lapse_model(action_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}
	
	if action_m === missing
		action_m = Matrix{Array{Int64,1}}(undef, data.n_subjects, data.n_sessions)
		action_m = [ [-1 for _ = 1:data.n_trials] for _ = 1:data.n_subjects for _ = 1:data.n_sessions ]
		action_m = reshape(action_m, (data.n_subjects, data.n_sessions))
	end

	s_upper = 30.0 

	μ_ε_v ~ filldist(Normal(0,1), data.n_interv)
	σ_ε_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_interv)

	μ_η_v ~ filldist(Normal(0,1), data.n_interv)
	σ_η_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_interv)

	μ_s_v ~ filldist(Normal(0,1), data.n_interv)
	σ_s_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_interv)

	ε_norm_m ~ filldist(Normal(0,1), data.n_subjects, data.n_interv)
	η_norm_m ~ filldist(Normal(0,1), data.n_subjects, data.n_interv)
	s_norm_m ~ filldist(Normal(0,1), data.n_subjects, data.n_interv)

	ε_m = cdf.(Normal(0,1), μ_ε_v .+ ε_norm_m * σ_ε_v)
	η_m = cdf.(Normal(0,1), μ_η_v .+ η_norm_m * σ_η_v)
	s_m = cdf.(Normal(0,1), μ_s_v .+ s_norm_m * σ_s_v) * s_upper

	for subject = 1 : n_subjects

		r_v = zeros(T, Int(3*data.n_sessions / 5))

		for session = 1 : n_sessions

			avail_actions_v = avail_actions_m[subject, session]

			interv = data.interv_m[subject, session]

			ε = ε_m[subject, interv]
			η = η_m[subject, interv]
			s = s_m[subject, interv]

			for trial = 1 : n_trials

				P_v = P_lapse(ε, r_v[avail_actions_v])

				action_m[subject, session][trial] ~ Categorical(P_v)

				action = avail_actions_v[action_m[subject, session][trial]]

				r_v[action] += η * (s * data.R_m[subject, session][trial] - r_v[action])
			end
		end
	end

	return (action_m, μ_ε, σ_ε, μ_η, σ_η, μ_s, σ_s, ε_v, η_v, s_v)
end

run_lapse(action_m, data::ABT_t) = sample(lapse_model(action_m, data), NUTS(1000, 0.65), MCMCThreads(), 2000, 4)

