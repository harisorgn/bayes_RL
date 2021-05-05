
@model function softmax_model(action_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}

	if action_m === missing
		action_m = Matrix{Array{Int64,1}}(undef, data.n_subjects, data.n_sessions)
		action_m = [ [-1 for _ = 1:data.n_trials] for _ = 1:data.n_subjects for _ = 1:data.n_sessions ]
		action_m = reshape(action_m, (data.n_subjects, data.n_sessions))
	end

	β_upper = 10.0 
	
	μ_β_v ~ filldist(Normal(0,1), data.n_interv)
	σ_β_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_interv)

	μ_η_v ~ filldist(Normal(0,1), data.n_interv)
	σ_η_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_interv)

	β_norm_m ~ filldist(Normal(0,1), data.n_subjects, data.n_interv)
	η_norm_m ~ filldist(Normal(0,1), data.n_subjects, data.n_interv)

	β_m = cdf.(Normal(0,1), μ_β_v .+ β_norm_m * σ_β_v) * β_upper
	η_m = cdf.(Normal(0,1), μ_η_v .+ η_norm_v * σ_η_v)

	for subject = 1 : data.n_subjects

		r_v = zeros(T, Int(3*data.n_sessions / 5))

		for session = 1 : data.n_sessions

			avail_actions_v = data.avail_actions_m[subject, session]

			interv = data.interv_m[subject, session]

			β = β_m[subject, interv]
			η = η_m[subject, interv]

			for trial = 1 : data.trial_m[subject, session]
				
				action_m[subject, session][trial] ~ BinomialLogit(1, β * (r_v[avail_actions_v[2]] - r_v[avail_actions_v[1]]))
				
				action = avail_actions_v[action_m[subject, session][trial] + 1]

				r_v[action] += η * (data.R_m[subject, session][trial] - r_v[action])
			end
		end	
	end

	return (action_m, μ_β, σ_β, μ_η, σ_η, β_v, η_v)
end

run_softmax(action_m, data::ABT_t) = sample(softmax_model(action_m, data), NUTS(1000, 0.65), MCMCThreads(), 2000, 4)

