
@model function softmax_model(action_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}

	if action_m === missing
		action_m = Matrix{Array{Int64,1}}(undef, data.n_subjects, data.n_sessions)
		action_m = [ [-1 for _ = 1:data.n_trials] for _ = 1:data.n_subjects for _ = 1:data.n_sessions ]
		action_m = reshape(action_m, (data.n_subjects, data.n_sessions))
	end

	β_upper = 10.0 
	
	μ_β ~ Normal(0,1)
	σ_β ~ truncated(Cauchy(0,5), 0, Inf)

	μ_η ~ Normal(0,1)
	σ_η ~ truncated(Cauchy(0,5), 0, Inf)

	β_norm_v ~ filldist(Normal(0,1), data.n_subjects)
	η_norm_v ~ filldist(Normal(0,1), data.n_subjects)

	β_v = cdata.(Normal(0,1), μ_β .+ σ_β * β_norm_v) * β_upper
	η_v = cdata.(Normal(0,1), μ_η .+ σ_η * η_norm_v)

	for subject = 1 : data.n_subjects

		r_v = zeros(T, Int(3*data.n_sessions / 5))

		β = β_v[subject]
		η = η_v[subject]

		for session = 1 : data.n_sessions

			avail_actions_v = data.avail_actions_m[subject, session]

			for trial = 1 : data.n_trials
				
				action_m[subject, session][trial] ~ BinomialLogit(1, β * (r_v[avail_actions_v[2]] - r_v[avail_actions_v[1]]))
				
				action = avail_actions_v[action_m[subject, session][trial] + 1]

				r_v[action] += η * (data.R_m[subject, session][trial] - r_v[action])
			end
		end	
	end

	return (action_m, μ_β, σ_β, μ_η, σ_η, β_v, η_v)
end

function run_softmax()

	n_sessions = 10
	n_subjects = 20
	n_trials = 20

	avail_actions_m = repeat([[1,2], [1,3], [1,2], [1,3], [2,3], [4,5], [4,6], [4,5], [4,6], [5,6]], 1, n_subjects)

	r_env_v = [0.0, 1.0, 1.0, 0.0, 1.0, 2.0]

	data = ABT_t(n_sessions, n_subjects, n_trials, avail_actions_m, r_env_v)

	prior = softmax_model(missing, data)
	
	(action_m, μ_β, σ_β, μ_η, σ_η, ) = prior()

	chn = sample(softmax_model(action_m, data), NUTS(1000, 0.65), MCMCThreads(), 2000, 4)

	@show μ_β
	@show σ_β
	@show μ_η
	@show σ_η

	return chn
end
