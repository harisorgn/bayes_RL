struct lapse_t
	α_ε::Float64
	β_ε::Float64
	α_η::Float64
	β_η::Float64
	ε_v::Array{Float64, 1}
	η_v::Array{Float64, 1}
end

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

	μ_ε ~ Normal(0,1)
	σ_ε ~ truncated(Cauchy(0,5), 0, Inf)

	μ_η ~ Normal(0,1)
	σ_η ~ truncated(Cauchy(0,5), 0, Inf)

	μ_s ~ Normal(0,1)
	σ_s ~ truncated(Cauchy(0,5), 0, Inf)

	ε_norm_v ~ filldist(Normal(0,1), n_subjects)
	η_norm_v ~ filldist(Normal(0,1), n_subjects)
	s_norm_v ~ filldist(Normal(0,1), n_subjects)

	ε_v = cdata.(Normal(0,1), μ_ε .+ σ_ε * ε_norm_v)
	η_v = cdata.(Normal(0,1), μ_η .+ σ_η * η_norm_v)
	s_v = cdata.(Normal(0,1), μ_s .+ σ_s * s_norm_v) * s_upper

	for subject = 1 : n_subjects

		r_v = zeros(T, Int(3*data.n_sessions / 5))

		ε = ε_v[subject]
		η = η_v[subject]
		s = s_v[subject]

		for session = 1 : n_sessions

			avail_actions_v = avail_actions_m[subject, session]

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

function run_lapse()

	n_sessions = 5
	n_subjects = 20
	n_trials = 20

	avail_actions_m = repeat([[1,3], [2,3], [1,3], [2,3], [1,2]], 1, n_subjects)

	r_env_v = [1.0, 1.0, 0.0]

	prior = lapse_model(missing, avail_actions_m, r_env_v, n_sessions, n_subjects, n_trials)

	(action_m, s_v) = prior()
	println(s_v)
	chn = sample(lapse_model(action_m, avail_actions_m, r_env_v, n_sessions, n_subjects, n_trials), 
				HMC(0.01, 20), MCMCThreads(), 2000, 4)

	return chn
end
