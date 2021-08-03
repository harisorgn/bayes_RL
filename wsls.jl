
@model function wsls_model(choice_v, trial_v, R_v, n_subjects, ::Type{T} = Float64) where {T <: Real}

	if choice_v === missing
		choice_v = [[-1 for _ = 1:trial_v[subject]] for subject = 1:n_subjects]
	end
	
	μ_p ~ Normal(0,1)
	σ_p ~ truncated(Cauchy(0,5), 0, Inf)

	μ_b ~ Normal(0,1)
	σ_b ~ truncated(Cauchy(0,5), 0, Inf)

	p_norm_v ~ filldist(Normal(0,1), n_subjects)
	b_v ~ filldist(Normal(μ_b, σ_b), n_subjects)

	p_v = cdf.(Normal(0,1), μ_p .+ p_norm_v .* σ_p)

	for subject = 1 : n_subjects

		p_stay = p_v[subject]
		b = b_v[subject]

		prev_choice = choice_v[subject][1]
		prev_R = R_v[subject][1]

		for trial = 2 : trial_v[subject]

			current_choice = choice_v[subject][trial]

			p = prev_choice == 0 ? prev_R*(1.0 - p_stay) + (1.0 - prev_R)*(1.0/(1.0 + exp(-b))) :
									prev_R*p_stay + (1.0 - prev_R)*(1.0/(1.0 + exp(-b)))

			choice_v[subject][trial] ~ Binomial(1, p)
			
			prev_choice = current_choice
			prev_R = R_v[subject][trial]
		end
	end
end


@model function wsls_model(choice_m, data::ABT_t, ::Type{T} = Float64) where {T <: Real}

	if choice_m === missing
		choice_m = Matrix{Array{Int64,1}}(undef, data.n_subjects, data.n_sessions)

		for subject = 1 : data.n_subjects
			for session = 1 : data.n_sessions
				choice_m[subject, session] = [-1 for _ = 1 : data.trial_m[subject, session]]
			end
		end
	end
	
	μ_p ~ Normal(0,1)
	σ_p ~ truncated(Cauchy(0,5), 0, Inf)

	μ_b_v ~ filldist(Normal(0,1), data.n_groups)
	σ_b_v ~ filldist(truncated(Cauchy(0,5), 0, Inf), data.n_groups)

	p_norm_v ~ filldist(Normal(0,1), data.n_subjects)
	b_norm_m ~ filldist(Normal(0,1), data.n_groups, data.n_subjects)

	p_v = cdf.(Normal(0,1), μ_p .+ p_norm_v .* σ_p)
	b_m = μ_b_v .+ b_norm_m .* σ_b_v

	for subject = 1 : data.n_subjects

		p_stay = p_v[subject]

		for session = 1 : data.n_sessions

			group = data.group_m[subject, session]
			b = b_m[group, subject]

			prev_choice = choice_m[subject, session][1]
			prev_R = data.R_m[subject, session][1]

			for trial = 2 : data.trial_m[subject, session]

				current_choice = choice_m[subject, session][trial]

				p = (1.0 - prev_choice)*(prev_R*(1.0 - p_stay) + (1.0 - prev_R)*(1.0/(1.0 + exp(-b)))) +
							prev_choice*(prev_R*p_stay + (1.0 - prev_R)*(1.0/(1.0 + exp(-b))))

				choice_m[subject, session][trial] ~ Binomial(1, p)
				
				prev_choice = current_choice
				prev_R = data.R_m[subject, session][trial]
			end
		end
	end
end

run_wsls(choice_m, data::ABT_t) = sample(wsls_model(choice_m, data), NUTS(1000, 0.85), MCMCThreads(), 2000, 4)
