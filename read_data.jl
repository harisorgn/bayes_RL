using CSV

parse_tuple(str) = Tuple(split(str[2 : end - 1], ','))

function map_session(day, week_idx, n_sessions_per_week)

	if day == "test"
		return n_sessions_per_week + (week_idx - 1)*n_sessions_per_week
	else
		return parse(Int, day[end]) + (week_idx - 1)*n_sessions_per_week
	end
end

function cb_map_functions(cb_file, group_d)

	df = CSV.File(cb_file) |> DataFrame

	choice_d = Dict("A" => 1, "B" => 2, "blank" => 3)

	interv_name = split(cb_file, '_')[2]

	test_reward_d = Dict{String, Dict{String, Float64}}()

	for ID in df.ID
		if interv_name == "2vs1"

			(avail_action_1, reward_magnitude_1) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== 1), :PD1][1])
			(avail_action_2, reward_magnitude_2) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== 1), :PD2][1])

			test_reward_d[ID] = Dict(avail_action_1 => parse(Float64, reward_magnitude_1),
									avail_action_2 => parse(Float64, reward_magnitude_2),
									"" => 0.0)
		else
			test_reward_d[ID] = Dict("A" => 1.0, "B" => 1.0, "" => 0.0)
		end
	end

	function group(ID, day, week)

		if day == "test"

			if "V" in keys(group_d)
				return group_d["V"]
			elseif "1" in keys(group_d)
				return group_d["1"]
			else
				println("Error: Vehicle/Control/Test-day intervention key not provided in group_dictionary")
				return -1
			end

		else

			(~, interv) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			if interv in keys(group_d)
				return group_d[interv]
			else
				return -1
			end
		end
	end

	function avail_actions(ID, day, week, week_idx, n_avail_actions_per_week)

		if day == "test"

			(avail_action_1, interv_1) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), :PD1][1])
			(avail_action_2, interv_2) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), :PD2][1])

			# vehicle-paired substrate (or 1-paired in 2vs1) is always the first available action in test sessions
			# and drug is second
			if interv_1 == "1" || interv_1 == "V"

				return [choice_d[avail_action_1] + (week_idx - 1)*n_avail_actions_per_week, 
						choice_d[avail_action_2] + (week_idx - 1)*n_avail_actions_per_week]
			else

				return [choice_d[avail_action_2] + (week_idx - 1)*n_avail_actions_per_week, 
						choice_d[avail_action_1] + (week_idx - 1)*n_avail_actions_per_week]		
			end		

		else

			(avail_action,) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			return [choice_d["blank"] + (week_idx - 1)*n_avail_actions_per_week, 
					choice_d[avail_action] + (week_idx - 1)*n_avail_actions_per_week]
		end
	end

	function choices(ID, day, week, choice_v)
		
		# choices need to be {0,1} to agree with Binomial samples
		# vehicle-paired substrate is first, needs to agree with avail_actions
		if day == "test" 

			(avail_action_1, interv_1) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), :PD1][1])
			(avail_action_2, interv_2) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), :PD2][1])
			
			choice_idx_d = (interv_1 == "1" || interv_1 == "V") ? Dict(avail_action_1 => 0, avail_action_2 => 1) : 
																Dict(avail_action_2 => 0, avail_action_1 => 1)

			return map(choice -> choice_idx_d[choice], choice_v)
		else 
			return parse.(Int, choice_v)
		end
	end

	function rewards(ID, day, week, choice_v)

		test_rewarded_choices_v = ["A", "B", "", "B", "", "A", "", 
									"A", "B", "A", "B", "", "B", "", 
									"A", "", "A", "B", "A", "B", "", 
									"B", "", "A", "", "A", "B", "A", 
									"B", ""]

		if day == "test"

			return map((choice, rewarded_choice) -> choice == rewarded_choice ? test_reward_d[ID][choice] : 0.0, choice_v, test_rewarded_choices_v)
		else

			(avail_action, ~) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			return parse.(Float64, choice_v) * test_reward_d[ID][avail_action]
		end
	end

	return (group, avail_actions, choices, rewards)
end

function count_sessions(cb_file, group_v, n_sessions_per_week)

	df = CSV.File(cb_file) |> DataFrame

	interv_v = vcat(getindex.(parse_tuple.(df.PD1),2), getindex.(parse_tuple.(df.PD2),2))

	n_subjects = length(unique(df.ID))

	n_weeks = Int(count(x -> x in group_v, interv_v) / n_subjects)

	return n_weeks * n_sessions_per_week
end

function read_ABT(file_v, cb_file_v, group_d ; only_test = false)

	CHOICE_V = Matrix[]
	AVAIL_ACTIONS_V = Matrix[]
	GROUP_V = Matrix[]
	R_V = Matrix[]
	TRIAL_V = Matrix[]

	interv_v = filter(x -> x != "V" && x != "1", keys(group_d))

	if only_test 
		n_sessions_per_week = 1
		n_avail_actions_per_week = 2
	else
		n_sessions_per_week = 5
		n_avail_actions_per_week = 3
	end

	n_sessions = length(interv_v) * n_sessions_per_week
	n_subjects = 0

	for (batch_file_v, batch_cb_file_v) in zip(file_v, cb_file_v)

	@assert length(batch_file_v) == length(batch_cb_file_v)
	
	df = CSV.File(batch_file_v[1]) |> DataFrame

	n_subjects_batch = length(unique(df.ID))

	choice_m = Matrix{Array{Int64,1}}(undef, n_subjects_batch, n_sessions)
	avail_actions_m = Matrix{Array{Int64,1}}(undef, n_subjects_batch, n_sessions)
	group_m = Matrix{Int64}(undef, n_subjects_batch, n_sessions)
	R_m = Matrix{Array{Float64,1}}(undef, n_subjects_batch, n_sessions)
	trial_m = Matrix{Int64}(undef, n_subjects_batch, n_sessions)

	offset_weeks = 0
	offset_sessions = 0
	offset_actions = 0

	for (file, cb_file) in zip(batch_file_v, batch_cb_file_v)

		df = CSV.File(file) |> DataFrame

		df = filter(x -> !ismissing(x.Choice) && x.Choice != "O", df)

		if only_test
			df = df[df.Day .== "test", :]
		end

		ID_v = unique(df.ID)
		day_v = unique(df.Day)
		week_v = unique(df.Week)

		(group_f, avail_actions_f, choices_f, rewards_f) = cb_map_functions(cb_file, group_d)

		for ID in ID_v

			ID_number = parse(Int64, ID[findlast('_', ID)+1 : end])

			week_idx = 0

			for week in week_v

				if group_f(ID, "PD1", week) != -1 && 
					group_f(ID, "PD2", week) != -1 &&
					group_f(ID, "PD3", week) != -1 &&
					group_f(ID, "PD4", week) != -1 

					week_idx += 1

					for day in day_v

						session = map_session(day, week_idx, n_sessions_per_week) + offset_sessions

						group_m[ID_number, session] = group_f(ID, day, week)

						choice_m[ID_number, session] = choices_f(ID, day, week,
																df[(df.ID .== ID) .& 
																	(df.Day .== day) .& 
																	(df.Week .== week), 
																	:Choice])

						avail_actions_m[ID_number, session] = avail_actions_f(ID, day, week, week_idx, n_avail_actions_per_week) .+ 
																offset_actions

						R_m[ID_number, session] = rewards_f(ID, day, week, 
															df[(df.ID .== ID) .& 
																(df.Day .== day) .& 
																(df.Week .== week), 
																:Choice])

						trial_m[ID_number, session] = length(df[(df.ID .== ID) .& 
																(df.Day .== day) .& 
																(df.Week .== week), 
																:Choice])
					end
				end
			end

			offset_weeks = week_idx
		end

		offset_actions += offset_weeks * n_avail_actions_per_week
		offset_sessions += count_sessions(cb_file, interv_v, n_sessions_per_week)
	end
	push!(CHOICE_V, choice_m)
	push!(AVAIL_ACTIONS_V, avail_actions_m)
	push!(GROUP_V, group_m)
	push!(R_V, R_m)
	push!(TRIAL_V, trial_m)

	n_subjects += n_subjects_batch
	end
	
	choice_m = reduce(vcat, CHOICE_V)
	avail_actions_m = reduce(vcat, AVAIL_ACTIONS_V)
	group_m = reduce(vcat, GROUP_V)
	R_m = reduce(vcat, R_V)
	trial_m = reduce(vcat, TRIAL_V)

	return (choice_m, ABT_t(n_sessions, 
							n_subjects, 
							length(unique(group_m)), 
							n_sessions_per_week, 
							n_avail_actions_per_week, 
							avail_actions_m, 
							group_m, 
							R_m, 
							trial_m))
end

function read_PRL(file, group_d)

	choice_d = Dict("A" => 0, "B" => 1)

	df = CSV.File(file) |> DataFrame
	df = filter(x -> !ismissing(x.Choice) && x.Choice != "O", df)

	ID_v = unique(df.ID)
	n_subjects = length(ID_v)
	n_groups = length(keys(group_d))

	choice_v = Array{Array{Int64,1},1}(undef, n_subjects)
	group_v = Array{Int64,1}(undef, n_subjects)
	R_v = Array{Array{Float64,1},1}(undef, n_subjects)
	trial_v = Array{Int64,1}(undef, n_subjects)

	i = 1
	for ID in ID_v
		choice_v[i] = map(x -> choice_d[x], df.Choice[df.ID .== ID])

		batch_ID = ID[1:findfirst('_', ID)-1]
		group_v[i] = group_d[batch_ID]

		R_v[i] = df.Reward[df.ID .== ID]

		trial_v[i] = length(R_v[i])

		i += 1 
	end

	return (choice_v, PRL_t(n_subjects, n_groups, group_v, R_v, trial_v))
end



