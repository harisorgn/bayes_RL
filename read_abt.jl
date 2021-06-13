using CSV
using DataFrames

parse_tuple(str) = Tuple(split(str[2 : end - 1], ','))

function map_session(day, week)

	if day == "test"
		return 5 + (week - 1)*5
	else
		return parse(Int, day[end]) + (week - 1)*5
	end
end

function cb_map_functions(cb_file, group_d)

	df = CSV.File(cb_file) |> DataFrame

	choice_d = Dict("A" => 1, "B" => 2, "blank" => 3)

	interv_name = split(cb_file, '_')[2]

	test_reward_d = Dict{String, Dict{String, Float64}}()

	for ID in df.ID
		if interv_name == "2vs1"

			(choice_1, reward_magnitude_1) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== 1), :PD1][1])
			(choice_2, reward_magnitude_2) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== 1), :PD2][1])

			test_reward_d[ID] = Dict(choice_1 => parse(Float64, reward_magnitude_1),
									choice_2 => parse(Float64, reward_magnitude_2),
									"" => 0.0)
		else
			test_reward_d[ID] = Dict("A" => 1.0, "B" => 1.0, "" => 0.0)
		end
	end

	function group(ID, day, week)

		if day == "test"

			return group_d["V"]

		else

			(~, interv) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			if interv in keys(group_d)
				return group_d[interv]
			else
				return -1
			end
		end
	end

	function avail_actions(ID, day, week, week_idx)

		if day == "test"

			return [choice_d["A"] + (week_idx - 1)*3, choice_d["B"] + (week_idx - 1)*3]

		else

			(avail_action,) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			return [choice_d["blank"] + (week_idx - 1)*3, 
					choice_d[avail_action] + (week_idx - 1)*3]
		end
	end

	function choices(day, choice_v)
		
		if day == "test" 
			# choices need to be {0,1} to agree with Binomial samples
			# that is why -1 is added
			return map(choice -> choice_d[choice] - 1, choice_v)
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

function count_sessions(cb_file, group_v)

	df = CSV.File(cb_file) |> DataFrame

	interv_v = vcat(getindex.(parse_tuple.(df.PD1),2), getindex.(parse_tuple.(df.PD2),2))

	n_subjects = length(unique(df.ID))

	n_weeks = Int(count(x -> x in group_v, interv_v) / n_subjects)

	return n_weeks * 5
end

function read_data(file_v, cb_file_v, group_d)

	batch_ID_v = String[]
	offset_ID = 0

	CHOICE_V = Matrix[]
	AVAIL_ACTIONS_V = Matrix[]
	GROUP_V = Matrix[]
	R_V = Matrix[]
	TRIAL_V = Matrix[]

	n_subjects = 0
	n_sessions = 0

	for (batch_file_v, batch_cb_file_v) in zip(file_v, cb_file_v)

	@assert length(batch_file_v) == length(batch_cb_file_v)

	interv_v = filter(x -> x != "V" && x != "1", keys(group_d))
	
	df = CSV.File(batch_file_v[1]) |> DataFrame
	n_subjects_batch = length(unique(df.ID))
	n_sessions_batch = length(interv_v) * 5
	batch_ID = split(df.ID[1], "_")[1]

	choice_m = Matrix{Array{Int64,1}}(undef, n_subjects_batch, n_sessions_batch)
	avail_actions_m = Matrix{Array{Int64,1}}(undef, n_subjects_batch, n_sessions_batch)
	group_m = Matrix{Int64}(undef, n_subjects_batch, n_sessions_batch)
	R_m = Matrix{Array{Float64,1}}(undef, n_subjects_batch, n_sessions_batch)
	trial_m = Matrix{Int64}(undef, n_subjects_batch, n_sessions_batch)

	offset_weeks = 0
	offset_sessions = 0
	offset_actions = 0

	for (file, cb_file) in zip(batch_file_v, batch_cb_file_v)

		df = CSV.File(file) |> DataFrame

		df = filter(x -> !ismissing(x.Choice) && x.Choice != "O", df)

		ID_v = unique(df.ID)
		day_v = unique(df.Day)
		week_v = unique(df.Week)

		(group_f, avail_actions_f, choices_f, rewards_f) = cb_map_functions(cb_file, group_d)

		for ID in ID_v

			ID_number = parse(Int64, ID[findlast('_', ID)+1 : end]) + offset_ID

			week_idx = 0

			for week in week_v

				if group_f(ID, "PD1", week) != -1 && 
					group_f(ID, "PD2", week) != -1 &&
					group_f(ID, "PD3", week) != -1 &&
					group_f(ID, "PD4", week) != -1 

					week_idx += 1

					for day in day_v

						session = map_session(day, week_idx) + offset_sessions

						group_m[ID_number, session] = group_f(ID, day, week)

						choice_m[ID_number, session] = choices_f(day, 
																df[(df.ID .== ID) .& 
																	(df.Day .== day) .& 
																	(df.Week .== week), 
																	:Choice])

						avail_actions_m[ID_number, session] = avail_actions_f(ID, day, week, week_idx) .+ 
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

		offset_actions += offset_weeks * 3
		offset_sessions += count_sessions(cb_file, interv_v)
	end
	push!(CHOICE_V, choice_m)
	push!(AVAIL_ACTIONS_V, avail_actions_m)
	push!(GROUP_V, group_m)
	push!(R_V, R_m)
	push!(TRIAL_V, trial_m)

	n_subjects += n_subjects_batch
	n_sessions = n_sessions_batch
	end
	
	choice_m = reduce(vcat, CHOICE_V)
	avail_actions_m = reduce(vcat, AVAIL_ACTIONS_V)
	group_m = reduce(vcat, GROUP_V)
	R_m = reduce(vcat, R_V)
	trial_m = reduce(vcat, TRIAL_V)

	return (choice_m, ABT_t(n_sessions, n_subjects, length(unique(group_m)), avail_actions_m, group_m, R_m, trial_m))
end




