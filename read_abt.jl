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

	#interv_d = Dict("2vs1" => Dict("V" => 1, "1" => 1, "2" => 1),
	#				"FG7142" => Dict("V" => 1, "A" => 2, "B" => 3, "C" => 1),
	#				"cort" => Dict("V" => 1, "A" => 2, "B" => 3, "C" => 4, "D" => 1))

	df = CSV.File(cb_file) |> DataFrame

	choice_d = Dict("A" => 1, "B" => 2, "blank" => 3)

	interv_name = split(cb_file, '_')[2]
	#group_d = interv_d[interv_name]

	reward_d = Dict{String, Dict{String, Float64}}()

	for ID in df.ID
		if interv_name == "2vs1"

			(choice_1, reward_magnitude_1) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== 1), :PD1][1])
			(choice_2, reward_magnitude_2) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== 1), :PD2][1])

			reward_d[ID] = Dict(choice_1 => parse(Float64, reward_magnitude_1),
								choice_2 => parse(Float64, reward_magnitude_2),
								"" => 0.0)
		else
			reward_d[ID] = Dict("A" => 1.0, "B" => 1.0, "" => 0.0)
		end
	end

	function group(ID, day, week)

		if day == "test"

			return group_d["V"]

		else

			(~, interv) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			try group_d[interv]
				return group_d[interv]
			catch e
				return -1
			end
		end
	end

	function avail_actions(ID, day, week, week_idx)

		if day == "test"

			return [choice_d["A"] + (week_idx - 1)*3, choice_d["B"] + (week_idx - 1)*3]

		else

			(avail_action,) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			return [choice_d["blank"] + (week_idx - 1)*3, choice_d[avail_action] + (week_idx - 1)*3]
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

			return map((choice, rewarded_choice) -> choice == rewarded_choice ? reward_d[ID][choice] : 0.0, choice_v, test_rewarded_choicess_v)
		else

			(avail_action, ~) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			return parse.(Float64, choice_v) * reward_d[ID][avail_action]
		end
	end

	return (group, avail_actions, choices, rewards)
end

function count_subjects_sessions(file_v)

	filename = file_v[1]

	df = CSV.File(filename) |> DataFrame

	n_sessions = length(unique(map((x,y) -> (x,y), df.Day, df.Week)))

	for filename in file_v[2:end]

		df_new = CSV.File(filename) |> DataFrame

		df = vcat(df, df_new)

		n_sessions += length(unique(map((x,y) -> (x,y), df_new.Day, df_new.Week)))
	end

	n_subjects = length(unique(df.ID))

	return (n_subjects, n_sessions)
end


function read_data(file_v, cb_file_v, group_d)

	#=
	batch_ID_v = String[]
	batch_ID = split(ID_v[1], "_")[1]

	if any(x -> occursin(batch_ID, x), batch_ID_v)
		offset_ID = 0
	else
		offset_ID = n_subjects
		n_subjects += length(ID_v)
		push!(batch_ID_v, batch_ID)
	end
	=#
	offset_ID = 0

	@assert length(file_v) == length(cb_file_v)

	(n_subjects, n_sessions) = count_subjects_sessions(file_v)

	choice_m = Matrix{Array{Int64,1}}(undef, n_subjects, n_sessions)
	avail_actions_m = Matrix{Array{Int64,1}}(undef, n_subjects, n_sessions)
	group_m = Matrix{Int64}(undef, n_subjects, n_sessions)
	R_m = Matrix{Array{Float64,1}}(undef, n_subjects, n_sessions)
	trial_m = Matrix{Int64}(undef, n_subjects, n_sessions)

	offset_sessions = 0
	offset_actions = 0

	for (file, cb_file) in zip(file_v, cb_file_v)

		df = CSV.File(file) |> DataFrame

		df = df[df.Choice .!= "O", :]

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

						group = group_f(ID, day, week)

						session = map_session(day, week_idx) + offset_sessions

						choice_m[ID_number, session] = choices_f(day, 
																df[(df.ID .== ID) .& (df.Day .== day) .& (df.Week .== week), :Choice])

						avail_actions_m[ID_number, session] = avail_actions_f(ID, day, week, week_idx) .+ offset_actions

						R_m[ID_number, session] = rewards_f(ID, day, week, 
															df[(df.ID .== ID) .& (df.Day .== day) .& (df.Week .== week), :Choice])

						trial_m[ID_number, session] = length(df[(df.ID .== ID) .& (df.Day .== day) .& (df.Week .== week), :Choice])
					end
				end
			end
		end

		offset_actions += length(unique(df.Week)) * 3
		offset_sessions += length(unique(map((x,y) -> (x,y), df.Day, df.Week)))
	end

	return (choice_m, ABT_t(n_sessions, n_subjects, length(unique(group_m)), avail_actions_m, group_m, R_m, trial_m))
end




