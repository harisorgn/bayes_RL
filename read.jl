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

function map_reward(day, choice_v)

	test_reward_v = ["A", "B", "", "B", "", "A", "", 
					"A", "B", "A", "B", "", "B", "", 
					"A", "", "A", "B", "A", "B", "", 
					"B", "", "A", "", "A", "B", "A", "B", ""]

	if day == "test"

		return Float64.(choice_v .== test_reward_v[1:length(choice_v)])

	else
		return parse.(Float64, choice_v)
	end

end

function map_action(day, choice_v)

	if day == "test"

		return map(x -> x == "A" ? 0 : 1, choice_v)

	else
		return parse.(Int, choice_v)
	end

end

function counterbalance_map(cb_file)

	df = CSV.File(cb_file) |> DataFrame

	action_d = Dict("A" => 1, "B" => 2)

	interv_d = occursin("2vs1", cb_file) ? Dict("V" => 1, "1" => 1, "2" => 1) : Dict("V" => 1, "A" => 2, "B" => 3, "C" => 4)

	function avail_actions(ID, day, week)

		if day == "test"

			return [action_d["A"] + (week - 1)*3, action_d["B"] + (week - 1)*3]

		else

			(avail_action,) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			return [3 + (week - 1)*3, action_d[avail_action] + (week - 1)*3]
		end
	end

	function intervention(ID, day, week)

		if day == "test"

			return interv_d["V"]

		else

			(~, interv) = parse_tuple(df[(df.ID .== ID) .& (df.Week .== week), Symbol(day)][1])

			return interv_d[interv]
		end
	end

	return (avail_actions, intervention)
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


function read_data(file_v, cb_file_v)

	@assert length(file_v) == length(cb_file_v)

	(n_subjects, n_sessions) = count_subjects_sessions(file_v)

	action_m = Matrix{Array{Int64,1}}(undef, n_subjects, n_sessions)
	avail_actions_m = Matrix{Array{Int64,1}}(undef, n_subjects, n_sessions)
	interv_m = Matrix{Int64}(undef, n_subjects, n_sessions)
	R_m = Matrix{Array{Float64,1}}(undef, n_subjects, n_sessions)
	trial_m = Matrix{Int64}(undef, n_subjects, n_sessions)

	n_subjects = 0
	n_sessions = 0
	n_actions = 0

	batch_ID_v = String[]

	for (file, cb_file) in zip(file_v, cb_file_v)

		df = CSV.File(file) |> DataFrame

		df = df[df.Choice .!= "O", :]

		ID_v = unique(df.ID)
		day_v = unique(df.Day)
		week_v = unique(df.Week)

		(avail_actions_f, interv_f) = counterbalance_map(cb_file)

		batch_ID = split(ID_v[1], "_")[1]

		if any(x -> occursin(batch_ID, x), batch_ID_v)
			offset_ID = 0
		else
			offset_ID = n_subjects
			n_subjects += length(ID_v)
			push!(batch_ID_v, batch_ID)
		end

		for ID in ID_v
			for day in day_v
				for week in week_v

					ID_number = parse(Int64, ID[findlast('_', ID)+1 : end]) + offset_ID
					session = map_session(day, week) + n_sessions

					action_m[ID_number, session] = map_action(day, df[(df.ID .== ID) .& (df.Day .== day) .& (df.Week .== week), :Choice])

					avail_actions_m[ID_number, session] = avail_actions_f(ID, day, week) .+ n_actions

					interv_m[ID_number, session] = interv_f(ID, day, week)

					R_m[ID_number, session] = map_reward(day, df[(df.ID .== ID) .& (df.Day .== day) .& (df.Week .== week), :Choice])

					trial_m[ID_number, session] = length(df[(df.ID .== ID) .& (df.Day .== day) .& (df.Week .== week), :Choice])
				end
			end
		end

		n_actions += length(unique(df.Week)) * 3
		n_sessions += length(unique(map((x,y) -> (x,y), df.Day, df.Week)))
	end

	return (action_m, ABT_t(n_sessions, n_subjects, length(unique(interv_m)), avail_actions_m, interv_m, R_m, trial_m))
end




