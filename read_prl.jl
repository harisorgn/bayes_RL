using CSV

function read_data(file)

	df = CSV.File(file) |> DataFrame

	n_subjects = length(unique(df.ID))

	action_v = Array{Array{Int64,1},1}(undef, n_subjects)
	group_v = Array{Int64,1}(undef, n_subjects)
	trial_v = Array{Int64,1}(undef, n_subjects)

	R_m = Matrix{Array{Float64,1}}(undef, n_subjects, n_sessions)
	
	n_subjects = 0

	batch_ID_v = String[]

	for file in file_v

		df = CSV.File(file) |> DataFrame

		df = df[df.Choice .!= "O", :]

		ID_v = unique(df.ID)

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




