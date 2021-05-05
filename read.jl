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

function counterbalance_map(counterbalance_filename)

	df = CSV.File(counterbalance_filename) |> DataFrame

	action_d = Dict("A" => 1, "B" => 2)
	interv_d = Dict("V" => 1, "A" => 2, "B" => 3, "C" => 4)

	function avail_actions(ID, day, week)

		if day == "test"

			return [1 + (week - 1)*3, 2 + (week - 1)*3]

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

function read_data(filename, counterbalance_filename)

	df = CSV.File(filename) |> DataFrame

	df.N = map(x -> parse(Int64, x[findlast('_', x)+1 : end]), df.ID)

	(avail_actions_f, interv_f) = counterbalance_map("./ER17_FG7142_counterbalance.csv")

	ID_v = unique(df.ID)
	day_v = unique(df.Day)
	week_v = unique(df.Week)

	n_sessions = length(unique(map((x,y) -> (x,y), df.Day, df.Week)))

	avail_actions_m = Matrix{Array{Int64,1}}(undef, length(ID_v), n_sessions)
	interv_m = Matrix{Int64}(undef, length(ID_v), n_sessions)
	R_m = Matrix{Array{Float64,1}}(undef, length(ID_v), n_sessions)

	for ID in ID_v
		for day in day_v
			for week in week_v

				ID_number = parse(Int64, ID[findlast('_', ID)+1 : end])
				session = map_session(day, week)

				avail_actions_m[ID_number, session] = avail_actions_f(ID, day, week)

				interv_m[ID_number, session] = interv_f(ID, day, week)

				R_m[ID_number, session] = map_reward(day, 
													df[(df.ID .== ID) .& (df.Day .== day) .& (df.Week .== week), :Choice])

			end
		end
	end

	return ABT_t(n_sessions, length(ID_v), length(unique(interv_m)), )
end




