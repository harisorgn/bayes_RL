
struct ABT_t
	n_sessions::Int64
	n_subjects::Int64
	n_groups::Int64
	n_sessions_per_week::Int64
	n_avail_actions_per_week::Int64
	avail_actions_m::Matrix{Array{Int64,1}}
	group_m::Matrix{Int64}
	R_m::Matrix{Array{Float64,1}}
	trial_m::Matrix{Int64}
end

struct PRL_t
	n_subjects::Int64
	n_groups::Int64
	group_v::Array{Int64,1}
	R_v::Array{Array{Float64,1},1}
	trial_v::Array{Int64,1}
end