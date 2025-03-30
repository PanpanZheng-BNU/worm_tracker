module Interpolation
    export interpolation_missing_frame, cal_missing, vis_trace
    using StatsKit, LinearAlgebra, Pipe, CairoMakie
    Makie.convert_arguments(::Type{<:AbstractPlot}, x::DataFrame) = ([Point2f(i) for i in Matrix(select(x, [:x, :y])) |> eachrow],)
    Makie.convert_arguments(::Type{<:AbstractPlot}, df::DataFrame, col::Vector{Symbol}) = 
        (map(Point2f, select(df, col) |> Matrix |> eachrow),)

    function interpolation(frame::Int64, prev::Vector{<:Number}, next::Vector{<:Number})
        prev_x, prev_y, prev_f = prev
        next_x, next_y, next_f = next
        pred_x = prev_x + (next_x - prev_x) * (frame - prev_f) / (next_f - prev_f)
        pred_y = prev_y + (next_y - prev_y) * (frame - prev_f) / (next_f - prev_f)
        return [pred_x, pred_y]
    end

    function find_prev_and_next(df::DataFrame)
        ori_df = df
        tmp_df = @pipe df |> 
            filter(:frames => x -> (x > 300 && x%20==1))  
        # @show tmp_df
        tmp_df = @pipe tmp_df |> 
            transform!(_, :frames => (x -> [missing; diff(x)]) => :diff_frame) |>
            filter(:diff_frame => x -> !ismissing(x) && x .> 20) |> 
            vcat(_, tmp_df[tmp_df.frames .∈ (_.frames .- _.diff_frame,), :]) |> 
            sort(_, :frames) 
        missing_frames = []
        key_df = DataFrame()
        for i in tmp_df[2:2:end, :] |> eachrow
            push!(missing_frames, i.frames .- (20:20:(i.diff_frame - 20)|>collect))
        end
        key_df[!, :missing_frames] = missing_frames
        key_df[!, :prev_tracker] = tmp_df[1:2:end, :trackers_id]
        key_df[!, :next_tracker] = tmp_df[2:2:end, :trackers_id]

        tmp_prev_df = @pipe ori_df |> filter(row -> row.trackers_id ∈ key_df.prev_tracker) |> 
            groupby(_, :trackers_id) |> 
            combine(_, :frames => maximum => :frames) |>
            leftjoin(_, ori_df, on = [:trackers_id,:frames]) |> 
            transform!(_, [:x, :y] =>  ByRow((x,y) -> Vector([x,y])) =>  :prev_position) |>
            select(_, [:frames, :prev_position]) |>
            DataFrames.rename!(_, :frames => :prev_frame)
        tmp_next_df = @pipe ori_df |> filter(row -> row.trackers_id ∈ key_df.next_tracker) |> 
            groupby(_, :trackers_id) |> 
            combine(_, :frames => minimum => :frames) |>
            leftjoin(_, ori_df, on = [:trackers_id,:frames]) |> 
            transform!(_, [:x, :y] =>  ByRow((x,y) -> Vector([x,y])) =>  :next_position) |>
            select(_, [:frames, :next_position]) |>
            DataFrames.rename!(_, :frames => :next_frame)
        return hcat(key_df, tmp_prev_df, tmp_next_df) 
    end

    function interpolation_missing_frame(df::DataFrame)
        if cal_missing(df) == 0
            return DataFrame([i => [] for i in names(df)])
        end
        ori_df = deepcopy(df)
        key_df = find_prev_and_next((ori_df))
        new_df = @pipe key_df |>
            transform!(_, [:missing_frames, :prev_frame, :prev_position, :next_frame, :next_position] => 
                ByRow((mf, pf, pp, nf, np) -> [interpolation(i, vcat(pp, pf), vcat(np, nf)) for i in mf]) => :pred_position)
        
        
        tmp_missing_frame = @pipe new_df.missing_frames |> vcat(_...)
        pred_x, pred_y = @pipe new_df.pred_position |> vcat(_...) |> hcat(_...)' |> eachcol |> Vector.(_)
        interpolation_df = @pipe [i == "frames" ? trunc.(Int64,tmp_missing_frame) : i == "x" ? pred_x : i == "y" ? pred_y : [missing for _ in tmp_missing_frame]  for i in names(ori_df)] |> 
            hcat(_...) |> 
            DataFrame(_, names(ori_df))
        interpolation_df[!, :frames] = trunc.(Int64, interpolation_df[!, :frames])
        return interpolation_df
    end

    function cal_missing(df::DataFrame)
        return @pipe df |> filter(:frames => x -> (x >300 &&  x%20==1), _) |>
            [0; diff(_.frames)] |> filter(x -> x > 20, _) |> sum((_ .- 20) / 20) |> trunc(Int, _)
    end

    function vis_trace(df::DataFrame)
        tmp_interpolation_df = interpolation_missing_frame(df)
        tmp_new_merge_df = @pipe df |> filter(:frames => x -> (x > 300) && (x%20 == 1)) |>
            vcat(_,tmp_interpolation_df) |> sort(_,:frames) 
        fig = Figure()
        ax = Axis(fig[1, 1], title = "Trace", xlabel = "X", ylabel = "Y")
        scatterlines!(ax, tmp_new_merge_df, [:x, :y])
        scatter!(ax, tmp_interpolation_df, [:x, :y], color = :red)
        return fig
    end
end
