"""
# Arguments
- `data::DataFrame`: data from which graphs will be created.
- `axes::Array{String, 1}`: accepted values are x, y, z
- `measures::Array{String, 1}`: accepted values are magnetometr, accelerometr, orientation
"""
# function createfullgraph(
#     data::DataFrame;
#     axes::Array{String, 1} = ["x", "y", "z"],
#     measures::Array{String, 1} = ["magnetometr", "acceleromter", "orientation"]
#     )
#     nrows = length(measures)
#     ncols = length(axes)
#     subplots = initfig(nrows, ncols)
#
#     tograph = []
#     for axis in axes
#         for measure in measures
#             push!(tograph, Symbol(measure * "_" * axis))
#         end
#     end
#
#     for (index, graph) in enumerate(tograph)
#
#         if contains(String(graph), "orientation")
#             subplots[index].set_ylim(-180, 360)
#         end
#         subplots[index].set_title(String(graph))
#     end
#
#     samples = unique(data[:, :path_sample])
#
#     for sample in samples
#         samplerows = data[!, :path_sample] .== sample
#         sampledata = data[samplerows, :]
#         timestep = sampledata[!, :timestep]
#         [subplots[i].plot(timestep, sampledata[!, tograph[i]]) for i=1:9]
#     end
#
#     return nothing
# end

function createcomparisongraphs(
    data1::DataFrame,
    data2::DataFrame;
    axes::Array{String,1} = ["x", "y", "z"],
    measures::Array{String,1} = ["magnetometr", "accelerometr", "orientation"],
)

    tograph = creategraphtags(axes, measures)
    titles = createtitles(tograph)
    limits = createlimits(tograph)
    x, y = createxy(tograph, data1, data2)
    pltsize = getsize(tograph)
    layout = createlayout(tograph)

    println(typeof(x), typeof(y))
    plot(
        x,
        y,
        layout = layout,
        size = pltsize,
        ylim = limits,
        titles = titles,
        label = ["path1" "path1" "path1" "path2" "path2" "path2"],
        w = 3,
        xlabel = "t = [s]"
    )
end

function createlayout(tograph)
    #return Plots.GridLayout(length(tograph), 2)
    return Plots.GridLayout(3, 1)
end

function creategraphtags(axes, measures)
    tograph = []
    for axis in axes
        for measure in measures
            push!(tograph, Symbol(measure * "_" * axis))
        end
    end
    sort(tograph)
end

function getsize(tograph)
    singlegraphwidth = 300
    signlegraphheight = 400
    n = length(tograph)
    width = singlegraphwidth * n
    height = signlegraphheight * n
    return width, height
end

function createxy(tograph, data1, data2)
    x, y = [], []
    for graph in tograph
        x1 = convert(Array{Float64,1}, data1[:, :timestep])
        push!(x, x1)
        y1 = convert(Array{Float64,1}, data1[:, graph])
        push!(y, y1)
    end
    for graph in tograph
        x2 = convert(Array{Float64,1}, data2[:, :timestep])
        push!(x, x2)
        y2 = convert(Array{Float64,1}, data2[:, graph])
        push!(y, y2)
    end

    return x, y
end

function createlimits(tograph)
    limits = []
    for graph in tograph
        if contains(String(graph), "orientation")
            limit = (-180, 360)
        else
            limit = 0
        end
        push!(limits, limit)
    end
    return limits
end

function createtitles(tograph)
    titles = []
    for graph in tograph
        push!(titles, String(graph))
    end
    return permutedims(titles)
end
