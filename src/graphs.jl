"""
# Arguments
- `data::DataFrame`: data from which graphs will be created.
- `axes::Array{String, 1}`: accepted values are x, y, z
- `measures::Array{String, 1}`: accepted values are magnetometr, accelerometr, orientation
"""
function createfullgraph(
    data::DataFrame;
    axes::Array{String, 1} = ["x", "y", "z"],
    measures::Array{String, 1} = ["magnetometr", "accelerometr", "orientation"]
    )
    nrows = length(measures)
    ncols = length(axes)
    subplots = initfig(nrows, ncols)

    tograph = []
    for axis in axes
        for measure in measures
            push!(tograph, Symbol(measure * "_" * axis))
        end
    end

    for (index, graph) in enumerate(tograph)
        if contains(String(graph), "orientation")
            subplots[index].set_ylim(-180, 360)
        end
    end

    samples = unique(data[:, :path_sample])

    for sample in samples
        samplerows = data[!, :path_sample] .== sample
        sampledata = data[samplerows, :]
        timestep = sampledata[!, :timestep]
        [subplots[i].plot(timestep, sampledata[!, tograph[i]]) for i=1:9]
    end

    return nothing
end

function createcomparisongraphs(
    data1::DataFrame,
    data2::DataFrame;
    axes::Array{String, 1} = ["x", "y", "z"],
    measures::Array{String, 1} = ["magnetometr", "accelerometr", "orientation"])

    nrows = length(axes) * length(measures)
    ncols = 2
    subplots = initfig(nrows, ncols)

    tograph = []
    for axis in axes
        for measure in measures
            push!(tograph, Symbol(measure * "_" * axis))
        end
    end
    sort!(tograph)

    for (index, graph) in enumerate(tograph)
        if contains(String(graph), "orientation")
            subplots[index].set_ylim(-180, 360)
            subplots[index+nrows].set_ylim(-180, 360)
        end
    end

    samples1 = unique(data1[:, :path_sample])
    for sample in samples1
        samplerows = data1[!, :path_sample] .== sample
        sampledata = data1[samplerows, :]
        timestep = sampledata[!, :timestep]
        [subplots[i].plot(timestep, sampledata[!, tograph[i]]) for i=1:9]
    end

    samples2 = unique(data2[:, :path_sample])
    for sample in samples2
        samplerows = data2[!, :path_sample] .== sample
        sampledata = data2[samplerows, :]
        timestep = sampledata[!, :timestep]
        [subplots[nrows+i].plot(timestep, sampledata[!, tograph[i]]) for i=1:9]
    end
end

function initfig(nrows, ncols)
    single_plot_size = 6
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*single_plot_size, nrows*single_plot_size))
    fig.tight_layout(pad=4.0)
    return axs
end
