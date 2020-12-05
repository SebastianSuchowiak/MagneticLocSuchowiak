function readdata(paths::Array{String, 1}; columns_to_get=[], add_tag::Bool = true)
    formated_data = DataFrame(reshape([], 0, length(columns_to_get)))
    rename!(formated_data, columns_to_get)

    for file in paths
        formated_data =
            vcat(formated_data, readdata(file, columns_to_get=columns_to_get, add_tag=add_tag))
    end

    return formated_data
end


function readdata(path::String; columns_to_get::Array=[], add_tag::Bool = true)
    df = readdatafromcsv(path)
    meta = getmetadata(df)
    data = getdatawithoutmeta(df)

    lonlat = calculatelonlat(meta)
    data = hcat(data, lonlat)

    if add_tag
        tag_columns = createtagcolumns(path, data)
        data = hcat(data, tag_columns)
    end

    data = cutoutliners(data)
    data[!, :timestep] = 0:nrow(data)-1
    data[!, :timestep] = data[!, :timestep] .* 0.1

    data = data[:, columns_to_get]
end


function readdatafromcsv(path)
    AXIS = ["_x" "_y" "_z"]
    NAMES = [:timestep]
    NAMES = vcat(NAMES, vec(Symbol.(["magnetometr"], AXIS)))
    NAMES = vcat(NAMES, vec(Symbol.(["acceleromter"], AXIS)))
    NAMES = vcat(NAMES, vec(Symbol.(["orientation"], AXIS)))

    df =
        CSV.read(path, DataFrame, header = NAMES, silencewarnings = true, threaded = false)
end


function getmetadata(df)
    METADATA_COLNAMES = [
        "latitude_start",
        "longitude_start",
        "latitude_end",
        "longitude_end",
        "first_sample",
        "last_sample",
    ]

    gapidx = findfirst(occursin.(r"<\d+>", df[!, "timestep"]))
    meta = df[gapidx+1:end, :][:, 1:6]
    rename!(meta, Symbol.(METADATA_COLNAMES))
    meta[!, "latitude_start"] = parse.(Float64, meta[!, "latitude_start"])
    meta[!, "first_sample"] = trunc.(Int, meta[!, "first_sample"])
    meta[!, "last_sample"] = trunc.(Int, meta[!, "last_sample"])
    meta = DataFrame(meta)
end


function getdatawithoutmeta(df)
    gapidx = findfirst(occursin.(r"<\d+>", df[!, "timestep"]))
    data = df[1:gapidx-1, :]
end


function calculatelonlat(meta)
    samples_num = last(meta, 1)[!, "last_sample"][1] + 1
    result = DataFrame(lat = 1:samples_num, lon = 1:samples_num)
    result[!, "lat"] = convert.(Float64, result[!, "lat"])
    result[!, "lon"] = convert.(Float64, result[!, "lon"])

    for corridor in eachrow(meta)
        first_sample_idx = corridor["first_sample"] + 1 # This dataset is 0-index-based
        last_sample_idx = corridor["last_sample"] + 1
        corridor_idxs = first_sample_idx:last_sample_idx

        corridor_samples_num = last_sample_idx - first_sample_idx + 1

        lat_diff = corridor["latitude_end"] - corridor["latitude_start"]
        lon_diff = corridor["longitude_end"] - corridor["longitude_start"]

        lat_step = lat_diff / corridor_samples_num
        lon_step = lon_diff / corridor_samples_num

        result[corridor_idxs, :lon] =
            (result[corridor_idxs, :lon] .- first_sample_idx) * lon_step .+
            corridor["longitude_start"]
        result[corridor_idxs, :lat] =
            (result[corridor_idxs, :lat] .- first_sample_idx) * lat_step .+
            corridor["latitude_start"]
    end

    return result
end


function createtagcolumns(path, data)
    n_rows = nrow(data)
    tag = match(r"(.*[/\\])*(?<id>.*)_(?<sample>.*)\.txt", path)
    result = DataFrame(
        path_id = repeat([tag[:id]], n_rows),
        path_sample = repeat([tag[:sample]], n_rows)
    )
end


function cutoutliners(data)
    data = data[setdiff(1:end, 1:3), :]
    data = data[setdiff(1:end, end-3:end), :]
end
