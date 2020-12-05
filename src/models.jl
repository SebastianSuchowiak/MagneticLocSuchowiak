function getfolds(data)
    fold = Tuple{Array{Int64,1},Array{Int64,1}}[]
    samples_tags = [1, 2, 3, 4, 5]
    for test_tag in samples_tags
        is_test = data.path_sample .== test_tag
        test = findall(is_test)
        train = findall(.!(is_test))
        push!(fold, (train, test))
    end

    return fold
end


function preparedata(data)
    y, X = unpack(data, y -> y ∈ [:lon, :lat],  colname -> colname ∉ [:path_id, :path_sample])
end


function crossvalidate(X, y, folds, model)
    results = []
    y1 = coerce(y[1], Continuous)
    y2 = coerce(y[2], Continuous)

    m1 = machine(model, X, y1)
    m2 = machine(model, X, y2)

    println(evaluate!(m1, resampling = folds, measure=[l1, l2], verbosity=0))
    println(evaluate!(m2, resampling = folds, measure=[l1, l2], verbosity=0))

    return(m1, m2)
end


function fitandmean!(model_x, model_y, train, test)
    folds = getfolds(train)

    y, X = preparedata(train)
    y1 = coerce(y[1], Continuous)
    y2 = coerce(y[2], Continuous)

    m_x = machine(model_x, X, y1);
    m_y = machine(model_y, X, y2);

    fit!(m_x, verbosity=0)
    fit!(m_y, verbosity=0)

    test_y, test_X = preparedata(test)
    ȳ1 = predict(m_x, test_X)
    ȳ2 = predict(m_y, test_X)
    predicted_y = hcat(ȳ1, ȳ2)

    errors = vechaversine(test_y, predicted_y)

    return mean(errors)
end


function vechaversine(x::AbstractVector, y::AbstractVector)
    if length(x) != length(y)
        throw(ArgumentError("x and y lengths must be equal"))
    end

    r = 6371000
    result = []
    for i in 1:length(x)
        point1 = x[i]
        point2 = x[i]
        push!(result, haversine(point1, point2, r))
    end

    return result
end
