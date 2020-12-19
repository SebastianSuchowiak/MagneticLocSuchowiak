knn = KNNRegressor()
forest = RandomForestRegressor()
nusvr = NuSVR()

grid = Grid(resolution = 100, shuffle = false)

function fitknn(traindf; klower::Int = 1, kupper::Int = 20)
    folds = getfolds(traindf)
    r1 = range(knn, :K, lower = klower, upper = kupper)

    knnm = TunedModel(
        model = knn,
        resampling = folds,
        tuning = grid,
        range = [r1],
        measure = l2,
    )

    doublefit(knnm, traindf)
end

function fitnusvr(traindf)
    folds = getfolds(traindf)
    r_tolerance = range(nusvr, :tolerance; values = [10e6])
    r_gamma = range(nusvr, :gamma; values = [0.1])
    r_nu = range(nusvr, :nu, values = [0.5, 1])

    nusvrm = TunedModel(
        model = nusvr,
        resampling = folds,
        tuning = grid,
        range = [r_gamma, r_nu],
        measure = l2,
    )

    doublefit(nusvrm, traindf)
end

function fitforest(
    traindf;
    ntrees = [100, 200, 500],
    maxdepths = [-1, 10, 100],
    minleafs = [1, 5, 20, 100]
)
    folds = getfolds(traindf)
    r_n_trees =
        range(forest, :n_trees; values = ntrees)
    r_max_depth = range(forest, :max_depth, values = maxdepths)
    r_min_leaf = range(forest, :min_samples_leaf, values = minleafs)

    forestm = TunedModel(
        model = forest,
        resampling = CV(nfolds=5),
        tuning = grid,
        range = [r_n_trees, r_max_depth, r_min_leaf],
        measure = l2,
        acceleration = CPUProcesses(),
    )

    doublefit(forestm, traindf)
end

function fitcustommodel(traindf, model)
    folds = getfolds(traindf)
    doublefit(model, traindf)
end

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

function doublefit(model, traindf)
    X, y = preparedata(traindf)
    m_lon = machine(model, X, y[:, 1])
    m_lat = machine(model, X, y[:, 2])
    fit!(m_lon)
    fit!(m_lat)

    return m_lon, m_lat
end

function preparedata(data)
    X, y = unpack(
        data,
        X -> X ∈ [:magnetometr_x, :magnetometr_y, :magnetometr_z],
        y -> y ∈ [:lon, :lat],
    )
    X = coerce(
        X,
        :magnetometr_x => Continuous,
        :magnetometr_y => Continuous,
        :magnetometr_z => Continuous,
    )
    y = coerce(y, :lon => Continuous, :lat => Continuous)

    return X, y
end

function testmachines(machine_lon, machine_lat, testdf, coordsstand)

    test_X, test_y = preparedata(testdf)
    test_y[!, [:lon, :lat]] = MLJ.inverse_transform(coordsstand, test_y[!, [:lon, :lat]])

    ȳlon = predict(machine_lon, test_X)
    ȳlat = predict(machine_lat, test_X)
    predy = DataFrame(lon=ȳlon, lat=ȳlat)
    predy = coerce(predy, :lon => Continuous, :lat => Continuous)
    predy = MLJ.inverse_transform(coordsstand, predy)

    ȳperdicted = collect(zip(predy[:, :lon], predy[:, :lat]))
    ytest = collect(zip(test_y[:, :lon], test_y[:, :lat]))

    evaluateresults(ytest, ȳperdicted)
end

function evaluateresults(ytest, ypred)
    error = vechaversine(ytest, ypred)

    return Dict(
        "mean" => mean(error),
        "max" => maximum(error),
        "min" => minimum(error),
        "median" => median(error),
        "std" => std(error),
        "80" => quantile(error, 0.8),
    )
end

function vechaversine(x, y)
    if length(x) != length(y)
        throw(ArgumentError("x and y lengths must be equal"))
    end

    r = 6371000
    result = []
    for i = 1:length(x)
        point1 = x[i]
        point2 = y[i]
        push!(result, haversine(point1, point2, r))
    end

    return result
end
