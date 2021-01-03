using DataFrames
using MLJ
using MagneticLocSuchowiak
using Plots

function getdata()
    DATA_PATH = "/home/sebastian/Desktop/PracaInz/data"
    CURVES = DATA_PATH * "/curves"
    LINES = DATA_PATH * "/lines"
    train_dirs = vcat(readdir(LINES, join = true), readdir(CURVES, join = true))
    TESTS = DATA_PATH * "/tests"
    test_files = readdir(TESTS, join = true)
    train_files = collect(Iterators.flatten(readdir.(train_dirs, join = true)))
    #train_files = [x for x in train_files if !contains(last(split(x, "\\")), "r")]

    ["magnetometr", "accelerometr", "orientation"]

    ALL_COLUMNS = [
        :timestep,
        :magnetometr_x,
        :magnetometr_y,
        :magnetometr_z,
        :acceleromter_x,
        :acceleromter_y,
        :acceleromter_z,
        :orientation_x,
        :orientation_y,
        :orientation_z,
        :lon,
        :lat,
        :path_id,
        :path_sample,
    ]

    TRAIN_COLUMNS = [
        :timestep,
        :magnetometr_x,
        :magnetometr_y,
        :magnetometr_z,
        :lon,
        :lat,
        :path_id,
        :path_sample,
    ]
    TEST_COLUMNS =
        [:timestep, :magnetometr_x, :magnetometr_y, :magnetometr_z,  :lon, :lat, :path_id]
    traindf = readdata(train_files, columns_to_get = TRAIN_COLUMNS, samplestocut = 3)
    testdf = readdata(test_files, columns_to_get = TEST_COLUMNS)

    coordinates = vcat(traindf[:, [:lon, :lat]], testdf[:, [:lon, :lat]])
    coordinates = coerce(coordinates, :lon => Continuous, :lat => Continuous)
    coordinatesstand = fit!(machine(Standardizer(), coordinates))
    traindf[!, [:lon, :lat]] = MLJ.transform(coordinatesstand, traindf[!, [:lon, :lat]])
    testdf[!, [:lon, :lat]] = MLJ.transform(coordinatesstand, testdf[!, [:lon, :lat]])

    # magcols = [:magnetometr_x, :magnetometr_y, :magnetometr_z]
    # mag_test = testdf[:, magcols]
    # mag_test = coerce(mag_test, :magnetometr_x => Continuous, :magnetometr_y => Continuous, :magnetometr_z => Continuous)
    # mag_test_stand = fit!(machine(Standardizer(), mag_test))
    # testdf[!, magcols] = MLJ.transform(mag_test_stand, testdf[!, magcols])
    #
    # mag_train =traindf[:, magcols]
    # mag_train = coerce(mag_train, :magnetometr_x => Continuous, :magnetometr_y => Continuous, :magnetometr_z => Continuous)
    # mag_train_stand = fit!(machine(Standardizer(), mag_train))
    # traindf[!, magcols] = MLJ.transform(mag_train_stand, traindf[!, magcols])

    return traindf, testdf, coordinatesstand
end

function msummary(lat, lon, traindf, testdf, coordinatesstand)
    println("Hyperparameters lat model: ", params(fitted_params(lat).best_model))
    println("Hyperparameters lon model: ", params(fitted_params(lon).best_model))
    println("Test set evaluation: ", testmachines(lon, lat, testdf, coordinatesstand))
    println("Train set evaluation: ", testmachines(lon, lat, traindf, coordinatesstand))
end

traindf, testdf, standardizer = getdata()
testdf = traindf[traindf[:, :path_sample].=="05", :]
traindf = traindf[.!(traindf[:, :path_sample].=="05"), :]


lon, lat = fitknn(traindf, klower = 50, kupper = 51)
lon, lat = fitforest(
    traindf,
    ntrees = [100],
    maxdepths = [50],
    minleafs = [50, 100],
)
lon, lat = fitnusvr(traindf, [0.1, 0.2, 0.3], [0.5, 1])

msummary(lat, lon, traindf, testdf, standardizer)

svmnuerror(traindf, standardizer)
svmgammaerror(traindf, standardizer)
knnkerror(traindf, standardizer)

ids = string.("tt", vcat(string.("0", 1:9), ["10", "11"]))
println("\n++++++++++++")
for id in ids
    path = testdf[testdf[:, :path_id].==id, :]
    testresult = testmachines(lon, lat, path, standardizer)
    print(id*" "*string(testresult["mean"])*"\n")
end
println("++++++++++++")

pathid = "tt03"
plotpathcomparison(pathid, testdf, lat, lon)
pathid = "l1n"
plottrain(pathid, traindf, lon, lat)


println("\n++++++++++++")
for id in ids
    path = testdf[testdf[:, :path_id].==id, :]

    realtestdf = MLJ.copy(path)
    realtestdf[!, [:lon, :lat]] =
        MLJ.inverse_transform(standardizer, realtestdf[!, [:lon, :lat]])
    meanlat = mean(realtestdf.lat)
    meanlon = mean(realtestdf.lon)
    naivepred = [(meanlon, meanlat) for i = 1:nrow(realtestdf)]
    ytest = collect(zip(realtestdf[:, :lon], realtestdf[:, :lat]))
    testresult = MagneticLocSuchowiak.evaluateresults(ytest, naivepred)

    print(id*" "*string(testresult["mean"])*"\n")
end
println("++++++++++++")


function foresterror()
    train_mean = []
    test_mean = []
    for i in 1:300
        lon, lat = fitcustommodel(traindf, RandomForestRegressor(n_trees=i))

        results_train = testmachines(lon, lat, traindf, standardizer)
        push!(train_mean, results_train["mean"])

        results_test = testmachines(lon, lat, testdf, standardizer)
        push!(test_mean, results_test["mean"])
    end
    plot(
        1:300,
        [train_mean, test_mean],
        label = ["test" "train"],
        title = "Mean Distance Error",
        ylabel = "s=[m]",
        xlabel = "n_trees",
    )
end

function svmnuerror(traindf, standardizer)
    train_mean = []
    test_mean = []
    nus = 0.1:0.1:0.9
    for i in nus
        lon, lat = fitcustommodel(traindf, NuSVR(gamma=0.1, nu=i))

        results_train = testmachines(lon, lat, traindf, standardizer)
        push!(train_mean, results_train["mean"])

        results_test = testmachines(lon, lat, testdf, standardizer)
        push!(test_mean, results_test["mean"])
    end
    display(plot(nus, [train_mean, test_mean], label=["train" "test"], title="Błąd w zależności od nu", ylabel="s=[m]", xlabel="nu"))
    savefig("svm_nu.png")
end

function svmgammaerror(traindf, standardizer)
    train_mean = []
    test_mean = []
    gammas = [0.1, 0.5, 1, 3, 5, 7, 10, 15]
    for i in gammas
        lon, lat = fitcustommodel(traindf, NuSVR(nu=0.1, gamma=i))

        results_train = testmachines(lon, lat, traindf, standardizer)
        push!(train_mean, results_train["mean"])

        results_test = testmachines(lon, lat, testdf, standardizer)
        push!(test_mean, results_test["mean"])
    end
    display(plot(gammas, [train_mean, test_mean], label=["train" "test"], title="Bląd w zależności od gamma", ylabel="s=[m]", xlabel="gamma"))
    savefig("svm_gamma.png")
end

function knnkerror(traindf, standardizer)
    train_mean = []
    test_mean = []
    ks = vcat(1:20, 21:10:200)
    for i in ks
        lon, lat = fitcustommodel(traindf, KNNRegressor(K=i))

        results_train = testmachines(lon, lat, traindf, standardizer)
        push!(train_mean, results_train["mean"])

        results_test = testmachines(lon, lat, testdf, standardizer)
        push!(test_mean, results_test["mean"])
    end
    plot(ks, [train_mean, test_mean], label=["train" "test"], title="Błąd w zależności od k", ylabel="s=[m]", xlabel="k")
    savefig("knn_k.png")
end

function plotpathcomparison(testpathfile, testdf, lat, lon)
    pathdf = testdf[testdf[:, :path_id].==testpathfile, :]
    X, y = MagneticLocSuchowiak.preparedata(pathdf)
    pred1 = MLJ.predict(lon, X)
    pred2 = MLJ.predict(lat, X)

    lonplot = plot(
        pathdf[:, :timestep],
        hcat(y[:, :lon], pred1),
        w = 3,
        title = "Longitude",
        label = ["Actual" "Predicted"],
        xlabel = "[t] = s",
        ylabel = "Standarized Logitude",
    )

    latplot = plot(
        pathdf[:, :timestep],
        hcat(y[:, :lat], pred2),
        w = 3,
        title = "Latitude",
        label = ["Actual" "Predicted"],
        xlabel = "[t] = s",
        ylabel = "Standarized Latitude",
    )

    latlonplot = plot(
        hcat(y[:, :lon], pred1),
        hcat(y[:, :lat], pred2),
        w = 3,
        title = "Path",
        label = ["Actual" "Predicted"],
        xlabel = "Standarized Logitude",
        ylabel = "Standarized Latitude",
    )

    display(lonplot)
    display(latplot)
    display(latlonplot)

    display(plot(lonplot, latplot, latlonplot, layout=grid(3,1), size=(600, 900)))
    return lonplot, latplot, latlonplot
end

function odstajace(traindf)
    sample =
        traindf[.&(traindf[:, :path_id] .== "l6n", traindf[:, :path_sample] .== "05"), :]
    plot([sample.timestep], [sample.magnetometr_x], w = 3, label = "składowa x natężenia")
    plot!(
        [sample.timestep[1:2]],
        [sample.magnetometr_x[1:2]],
        seriestype = :scatter,
        title = "",
        label = "wartości odstające",
        xlabel = "[t] = s",
        marker = (10, 5, :x),
        palette = [:red],
    )
    title!("Ścieżka l6n")
    png("Odstające")
end

function plottrain(pathid, traindf, mlon, mlat)
    singlepath = traindf[traindf[:, :path_id].==pathid, :]
    sampleslabels = unique(singlepath[:, :path_sample])
    singlesamples = [
        singlepath[singlepath[:, :path_sample].==samplelabel, :]
        for samplelabel in sampleslabels
    ]
    lons = [x[:, :lon] for x in singlesamples]
    lats = [x[:, :lat] for x in singlesamples]
    timesteps = [x[:, :timestep] for x in singlesamples]

    sampletopred = singlesamples[1]
    X, y = MagneticLocSuchowiak.preparedata(sampletopred)
    predlon = MLJ.predict(mlon, X)
    predlat = MLJ.predict(mlat, X)
    push!(lons, predlon)
    push!(lats, predlat)
    push!(timesteps, sampletopred[:, :timestep])

    labels = permutedims(vcat(sampleslabels, "prediction"))

    lonplot = plot(
        timesteps,
        lons,
        w = 2,
        title = "Longitude",
        label = labels,
        xlabel = "[t] = s",
        ylabel = "Standarized Longitude",
    )

    latplot = plot(
        timesteps,
        lats,
        w = 2,
        title = "Latitude",
        label = labels,
        xlabel = "[t] = s",
        ylabel = "Standarized Latitude",
    )

    lonlatplot = plot(
        lons,
        lats,
        title = "Path",
        label = labels,
        xlabel = "Standarized Logitude",
        ylabel = "Standarized Latitude",
    )

    display(plot(lonplot, latplot, lonlatplot, layout=grid(3,1)))
end

path1 = traindf[.&(traindf[!, :path_id] .== "l3n", traindf[!, :path_sample] .== "01"), :]
path2 = traindf[.&(traindf[!, :path_id] .== "l5n", traindf[!, :path_sample] .== "01"), :]
l = minimum([nrow(path1), nrow(path2)])
p1m = path1[1:90, :]
p2m = path2[1:90, :]
x = rms(coerce(p1m.magnetometr_x, Continuous), coerce(p2m.magnetometr_x,Continuous))
y = rms(coerce(p1m.magnetometr_y, Continuous), coerce(p2m.magnetometr_y,Continuous))
z = rms(coerce(p1m.magnetometr_z, Continuous), coerce(p2m.magnetometr_z,Continuous))
println(x)
println(y)
println(z)

createcomparisongraphs(
    path1,
    path2,
    axes = ["x", "y", "z"],
    measures = ["magnetometr"],

)

r = traindf[traindf[:, :path_id].=="l1n", :]
createfullgraph(r)
sampleslabels = unique(r[:, :path_sample])
singlesamples = [
    r[r[:, :path_sample].==samplelabel, :]
    for samplelabel in sampleslabels
]
for sample in singlesamples
    maxrow =
end

singlepath = traindf[traindf[:, :path_id].=="l2n", :]
sampleslabels = unique(singlepath[:, :path_sample])
singlesamples = [
    singlepath[singlepath[:, :path_sample].==samplelabel, :]
    for samplelabel in sampleslabels
]
orientx = [x[:, :orientation_x] for x in singlesamples]
orienty = [x[:, :orientation_y] for x in singlesamples]
orientz = [x[:, :orientation_z] for x in singlesamples]
magx = [x[:, :magnetometr_x] for x in singlesamples]
magy = [x[:, :magnetometr_y] for x in singlesamples]
magz = [x[:, :magnetometr_z] for x in singlesamples]
timesteps = [x[:, :timestep] for x in singlesamples]
idxs = [1:nrow(x) for x in singlesamples]
snip = testdf[testdf[:, :path_id].=="tt01", :]
snip = snip[1:166, :]
push!(magx, snip.magnetometr_x)
push!(magy, snip.magnetometr_y)
push!(magz, snip.magnetometr_z)
push!(idxs, 1:nrow(snip))
idxs = idxs ./ 10

Plots.plot(
    timesteps[1],
    [orientx[1], orienty[1], orientz[1]],
    label = ["orientation x" "orientation y" "orientation z"],
    xlabel = "t = [s]",
    w=2,
    layout=3,
    ylim=(-180,180)
)

p1=plot(
    idxs,
    magx,
    w = [1 1 1 1 1 2],
    label = ["train" "train" "train" "train" "train" "test"],
)
p2=plot(
    idxs,
    magy,
    w = [1 1 1 1 1 2],
    label = ["train" "train" "train" "train" "train" "test"],
)
p3=plot(
    idxs,
    magz,
    w = [1 1 1 1 1 2],
    label = ["train" "train" "train" "train" "train" "test"],
)
plot(p1,p2,p3, layout=grid(3,1), size=(900, 1200), xlabel = "t = [s]", title = ["magnetometr_x" "magnetometr_y" "magnetometr_z"])
