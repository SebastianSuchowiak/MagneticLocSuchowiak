using CSV
using DataFrames
using MagneticLocSuchowiak
using MLJ
using MLJModels
using NearestNeighbors
using GLM
using StableRNGs
using Plots

DATA_PATH = "C:/Users/Sebastian/PracaInz/data"
CURVES = DATA_PATH * "/curves"
LINES = DATA_PATH * "/lines"
TESTS = DATA_PATH * "/tests"
train_dirs = vcat(readdir(LINES, join=true), readdir(CURVES, join=true))
test_files = readdir(TESTS, join=true)
train_files = collect(Iterators.flatten(readdir.(train_dirs, join=true)))

test = CSV.read("C:/Users/Sebastian/PracaInz/train_data.csv", DataFrame)
a = findall(test[:, :path_id] .== "l1n")
df = test[a, :]

b = findall(test[:, :path_id] .== "l3n")
df2 = test[b, :]

createfullgraph(df, measures=["magnetometr", "acceleromter", "orientation"])
createcomparisongraphs(df, df2, measures=["magnetometr", "acceleromter", "orientation"])

ALL_COLUMNS = [:timestep, :magnetometr_x, :magnetometr_y, :magnetometr_z,
        :acceleromter_x, :acceleromter_y, :acceleromter_z,
        :orientation_x, :orientation_y, :orientation_z,
        :lon, :lat, :path_id, :path_sample]
TRAIN_COLUMNS = [:timestep, :magnetometr_x, :magnetometr_y, :magnetometr_z,
        :lon, :lat, :path_id, :path_sample]
TEST_COLUMNS = [:timestep, :magnetometr_x, :magnetometr_y, :magnetometr_z, :lon, :lat]
traindf = readdata(train_files, columns_to_get=TRAIN_COLUMNS)
testdf = readdata(test_files, columns_to_get=TEST_COLUMNS, add_tag=false)

folds = MagneticLocSuchowiak.getfolds(traindf)

using DecisionTree
knn = @load RandomForestRegressor pkg=DecisionTree
r1 = range(knn, :n_trees; upper=20, lower=1, scale=:log);
knn_x = TunedModel(model=knn, resampling=folds, tuning=Grid(resolution=10), range=[r1], measure=l2);
knn_y = TunedModel(model=knn, resampling=folds, tuning=Grid(resolution=10), range=[r1], measure=l2);

y, X = MagneticLocSuchowiak.preparedata(traindf)
y1 = coerce(y[:,1], Continuous)

m_x = machine(knn_x, X, y1);
m_y = machine(knn_x, X, y[:, 2]);

println(scitype(X))
println(scitype(y1))
println(typeof(y1))
println(typeof(X))
fit!(m_x)
fit!(m_y, verbosity=0)

fitandmean!(knn_x, knn_y, traindf, testdf)
