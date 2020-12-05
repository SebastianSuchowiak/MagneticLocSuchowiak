using CSV
using DataFrames
using MagneticLocSuchowiak
using MLJ
using NearestNeighbors

DATA_PATH = "/home/sebastian/Desktop/PracaInz/data"
CURVES = DATA_PATH * "/curves"
LINES = DATA_PATH * "/lines"
TESTS = DATA_PATH * "/tests"
train_dirs = vcat(readdir(LINES, join=true), readdir(CURVES, join=true))
train_files = collect(Iterators.flatten(readdir.(train_dirs, join=true)))
test_files = readdir(TESTS, join=true)

test = CSV.read("/home/sebastian/Desktop/PracaInz/train_data.csv", DataFrame)
a = findall(test[:, :path_id] .== "l1n")
df = test[a, :]

b = findall(test[:, :path_id] .== "l3n")
df2 = test[b, :]

createfullgraph(df, measures=["magnetometr", "acceleromter", "orientation"])
createcomparisongraphs(df, df2, measures=["magnetometr", "acceleromter", "orientation"]
)

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

@load KNNRegressor
r1 = range(KNNRegressor(), :K, lower=1, upper=20);
knn_x = TunedModel(model=KNNRegressor(), resampling=folds, tuning=Grid(resolution=10), range=[r1], measure=l2);
knn_y = TunedModel(model=KNNRegressor(), resampling=folds, tuning=Grid(resolution=10), range=[r1], measure=l2);
fitandmean!(knn_x, knn_y, traindf, testdf)
