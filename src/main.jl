using CSV
using DataFrames
using MagneticLocSuchowiak

test = CSV.read("/home/sebastian/Desktop/PracaInz/train_data.csv", DataFrame)
a = findall(test[:, :path_id] .== "l1n")
df = test[a, :]

b = findall(test[:, :path_id] .== "l3n")
df2 = test[b, :]

createfullgraph(df, measures=["magnetometr", "acceleromter", "orientation"])
createcomparisongraphs(df, df2, measures=["magnetometr", "acceleromter", "orientation"]
)

TRAIN_COLUMNS = [:timestep, :magnetometr_x, :magnetometr_y, :magnetometr_z,
        :acceleromter_x, :acceleromter_y, :acceleromter_z,
        :orientation_x, :orientation_y, :orientation_z,
        :lon, :lat, :path_id, :path_sample]
TEST_COLUMNS = [:timestep, :magnetometr_x, :magnetometr_y, :magnetometr_z, :lon, :lat]
datafile="/home/sebastian/Desktop/PracaInz/data/curves/c4/c4r2r_05.txt"
d=readdatafromfile(datafile, columns_to_get=TRAIN_COLUMNS)
