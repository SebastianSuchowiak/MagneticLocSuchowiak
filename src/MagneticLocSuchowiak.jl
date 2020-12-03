module MagneticLocSuchowiak

using PyPlot
using DataFrames
using CSV
using GLM


include("graphs.jl")
include("datareader.jl")


export createfullgraph, createcomparisongraphs, readdatafromdirs, readdatafromfile

end
