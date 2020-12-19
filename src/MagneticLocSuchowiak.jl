__precompile__(false)

module MagneticLocSuchowiak

using Plots
using DataFrames
using CSV
using MLJ
using Distances
using Statistics

@load XGBoostRegressor
@load NeuralNetworkRegressor
@load KNNRegressor
@load NuSVR
@load RandomForestRegressor pkg=DecisionTree
#@load LGBMRegressor


include("graphs.jl")
include("datareader.jl")
include("models.jl")


export createfullgraph, createcomparisongraphs, readdata, fitknn, testmachines, fitforest, fitnusvr, fitcustommodel

end
