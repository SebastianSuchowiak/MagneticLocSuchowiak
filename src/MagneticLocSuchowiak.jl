module MagneticLocSuchowiak

using PyPlot
using DataFrames
using CSV
using GLM
using MLJ
using MLJModels
using NearestNeighbors
using StableRNGs
using Printf
using Distances
using Plots
using DecisionTree
using EvoTrees
using GLM
using LIBSVM
using ScikitLearn
using LightGBM
using MLJLinearModels


include("graphs.jl")
include("datareader.jl")
include("models.jl")


export createfullgraph, createcomparisongraphs, readdata, fitandmean!

end
