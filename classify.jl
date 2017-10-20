push!(LOAD_PATH, "./src/")

using ArgParse
using AdaGram
include("src/util.jl")

s = ArgParseSettings()

@add_arg_table s begin
  "input"
    help = "AdaGram model to use"
    arg_type = AbstractString
    required = true
  "output"
    help = "name of clustered output file"
    arg_type = AbstractString
    required = true
  "--k"
    help = "number of clusters to use in modified k-means"
    arg_type = Int64
    default = 100
  "--min-prob"
    help = "lower threshold to include a meaning"
    arg_type = Float64
    default = 1e-2
  "--termination-fraction"
    help = "perecentage of senses to cluster"
    arg_type = Float64
    default = 0.8
  "--merging-threshold"
    help = "cosine-distance to merge clusters"
    arg_type = Float64
    default = 0.9
  "--fraction-increase"
    help = "clustering fraction iteration step"
    arg_type = Float64
    default = 0.05
  "--embeddings-flag"
    help = "activates embedding vectors output"
    action = :store_true
  "--embeddings-filename"
    help = "filename for embeddings vector file"
    arg_type = AbstractString
    default = "embeddings.dat"
end

args = parse_args(ARGS, s)

print("Starting clustering...")

vm, dict = load_model(args["input"])
clarkCluster(vm, dict, args["output"]; K = args["k"], min_prob = args["min-prob"], 
  termination_fraction = args["termination-fraction"], merging_threshold = args["merging-threshold"],
  fraction_increase = args["fraction-increase"], 
  embeddings_flag = args["embeddings-flag"], embeddings_filename = args["embeddings-filename"])
println("Done!")