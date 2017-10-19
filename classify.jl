push!(LOAD_PATH, "./src/")

using ArgParse
using AdaGram
include("src/util.jl")

s = ArgParseSettings()

@add_arg_table s begin
  "input"
    help = "file where the word embeddings are saved"
    arg_type = AbstractString
    required = true
  "output"
    help = "file to save the clustering (in text format)"
    arg_type = AbstractString
    required = true
  "--k"
    help = "number of clusters to use in k-means"
    arg_type = Int64
    default = 100
  "--min-prob"
    help = "lower threshold to include a meaning"
    arg_type = Float64
    default = 1e-2
end

args = parse_args(ARGS, s)


print("Starting clustering...")

vm, dict = load_model(args["input"])
clarkCluster(vm, dict, args["output"]; K = args["k"], min_prob = args["min-prob"])
println("Done!")