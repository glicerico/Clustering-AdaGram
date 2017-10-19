# Clustering-AdaGram
Clustering algorithm for AdaGram models

Modified K-means clustering routine for AdaGram models, inspired by Clark's[1] clustering algorithm.
The modified K-means algorithm classifies only the "termination_fraction" fraction of the total word senses (the ones cosine-distance closer to their closest cluster), using their embedding vectors. Clusters are merged when
their center's cosine-distance is lower than "merging_threshold".
```
function clarkClustering(vm::VectorModel, dict::Dictionary, outputFile::AbstractString;
	    K::Integer=100, min_prob=1e-2, termination_fraction=0.8, merging_threshold=0.9,
        fraction_increase=0.05, embeddings_flag=false, embeddings_filename="embeddings.dat")
```
The parameters are:
* vm: trained adagram vector model to use
* dict: dictionary structure to use
* outputFile: name to assign to output file
Optional keyword paramenters:
* K is the number of clusters to be obtained
* min_prob specifies the minimum probability that a sense needs to have to be considered for clustering
* termination_fraction is the percentage of all the word senses that will be clustered
* merging_threshold specifies how close (cosine distance) the centers of two clusters need to be to be merged into one cluster
* fraction_increase determines the percentage increase of words to be clustered in each iteration
* embeddings_flag determines if the embedding vectors for the model's senses will be written to file
* embeddings_filename determines the name of the file to write the model's embedding vectors

[1] Clark, 2000: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.480.9220&rep=rep1&type=pdf