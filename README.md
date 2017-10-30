# Clustering-AdaGram
Clustering-AdaGram is a Julia-implemented clustering algorithm for AdaGram models.
It's a modified K-means clustering routine for AdaGram models, inspired by Clark's[1] clustering algorithm.
The modified K-means algorithm iteratively classifies only a fraction of the total word senses (the fraction that is cosine-distance closer to their closest cluster), using their embedding vectors. Clusters are merged when
their center's cosine-distance is higher than a specified threshold.

## Installation

Clustering-AdaGram has been tested in Julia v0.4.5; it should be installed in the following way:
```
Pkg.clone("https://github.com/glicerico/Clustering-AdaGram.git")
Pkg.build("Clustering-AdaGram")
```

## Clustering an AdaGram model

The most straightforward way to cluster a model is to use `classify.sh` script. If you run it with no parameters passed or with `--help` option, it will print usage information:
```
usage: classify.jl [--k K] [--min-prob MIN-PROB]
                   [--termination-fraction TERMINATION-FRACTION]
                   [--merging-threshold MERGING-THRESHOLD]
                   [--fraction-increase FRACTION-INCREASE]
                   [--embeddings-flag]
                   [--embeddings-filename EMBEDDINGS-FILENAME] input
                   output
```
Here is the description of all parameters:
* K: number of clusters to use in modified k-means
* MIN_PROB: minimum probability that a sense needs to have to be considered for clustering
* TERMINATION_FRACTION: percentage of all the word senses that will be clustered
* MERGING_THRESHOLD: how close the centers of two clusters need to be to be merged into one cluster. If their cosine-distance is higher than this, they merge.
* FRACTION_INCREASE: percentage increase of fraction of words to be clustered in each iteration.
* embeddings_flag: allows to write to file the embedding vectors for the model's senses, for visualization purposes.
* EMBEDDINGS_FILENAME: name of the file to write the model's embedding vectors
* input: AdaGram model to use
* output: name of clustered output file

[1] Clark, 2000: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.480.9220&rep=rep1&type=pdf

## Generating word embeddings to visualize
Instructions for using TensorBoard to visualize word embeddings and, optionally, their clustering

\begin{itemize}
    \item You need to have a trained AdaGram model (either regular AdaGram or AdaGramWP, does not matter as both processes produce a model with the same format).

    \item Install TensorFlow. Follow steps for Ubuntu install in the following link. I used virtualenv installation in directory $\sim$/tensorflow:\\
    https://www.tensorflow.org/install/install\_linux

    \item Perform clustering to get word embedding vectors: run Clustering\_AdaGram on AdaGram model with preferred parameters and keyword argument embeddings\_flag=true (and optionally with keyword argument embeddings\_filename)
    \item From folder where embeddings are stored, run:
    visualizeEmbeddings.sh -i EMBEDDINGS\_FILE [-m CLUSTERS\_FILE]
    \item Open web browser and go to localhost:6006, then choose the Embeddings or Projector tab.

\end{itemize}
