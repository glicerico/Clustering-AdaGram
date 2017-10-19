using AdaGram
import AdaGram.Dictionary
import AdaGram.VectorModel

# Performs clustering using K-means algorithm adapted from word2vec
# clustering routine, but handling the representation vector for each
# different significant meaning of a word. A word can (and probably should)
# end up in different clusters, according to its different meanings.
function clustering(vm::VectorModel, dict::Dictionary, outputFile::AbstractString,
        K::Integer=100; min_prob=1e-3)
	wordVectors = Float32[]
	words = AbstractString[]

	# Builds arrays of words and their vectors
	for w in 1:V(vm)
		probVec = expected_pi(vm, w)
		for iMeaning in 1:T(vm)
			# ignores senses that do not reach min probability
			if probVec[iMeaning] > min_prob
				push!(words, dict.id2word[w])
				currentVector = vm.In[:, iMeaning, w]
				for currentValue in currentVector 
					push!(wordVectors, currentValue)
				end
			end
		end
	end

	# Calls the actual classifier, from a c-function
	ccall((:kmeans, "superlib"), Void,
	    (Ptr{Ptr{Cchar}}, Ptr{Float32},
	    	Int, Int, Int, Ptr{Cchar}), 
	    words, wordVectors, K, size(words, 1), M(vm), outputFile)

	println("Finished clustering")
end

# clustering routine using k-means with cosine distance, modified in the spirit of Clark2000 to account
# for words that don't clearly fit in a cluster, as well as cluster merging.
# Added tag_flag to find most appropriate tag from separate tagged dictionary file (tagged
# with Link Grammar) and write it in the clustered file (for evaluation of clusters purposes).
# Added embeddings_flag to write to file the embedding vectors of the words (for visualization purposes).
function clarkCluster(vm::VectorModel, dict::Dictionary, outputFile::AbstractString;
	    K::Integer=15, min_prob=1e-2, termination_fraction=0.8, merging_threshold=0.9,
        fraction_increase=0.05, tag_flag = false, dict_path = "null", min_freq = 1, 
        embeddings_flag = false, embeddings_filename = "embeddings.dat")
    embeddings = [] # non-normalized embeddings
    wordVectors = []
    senses = Int64[]
    senseFrequencies = Int64[]
    clusters = []

    function calculateCenter(currentCluster::Int64)
        currentCenter = zeros(Float32, M(vm))
        for iMember in 1:length(clusters[currentCluster])
            currentCenter += wordVectors[clusters[currentCluster][iMember]]
        end
        #currentCenter /= length(clusters[currentCluster]) # averages the centers of every member of the class
        currentCenter /= norm(currentCenter) # normalizes the center vector
        return currentCenter
    end

    # Creates a dictionary that will store each word root in the
    # dict_path provided, with its different tags
    # and corresponding frequency ranking
    function loadTaggedDict(dict_path::AbstractString, min_freq::Int)
        wordDict = Dict()
        fi = open(dict_path, "r")
        while !eof(fi)
            line = split(readline(fi))
            sense = line[1]
            senseFreq = parse(Int, line[2])

            # only stores words that occur at least min-freq times
            if senseFreq >= min_freq
                # look for the tag
                splitWord = rsplit(UTF8String(sense), "[", limit = 2)
                taggedPart = rsplit(UTF8String(splitWord[end]), ".", limit = 2)
                #finalTaggedPart = rsplit(UTF8String(taggedPart[end]), "#", limit = 2)

                if length(taggedPart) > 1
                #    tag = finalTaggedPart[end]
                    tag = taggedPart[end]
                else
                    tag = "untagged"
                end
                # look for the root word
                if length(splitWord) > 1
                    wordKey = splitWord[1]
                else
                    wordKey = taggedPart[1]
                end

                if !haskey(wordDict, wordKey)
                    wordDict[wordKey] = []
                end
                push!(wordDict[wordKey], [tag, senseFreq])
            end
        end

        # calculates frequency ranking of tag compared to other
        # tags for the same word
        for iEntry in wordDict
            suma = 0
            for iValue in iEntry[2]
                suma += iValue[2]
            end
            for iValue in iEntry[2]
                iValue[2] /= suma
            end
        end
        close(fi)
        return wordDict
    end

    ###########################################
    # Clustering routine starts

    # Builds arrays of senses and their vectors
    for w in 1:V(vm)
        probVec = expected_pi(vm, w)
        for iMeaning in 1:T(vm)
            # ignores senses that do not reach min probability
            if probVec[iMeaning] > min_prob
                push!(senses, w)
                push!(senseFrequencies, round(vm.counts[iMeaning, w]))
                currentVec = vm.In[:, iMeaning, w]
                push!(embeddings, currentVec)
                push!(wordVectors, currentVec/norm(currentVec)) # normalizes wordVectors
            end
        end
    end


    numSenses = length(senses) # total num of unique senses to cluster
    orderFreq = sortperm(senseFrequencies, rev = true) # ordered indexes of most freq. senses

    # Initialize clusters with the next most frequent sense available
    # and cluster centers with zeros
    clusterCenters = []
    for iCluster in 1:K
        push!(clusters, [orderFreq[iCluster]])
        push!(clusterCenters, zeros(Float32, M(vm))) 
    end

    # initialize closestCluster, closestClusterDistance
    closestCluster = zeros(Int32, numSenses)
    closestClusterDistance = zeros(Float32, numSenses)

    numClusteredSenses = K 
    orderDistance = Int32[]
    # keeps clustering senses until termination_fraction of them are clustered
    while numClusteredSenses <= numSenses * termination_fraction
        # calculate cluster centers
        for iCluster in 1:K
            clusterCenters[iCluster] = calculateCenter(iCluster)
        end

        mergeFlag = false
        # If two clusters are close enough, merge them and activate flag to return to loop start
        for iCluster in 1:K - 1
            for iCluster2 in iCluster + 1:K
                separation = dot(clusterCenters[iCluster], clusterCenters[iCluster2])
                if separation > merging_threshold
                    append!(clusters[iCluster], clusters[iCluster2])
                    # Resets merged cluster to highest freq. unclustered sense
                    numClusteredSenses += 1
                    clusters[iCluster2] = [orderFreq[numClusteredSenses]]
                    println("Merged clusters $iCluster and $iCluster2, reassigned the latter")
                    mergeFlag = true
                    break
                end
            end
            if mergeFlag break end
        end
        if mergeFlag continue end

        # calculate each sense's projection to cluster centers, only keep the closest one
        for iWord in 1:numSenses
            projection = -Inf
            clusterId = 0
            for iCluster in 1:K
                dotProd = dot(wordVectors[iWord], clusterCenters[iCluster]) 
                if dotProd > projection
                    projection = dotProd
                    clusterId = iCluster
                end
            end
            closestCluster[iWord] = clusterId
            closestClusterDistance[iWord] = projection # cosine distance, larger is closer
        end

        # get sense order relative to distance to their nearest cluster
        orderDistance = sortperm(closestClusterDistance, rev = true)
        numClusteredSenses += round(Int32, fraction_increase * numSenses)
        # reset clusters to allow membership change
        clusters = []
        for iCluster in 1:K
            push!(clusters, [])
        end
        # assign the best senses as members of their closest cluster
        for iBest in 1:numClusteredSenses
            push!(clusters[closestCluster[orderDistance[iBest]]], orderDistance[iBest])
        end

        @printf "Percentge of word senses clustered: %0.3f \n" numClusteredSenses/numSenses
    end

    # the less frequent senses fall into a cluster in position K+1 (unclustered senses)
    push!(clusters, orderDistance[numClusteredSenses + 1:end])
    println("Cluster size percentage:")
    for i in 1:K + 1
        @printf("%0.2f ", length(clusters[i])/numSenses * 100)
    end
    @printf("\n")

    # write results to file(s)
    fo = open(outputFile, "w")
    # different routes if words must be tagged or not
    if tag_flag
        counterNotFound = 0
        println("Writing tagged clusters file")
        wordDict = loadTaggedDict(dict_path, min_freq)
        for iCluster in 1:length(clusters)
            for iMember in 1:length(clusters[iCluster])
                wordId = senses[clusters[iCluster][iMember]] 
                word = dict.id2word[wordId]
                notFound = false

                # ranks the sense relative to the total word count
                senseFreq = senseFrequencies[clusters[iCluster][iMember]]
                senseRank = senseFreq / vm.frequencies[wordId]

                if !haskey(wordDict, word)
                    # try to find lowercase version
                    word = string(lowercase(word[1]), word[2:end])
                    if !haskey(wordDict, word)
                        #println("$word not found in dictionary")
                        counterNotFound += 1
                        notFound = true
                    end
                end
                if !notFound
                    # compares sense rank to the rank of the tags for the word
                    # to assign most adequate tag
                    arrayTags = wordDict[word]
                    rankDistance = Inf
                    tag = "" # just to avoid UndefVarError
                    for iTag in arrayTags
                        currentDistance = abs(iTag[2] - senseRank) 
                        if currentDistance < rankDistance
                            rankDistance = currentDistance
                            tag = iTag[1]
                        end
                    end
                    taggedWord = string(word, ".", tag)
                else 
                    taggedWord = string(word, ".", "notFound")
                end
                @printf(fo, "%s\t%d\n", taggedWord, iCluster)
            end
        end
        println("$counterNotFound words not found in dictionary")
    else
        @printf(fo, "Word\tClusterNbr\n")
        # decides if write embeddings or not
        if embeddings_flag
            fEmbeddings = open(embeddings_filename, "w")
            # write to specified output file
            println("Writing clusters file")
            for iCluster in 1:length(clusters)
                for iMember in 1:length(clusters[iCluster])
                    @printf(fo, "%s\t%d\n", dict.id2word[senses[clusters[iCluster][iMember]]], iCluster)
                    for iDim in 1:M(vm)
                        @printf(fEmbeddings, "%f ", embeddings[clusters[iCluster][iMember]][iDim])
                    end
                    @printf(fEmbeddings, "\n")
                end
            end
            close(fEmbeddings)
        else
            # write to specified output file
            println("Writing clusters file")
            for iCluster in 1:length(clusters)
                for iMember in 1:length(clusters[iCluster])
                    @printf(fo, "%s\t%d\n", dict.id2word[senses[clusters[iCluster][iMember]]], iCluster)
                end
            end
        end
    end

    println("Finished writing clusters file")
    close(fo)
end

# Writes embeddings to file to be used for visualization with TensorBoard
function writeEmbeddings(vm::VectorModel, dict::Dictionary, embeddings_file::AbstractString; min_prob = 1e-2)
    fo = open(embeddings_file, "w")
    fMetadata = open(string("metadata_", embeddings_file), "w")
    for iWord in 1:V(vm)
        probVec = expected_pi(vm, iWord)
        for iSense in 1:T(vm)
            if probVec[iSense] > min_prob
                for iDim in 1:M(vm)
                    @printf(fo, "%f ", vm.In[iDim, iSense, iWord])
                end
                @printf(fo, "\n")
                @printf(fMetadata, "%s\n", dict.id2word[iWord])
            end
        end
    end
    close(fo)
    close(fMetadata)
end

export clustering, clarkClustering, writeEmbeddings