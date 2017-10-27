# Visualize embeddings of word vectors stored in
# input_file, with metadata stored in metadata_file
#
# Usage: 
# 1. start a virtual environment with:
#       source ~/tensorflow/bin/activate
# 2. run this script with:
#       python3 visualizeEmbeddings -i <inputFile> [-m <metadataFile>]
# 3. run tensorboard with:
#       tensorboard --logdir=/tmp/tensorflow/logs
# 4. open http://localhost:6006/#embeddings in web browser

#!usr/bin/python3

import os
import sys, getopt
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

def main(argv):

    inputFile = ''
    metadataFile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:m:", ["ifile=","mfile="])
    except getopt.GetoptError:
        print('visualize_embeddings.py -i <inputFile> [-m <metadataFile>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('visualize_embeddings.py -i <inputFile> [-m <metadataFile>]')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputFile = arg
        elif opt in ("-m", "--mfile"):
            metadataFile = arg
    
    LOG_DIR = '/tmp/tensorflow/logs'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    with open(inputFile, 'r') as dataFile:
        data = [[float(e) for e in r.split()] for r in dataFile]

    embeddings = tf.Variable(data, name='wordEmbeddings')

    with tf.Session() as sess:
        saver = tf.train.Saver([embeddings])

        sess.run(embeddings.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'embeddings.ckpt'))

        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = embeddings.name

        if metadataFile != '':
            embedding.metadata_path = os.getcwd() + '/' + metadataFile

        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

if __name__ == "__main__":
    main(sys.argv[1:])
