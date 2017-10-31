#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

source ~/tensorflow/bin/activate
python $DIR/visualizeEmbeddings.py $@
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "open http://localhost:6006/#embeddings in web browser"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
tensorboard --logdir=/tmp/tensorflow/logs

