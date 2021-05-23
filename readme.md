

This project includes two major parts -- 1) attention models of protein sequences and structures 2) protein-protein interactions. 

### attention models of protein sequences and structures 

train Bert and GPT like language models and transfer the model to various tasks. 

Different ways of attentions are explored here. 
One new idea is to add the distance matrix of the homology structure to the model of sequences. 

### protein-protein interactions
Here the idea is to transfer the model based on block-block interactions to protein-protein interactions.

1) extract block-block interactions from the structure of single protein chain. 
   Here a block is a sub-domain sequence segment. Block pairs are extracted from protein structures. 

2) train models of block-block interactions.
The first model is a sequence based model, ie. to predict whether two sequence segment are likely to interact given only their sequences.
In the second model, structure information (the distance matrices) of both input blocks are also given, and the goal is to predict the distance map of their interaction. 

3) transfer the model of block-block interactions to protein-protein interactions. 
The transferability of current models is very poor, ie. they don't work in protein-protein interactions. :( 




