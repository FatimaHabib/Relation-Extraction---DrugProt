# Relation-Extraction---DrugProt

This reposetory cosists of three folder contains three models to perform the task of relation extraction between chemical and protien entities. 

Each folder contains train file. 
To train the models:

1. CNN : 
      1.1 Download the pre-traind embedding model: BioWordVec vector 13GB (200dim, trained on PubMed+MIMIC-III, word2vec bin format) from 
           https://github.com/ncbi-nlp/BioSentVec
      2.2 Build vocabulary: 
          python3 build_vocab.py
      3.3 Train the model
  
          python3 train.py  const_vec high_num_sent 
          dec_vec: stands for padding the position vectors 
          
2.SciBERT+CNN and SciBERT+LSTM+CNN:
                   
          python3 train.py 16 50 no_cv 0.00002 
   
   batch size, number of epochs, training with or withot cross validation set to Y_CV if you want to train it with CV, learning rat
   
