# table-embeddings
CSCI 699 course project: Attribute semantic labeling for data on the web

Use the following to run different models from src directory:
```bash
    python run.py [--h] [--t training_file] [-e testing_file] [-m model]
```

List of models can be found:
   * 0: LSTM with Character CNN for attribute classification
   * 1: Manhattan Siamese LSTM with Character CNN for attribute similarity
   * 2: Hierarchical Attention Network for attribute classification
   * 3: TFIDF (should be an attribute similarity method)
   
Embedding pretrained models can be downloaded from: https://nlp.stanford.edu/projects/glove/
Pickled data files can be downloaded from: https://drive.google.com/file/d/1Hs1hCf4MVy38Kt8ecFCCaK1c_1amEopZ/view?usp=sharing
