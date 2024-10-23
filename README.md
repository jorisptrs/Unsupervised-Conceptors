## Extracting Concepts From Neural Networks Using Conceptor-based Clustering

### Setup

To get the code to run, make sure to:
1. Download the TIMIT dataset (e.g., from https://catalog.ldc.upenn.edu/LDC93S1, or Kaggle) and place it into the
root directory. The data folder should be named TIMIT. This may be ignored when relying on the cahced files in 'code/cache'
2. Install python 3.10 or higher
3. Install Anaconda
4. Run `conda env create -f environment.yml` from the root directory to load the dependencies into a new environment
Now, you should be able to use the jupyter notebooks.

### Abstract

Conceptors are versatile neuro-symbolic formalizations of concepts as they arise in neural networks, with promising results on supervised tasks. However, the use of conceptors in unsupervised settings remains largely unexplored. Meanwhile, previous brain science and AI research used clustering to extract concepts from neural representations. This study combines conceptor-based representations with clustering methods for the unsupervised extraction of human- meaningful and coherent concepts from the activations of neural networks. Concretely, experiments are conducted on the responses of an Echo State Network (ESN), a type of recurrent neural network, to phoneme utterances from the TIMIT Acoustic-Phonetic Continuous Speech Corpus. In preparation, conceptor-based classification was demonstrated, and ESN hyperparameters were tuned. Then, two clustering methods, generalized centroid-based hard clustering and hierarchical agglomerative clustering, are adapted to work with conceptors and extract concepts from the ESN’s responses. The resulting concepts and concept hierarchies were significantly human-meaningful, resembling established phonetic categories, and coherent. Conceptor-based clustering, although in its infancy, represents a promising approach to unsupervised concept extraction and forming conceptors without supervision. Applications in neuro-symbolic computational creativity, brain sciences, time-series clustering, and neural network explainability are suggested.
