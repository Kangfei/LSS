LSS: A Learned Sketch for Subgraph Counting
-----------------
A PyTorch + torch-geometric implementation of LSS, as described in the paper:
Kangfei Zhao, Jeffrey Xu Yu, Hao Zhang, Qiyan Li, Yu Rong. [A Learned Sketch for Subgraph Counting](https://dl.acm.org/doi/10.1145/3448016.3457289)

### Requirements
```
python 3.7
networkx
numpy
scipy
scikit-learn
torch >= 1.5.0
torch-geometric 1.6
```

Import the conda environment by running
```
conda env create -f gpcard.yml
conda activate gpcard
```



### Usage
Running cross validation
```
python active_train.py    \
       --dataset aids     \
       --embed_type prone \
       --mode cross_val   
```
Running active learning
```
python active_train.py \
       --dataset aids \
       --mode train \
       --embed_type prone \
       --active_iters 1 \
       --uncertainty consist
```
Running ensemble (active) learning
```
python active_train.py \
       --dataset aids \
       --mode ensemble \
       --embed_type prone \
       --ensemble_num 5 \
       --active_iters 1 
```

### Key Parameters
All the parameters with their default value are in active_train.py

| name | type   | description | 
| ----- | --------- | ----------- |
| model_type | String |  type of GCN layer (GIN, GINE, GAT, NN, GCN, SAGE, NNGIN, NNGINConcat)           |
| num_layers  | Int    | number of GNN layers    |
| embed_type | String | the node feature encoding, (freq, n2v, prone, nrp, prone_concat)|
| pool_type  | String | type of the substructure pooling layer (att, mean, max, sum)  |
| epochs  | Int   | number of training epochs  |
| batch_size  | Int   | mini-batch size for sgd  |
| num_fold  | Int   | number of folder for cross validate |
| k  | Int   | number of hops to decompose query graphs |
| learning rate | Float   | learning rate  |
| multi_task    | Bool | enable/disable the count classification task, for active learning|
| num_ensemble  | Int   | number of LSS modes for model ensemble |
| active_iters  | Int   | number of iteration for active learning |
| budget  | Int   | number of query to selected for each iteration of active learning |
| uncertainty | String | uncertainty function for active learning, e.g., entropy, margin, confident |


### Project Structure
```
active_train.py # project extrance, parameters settings
train.py        # deprecated codes for only supervied training
QuerySampler.py # data and query graph preprocessing, query decomposition
Queryset.py     # input encoding, torch dataloader
util.py         # model load/save
baseline.py     # old codes for result evaluation
    /active
        active_util.py
        ActiveLearner.py             # source of the active learner
    /cardnet                    # source of the LSS model
        GINlayers.py                 # revised GIN layers (NNGIN, NNGINConcat)
        layers.py                    # basic layers
        model.py                     # LSS model
    /scripts                    # old scripts for testing
    /data                       # aids dataset sample to run the code
        /dataset/aids                # graph data
        /queryset_homo/aids          # queries
        /true_homo/aids              # true count of the queries in queryset_homo
        /prone/aids                  # prone embdding matrix and node mapping file
```
To use your own dataset, you can put the data graphs, queries, true counts, and embedding to 
'/data/dataset/DATASET_NAME', '/data/queryset_homo/DATASET_NAME', '/data/true_homo/DATASET_NAME', '/data/prone/DATASET_NAME', respectively, and set the args.dataset to DATASET_NAME in 'active_train'.py  

The format of input query graph and data graph follows [GCARE](https://github.com/yspark-dblab/gcare).

### Contact
Open an issue or send email to zkf1105@gmail.com if you have any problem

### Cite Us
```
@inproceedings{DBLP:conf/sigmod/ZhaoYZLR21,
  author    = {Kangfei Zhao and
               Jeffrey Xu Yu and
               Hao Zhang and
               Qiyan Li and
               Yu Rong},
  title     = {A Learned Sketch for Subgraph Counting},
  booktitle = {{SIGMOD} '21},
  pages     = {2142--2155},
  publisher = {{ACM}},
  year      = {2021}
}
```