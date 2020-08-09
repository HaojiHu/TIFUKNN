# TIFUKNN

This is our implementation for the paper: 

Haoji Hu, Xiangnan He, Jinyang Gao, Zhi-Li Zhang (2020). [Modeling Personalized Item Frequency Information for Next-basket Recommendation.](https://dl.acm.org/doi/pdf/10.1145/3397271.3401066) In the 43th International ACM SIGIR Conference on Research and Development in Information Retrieval.

**Please cite our paper if you use our codes and datasets. Thanks!** 
```
@inproceedings{hu2020TIFUKNN,
  title={Modeling Personalized Item Frequency Information for Next-basket Recommendation},
  author={Hu, Haoji and He, Xiangnan and Gao, Jinyang and Zhang, Zhi-Li},
  booktitle={Proceedings of the 43th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020},
  organization={ACM}
}
```

Author: Haoji Hu

## Environment Settings
- Python version: '3.6.8'

## A quick start to run the codes with Ta-Feng data set.


```
python TIFUKNN.py ./data/TaFang_history_NB.csv ./data/TaFang_future_NB.csv 300 0.9 0.7 0.7 7 10
```

TaFang_history_NB.csv contains the historical records of all the customers. TaFang_future_NB.csv contains the future records of all the customers. The 300 is the number neighbors. 0.9 is the time-decayed ratio within each group. The first 0.7 is the time-decayed ratio accross groups. The second 0.7 is the alpha for combining two parts in prediction. 7 is the group size. 10 is the top k items recommened.


