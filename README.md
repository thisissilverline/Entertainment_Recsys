# Entertainment_Recsys
[Project] Entertainment Recommendation System

팀원)
인공지능융합학과 성아영
인공지능융합학과 최은선


### Environment

python==3.7.6

  ```bash
  pip install torch==1.10.2+cu102, numpy==1.19.5, scipy==1.5.2, tensorboard==2.11.2
  ```
  
or

  ```bash
  pip install torch==1.11.0+cu102, numpy==1.21.6, scipy==1.7.3, tensorboard==2.11.2
  ```




### Results on Amazon-Books

| Model                                                     | Recall@20  | NDCG@20    |
|:--------------------------------------------------------- |:----------:|:----------:|
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]    | **0.0679** | **0.0555** |
| UltraGCN + Degree Centrality                              | **0.0678** | **0.0553** |

+ Follow the script below to reproduce the results
  
  ```bash
  python {main.py} --config_file ./config/{ultragcn_amazonbooks_m1.ini}
  ```


### Results on Movielens-1M

| Model                                                   | Recall@20  | NDCG@20    |
|:------------------------------------------------------- |:----------:|:----------:|
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]  | **0.2769** | **0.2628** |
| UltraGCN + Degree Centrality                            | **0.2784** | **0.2634** |
| UltraGCN + Top-k negative sampling                      | **0.2800** | **0.2669** |


+ Follow the script below to reproduce the results
  
  ```bash
  # convert data format
  cd data/Movielens1M_m1
  python convert_data.py
  
  python {main.py} --config_file ./config/{ultragcn_movielens1m_m1.ini}
  ```






Reference : Mao, K., Zhu, J., Xiao, X., Lu, B., Wang, Z., & He, X. (2021, October). UltraGCN: ultra simplification of graph convolutional networks for recommendation. In Proceedings of the 30th ACM international conference on information & knowledge management (pp. 1253-1262).
