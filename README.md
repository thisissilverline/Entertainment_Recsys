# Entertainment_Recsys
[Project] Entertainment Recommendation System

팀원)
인공지능융합학과 성아영
인공지능융합학과 최은선


### Environment

python==3.7.6

  '''bash
  pip install torch==1.10.2+cu102, numpy==1.19.5, scipy==1.5.2, tensorboard==2.11.2
  '''
or
  '''bash
  pip install torch==1.11.0+cu102, numpy==1.21.6, scipy==1.7.3, tensorboard==2.11.2
  '''




### Results on Amazon-Books

| Model                                                     | Recall@20  | NDCG@20    |
|:--------------------------------------------------------- |:----------:|:----------:|
| NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]       | 0.0344     | 0.0263     |
| LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)]   | 0.0411     | 0.0315     |
| SGL-ED [[SIGIR'21](https://arxiv.org/pdf/2010.10783.pdf)] | 0.0478     | 0.0379     |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]    | **0.0681** | **0.0556** |

+ Follow the script below to reproduce the results
  
  ```bash
  python {main.py} --config_file ./config/{ultragcn_amazonbooks_m1.ini}
  ```

+ See the running log: [results/{ultragcn_amazonbooks_m1.log}](./results/{ultragcn_amazonbooks_m1.log}) 


### Results on Movielens-1M

| Model                                                   | F1@20      | NDCG@20    | Recall@20  |
|:------------------------------------------------------- |:----------:|:----------:|:----------:|
| NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]     | 0.1582     | 0.2511     | 0.2513     |
| LCFN [[ICML'20](https://arxiv.org/abs/2006.15516)]      | 0.1625     | 0.2603     |            |
| LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)] |            | 0.2427     | 0.2576     |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]  | **0.2004** | **0.2642** | **0.2787** |

+ Follow the script below to reproduce the results
  
  ```bash
  # convert data format
  cd data/Movielens1M_m1
  python convert_data.py
  
  python {main.py} --config_file ./config/{ultragcn_movielens1m_m1.ini}
  ```

+ See the running log: [results/{ultragcn_movielens1m_m1.log}](./results/{ultragcn_movielens1m_m1.log}) 






Reference : Mao, K., Zhu, J., Xiao, X., Lu, B., Wang, Z., & He, X. (2021, October). UltraGCN: ultra simplification of graph convolutional networks for recommendation. In Proceedings of the 30th ACM international conference on information & knowledge management (pp. 1253-1262).
