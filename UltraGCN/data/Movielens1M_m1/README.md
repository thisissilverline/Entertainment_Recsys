# Movielens1M_m1

+ **Dataset description:**
  
  The MovieLens-1M dataset contain 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users. We follow the LCF work to split and preprocess the data into training, validation, and test sets, respectively.

+ **Data format:**

  Each user corresponds to a list of interacted items: [[item1, item2], [item3, item4, item5], ...]

+ **Source:** https://grouplens.org/datasets/movielens/1m/
+ **Download:** https://huggingface.co/datasets/reczoo/Movielens1M_m1/tree/main
+ **RecZoo Datasets:** https://github.com/reczoo/Datasets


+ **Check the md5sum for data integrity:**
  ```bash
  $ md5sum *.json
  cdd3ad819512cb87dad2f098c8437df2  test_data.json
  4229bc5369f943918103daf7fd92e920  train_data.json
  60be3b377d39806f80a43e37c94449f6  validation_data.json
  ```
