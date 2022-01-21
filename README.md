# FIRE
This repository is the implementation of the paper: FIRE: Fast Incremental Recommendation with Graph Signal Processing (TheWebConf 2022).

If you use the code in your work, please cite our paper.



### How to run the code

##### Step 1: Check the compatibility of your python packages, we recommend you to use the following setting that has passed our test if you want to reproduce the results.
```python
numpy == 1.19.4
pandas == 1.1.4
python == 3.6.11
scipy == 1.5.3
sklearn == 0.23.2
sparsesvd == 0.2.2
```

##### Step 2: prepare the dataset.

* If you use the default datasets (```Movielens 1M``` or ```Douban Movie```), please unzip the dataset under the directory ```dataset/``` first;
* If you use your own dataset, you should make sure that the format of your dataset is compatible with the setting (i.e. the following variables) in ```dataloader.py ```
  *  ```sep``` : the separation symbol between columns in dataset (default: '\t')
  *  ```header_name```: the header name of data frame generated from dataset (default: ['u', 'i', 'r', 't'])
  *  ```pos_type```: The type of positive interactions (default: [4.0, 5.0]).
  
  > Note: You should disable the comment of Line 23-27 in ```dataloader.py``` to generate a new column named 'm' when loading data if you use your own dataset.

##### Step 3: run the model.
* For ```Movielens 1M``` dataset:
  ```python
  python main.py --dataset ml1m  --use_user_si --use_item_si
  ```

* For ```Douban Movie``` dataset:
  ```
  python main.py --dataset douban_movie --num_his_month 9 --num_cur_month 1 --decay_factor 1e-8 --pri_factor 128 --alphas '[0.2,1.0,0,0.2]' --use_item_si --user_threshold 0 --item_threshold 0.6
  ```

  