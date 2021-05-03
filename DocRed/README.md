## Code cloned from https://github.com/thunlp/DocRED/tree/master/code 
# Baseline code

## Requirements and Installation
python3

pytorch>=1.0

```
pip3 install -r requirements.txt
```

## preprocessing data
Download metadata from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK) for baseline method and put them into prepro_data folder.


```
python3 DocRed/gen_data.py --in_path data --out_path prepro_data
```

## relation extration

training:
```
CUDA_VISIBLE_DEVICES=0 python3 DocRed/train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev
```

testing (--test_prefix dev_dev for dev set, dev_test for test set):
```
CUDA_VISIBLE_DEVICES=0 python3 DoRed/test.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev --input_theta 0.3601
```

## evaluation

dev result can evaluated by 
```
python3 evalutaion result.josn ../data/dev.json
```

test result should be submit to Codalab.



