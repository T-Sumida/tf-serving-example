# tf-serving-example
A sample repository of tf-serving.


# Overview
This repository is an example of Tf-Serving.

It contains a program that tries a series of steps to deploy a model trained on fashion MNIST to TF-Serving.

# Environment
- Window10 Home 64bit
- WSL2 on Docker
- Anaconda

```
$conda create -n {env_name} python=3.7
$activate {env_name}
$pip install -r requirements.txt
```


# Usage

## Training
1. Modify settings.yml
   - Edit export_dir and version_number
2. Start Train
   ```
   $python train.py settings.yml
   ```


# Tf-Serving
1. Modify docker_start.sh
  ```
  MODEL_NAME="tf-serving" # = settings.yml's export_dir
  ```
2. Start Tf-Serving
```
$bash docker_start.sh
```

# Test
```
$python test_post.py --model_name {settings.yml's export_dir} --version {version_num}
```

# License
Copyright © 2020 T_Sumida Distributed under the MIT License.