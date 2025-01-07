# This is the code for paper:  Sequential Ensemble Learning for Next Item Recommendation


## Requirement

- python 3.6
- tensorflow-gpu==1.14.0

## Run code

### STEP 1

从百度网盘下载数据集，并解压到'/datasets/'路径下

链接: https://pan.baidu.com/s/1E5q9zbVYdaXFD_CoYHeOBg?pwd=1234

### STEP 2

运行 'basemodel/main.py' 为每个数据集生成基础模型。

或者，您可以从百度网盘下载 "Kindle" 文件，并解压到 '/datasets/basemodel/' 路径下

链接: https://pan.baidu.com/s/1BxW3bToTWNNStVtuOXinhw?pwd=1234

### STEP 3

运行 'main.py' 进行提出的方法。

## NOTE

main.py 文件描述了所提出方法及其消融实验的超参数设置，具体如下:

| method_name        | tradeoff   |  user_module  | model_module| div_module|
| ---------- | :-----------:  | :-----------: |  :-----------: |  :-----------: |
|SEM:        |tradeoff[data]|'SAtt'     |'dynamic'   |'cov'     |
|w/o uDC:    |tradeoff[data]|'static'   |'dynamic'   |'cov'     |
|w/o bDE:    |tradeoff[data]|'SAtt'     |'static'    |'cov'     |
|w/o Div:    |0.0           |'SAtt'     |'dynamic'   |'cov'     |
|w/o TPDiv:  |tradeoff[data]|'SAtt'     |'dynamic'   |'AEM-cov' |
