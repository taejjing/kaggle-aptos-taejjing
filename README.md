# 실험
## randomstate 42

## processing_v1
- img_size 256
- efn b4
- Median subtraction
- Only APTOS 2019 
- 5fold
- coef = [0.57, 1.37, 2.57, 3.57] # 1.37...
- steplr

(mse loss / kappa score)

|                        | fold1         | fold2         | fold3         | fold4         | fold5         | lb    |
|------------------------|---------------|---------------|---------------|---------------|---------------|-------|
| processing_v1 (tta 5)  | 0.2584/0.9074 | 0.2614/0.9127 | 0.2552/0.9110 | 0.2716/0.9032 | 0.2831/0.8943 | 0.762 |
| processing_v1 (no tta) |               |               |               |               |               | 0.781 |
|                        |               |               |               |               |               |       |

## processing_v2
- img_size 256
- efn b4
- Median subtraction
- Only APTOS 2019 
- 5fold, 20ep, lr:1e-4
- coef = [0.57, 1.57, 2.57, 3.57]
- v2 transform
- steplr(5, 0.1)

(mse loss / kappa score)  

|                        | fold1         | fold2         | fold3         | fold4         | fold5         | lb    |
|------------------------|---------------|---------------|---------------|---------------|---------------|-------|
| processing_v2 (no tta) | 0.2208/0.9241 | 0.2279/0.9230 | 0.2095/0.0185 | 0.2327/0.9170 | 0.2338/0.9175 | 0.768 |
|                        |               |               |               |               |               |       |

## processing_v3
- img_size 256
- efn b4
- Median subtraction
- 2019 + prev(sample)
- 5fold, 30ep, lr:1e-3
- coef = [0.57, 1.57, 2.57, 3.57]
- v2 transform
- steplr(3, 0.2)
- dropout(0.3)

(mse loss / kappa score)  

|                        | fold1         | fold2         | fold3         | fold4         | fold5         | lb    |
|------------------------|---------------|---------------|---------------|---------------|---------------|-------|
| processing_v3 (no tta) | 0.4826/0.8523 | 0.3268/0.9049 | 0.2545/0.9328 | 0.2363/0.9368 | 0.2513/0.9339 | 0.798 |
|                        |               |               |               |               |               |       |

## processing_v4
- img_size 256
- efn b4, 5fold, 20ep, lr 1e-3
- 2019 + prev(sample) add_prev_v1
- coef = [0.5, 1.5, 2.5, 3.5]
- steplr(3, 0.2)
- dropout(0.2)


## site
https://www.kaggle.com/jeru666/aptos-preprocessing-update-histogram-matching


## TODO
- add_prev_v1 : 데이터가 늘어났긴 하지만 acc, kappa가 생갹보다 많이 늘어남.
- add_prev_v2 : v1에 의해 v2도 실험해볼 필요가 있음.
- circle_v3
- coef weight
- regression -> classification
