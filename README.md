# Color as the Impetus: Transforming Few-Shot Learner
![](https://img.shields.io/badge/Research-CoSeLearner-brightgreen)
![](https://img.shields.io/badge/Research-CoSeDistiller-brightred)
![](https://img.shields.io/badge/CoarseGrained-FSL-brightred)
![](https://img.shields.io/badge/FineGrained-FSL-brightred)
![](https://img.shields.io/badge/CrossDomain-FSL-brightred)
![](https://img.shields.io/badge/Image-Recognition-brightred)
![](https://img.shields.io/badge/PyTorch-%3E%3Dv1.10-green)
![](https://img.shields.io/badge/Python-%3E%3Dv3.8-yellowgreen)

This package includes our codes for implementing "Color as the Impetus: Transforming Few-Shot Learner". 
(First Release Date: 2025-05-22)

>Benchmark Link: https://pan.baidu.com/s/1KiIZ0FXkGPnhsq0sXjrsZA Code: cf5w

>Our Paper: https://arxiv.org/pdf/2507.22136

## 1.Introduction

*Humans possess innate meta-learning capabilities, partly attributable to their exceptional color perception. 
In this paper, we pioneer an innovative viewpoint on few-shot learning by simulating human color perception mechanisms. We propose the ColorSense Learner, a bio-inspired meta-learning framework that capitalizes on inter-channel feature extraction and interactive learning. 
By strategically emphasizing distinct color information across different channels, our approach effectively filters irrelevant features while capturing discriminative characteristics. 
Color information represents the most intuitive visual feature, yet conventional meta-learning methods have predominantly neglected this aspect, focusing instead on abstract feature differentiation across categories. 
Our framework bridges the gap via synergistic color-channel interactions, enabling better intra-class commonality extraction and larger inter-class differences.
Furthermore, we introduce a meta-distiller based on knowledge distillation, ColorSense Distiller, which incorporates prior teacher knowledge to augment the student network's meta-learning capacity. 
We've conducted comprehensive coarse/fine-grained and cross-domain experiments on eleven few-shot benchmarks for validation.
Numerous experiments reveal that our methods have extremely strong generalization ability, robustness, and transferability, and effortless handle few-shot classification from the perspective of color perception.*


## 2.Few-shot Benchmarks Preparation   
https://github.com/ChaofeiQI/CoSeLearner/releases/tag/Pickle-and-Unpickle-Dataset
```
12 Benchmarks Materials:
├── CIFAR_FS                     ├── FC100                        ├── mini_imagenet                  ├── tieredimagenet_npz
│   ├── CIFAR_FS_train.pickle    │   ├── FC100_train.pickle       │   ├── mini_imagenet_train.pickle │   ├── train_images.npz,train_labels.pkl
│   ├── CIFAR_FS_test.pickle     │   ├── FC100_test.pickle        │   ├── mini_imagenet_test.pickle  │   ├── test_images.npz,test_labels.pkl
│   ├── CIFAR_FS_val.pickle      │   ├── FC100_val.pickle         │   ├── mini_imagenet_val.pickle   │   ├── val_images.npz,val_labels.pkl
├── aircraft_fs                  ├── meta_iNat                    ├── cub_cropped                    ├── tiered_meta_iNat
│   ├── aircraft_fs_train.pickle │   ├── meta_iNat_train.pickle   │   ├── cub_cropped_train.pickle   │   ├── tiered_meta_iNat_train.pickle
│   ├── aircraft_fs_test.pickle  │   ├── meta_iNat_test.pickle    │   ├── cub-cropped_test.pickle    │   ├── tiered_meta_iNat.pickle
│   ├── aircraft_fs_val.pickle   │   ├── meta_iNat_val.pickle     │   ├── cub-cropped_val.pickle     │   ├── tiered_meta_iNat.pickle
├── places                       ├── Stanford_Car                 ├── CropDisease                    ├── EuroSAT
│   ├── places_test.pickle       │   ├── Stanford_Car_test.pickle │   ├── CropDisease_test.pickle    │   ├── EuroSAT_test.pickle
```


## 3.Meta-training,  Meta-testing, and Knowledge distillation

*Meta-training & -testing*: following commands provide an example to train and eval our CoSeLearner.
```bash
# Usage: python3 main1_meta.py [config config-file] [device index] [mode style] [log-step]
python3 main1_meta.py     --config config/query_15/5way_1shot_mini-imagenet.py  --device $GPU --mode train  --log_step 5
python3 main1_meta.py     --config config/query_15/5way_1shot_mini-imagenet.py  --device $GPU --mode eval
```

*Knowledge distillation*: following commands provide an example to train and eval our CoSeDistiller.
```bash
# Usage: python3 main2_distill.py [config config-file] [device index] [mode style] [log-step]
python3 main2_distill.py  --config config/query_15/5way_1shot_mini-imagenet.py  --gen_stu 5 --device $GPU --mode train --log_step 5
python3 main2_distill.py  --config config/query_15/5way_1shot_mini-imagenet.py  --gen_stu 5 --device $GPU --mode eval
```

*Cross-Domain Meta-testing*: following command provides an example to infer novel subset with pretrained model.
```bash
# Usage: python3 main3_meta_cross_domain.py [config config-file] [device index] [mode style] [log-step]
python3 main3_meta_cross_domain.py  --config config/query_15_cross_domain/5way_1shot_places.py  --device $GPU --mode eval
```
```bash
# Usage: python3 main4_distill_cross_domain.py [config config-file] [device index] [mode style] [log-step]
python3 main4_distill_cross_domain.py  --config config/query_15_cross_domain/5way_1shot_places.py  --device $GPU --mode eval
```

## 4.Results of CoSeLearner & CoSeDistiller
```bash
-- Link: https://pan.baidu.com/s/1dYuQs_17UXjUGxH6xEDQXw    --Code: hf8m
```

## License
- This repository is released under the MIT License. License can be found in [LICENSE](LICENSE) file.
