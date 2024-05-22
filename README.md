# Supplementary Material for Submission3212

## Title: MergeUp Augmented Semi-weakly Supervised Learning for Whole Slide Image Classification
The structure of the supplementary material is as follows:

```
├── normal-MIL # MIL tool package, used to train different MIL models
│   ├── npis.sh # the main shell command to train MILs on CAMELYON16
│   ├── npis_bracs.sh # the main shell command to train MILs on BRACS
│   ├── ...
├── SWS-MIL # the main dir
│   ├── instance_eval # instance classification
│   │   ├── ...
│   │   ├── main.py # the main file 
│   ├── vis # wsi visualization
│   │   ├── ...
│   │   ├── vis.py # the main file 
│   ├── ...
│   ├── test_A100.sh # the main shell command to train SWS-MIL
│   ├── test_A100_mix.sh # the main shell command to train SWS-MIL on different mix
```

