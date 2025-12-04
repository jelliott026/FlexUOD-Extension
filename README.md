# FlexUOD-Extension

Working to expand the applicability of FlexUOD to video formats.

## Original Work

This repository is essentially a fork of the original FlexUOD repo: <https://github.com/zhliu-uod/FlexUOD>

FlexUOD is an unsupervised outlier detection model proposed by Liu _et al._ in this paper: <https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_FlexUOD_The_Answer_to_Real-world_Unsupervised_Image_Outlier_Detection_CVPR_2025_paper.pdf>

## Environment Setup

We used `conda` to manage our environments. You can create the corresponding environment using the following:

```bash
conda env create -f environment.yml
```

**IMPORTANT:** Our method requires a GPU to run since we used `cupy`, a GPU implementation of `numpy`, to improve performance. `cupy` has different versions depending on your GPU and driver versions. If you encounter any module/package issues, please consult [this guide](https://docs.cupy.dev/en/stable/install.html). You can check your installed version of CUDA using `nvcc --version` if you have it installed. The version used in testing was `cupy-cuda13x`, which corresponds with Nvidia GPUs and CUDA version 13. Specifically, tests were performed using a Nvidia GeForce RTX 3070 Ti.

### Data

The data used can be downloaded here:

- ResNet: <https://drive.google.com/drive/folders/1UsDMbE3cbA4FgcD4lDiaL2fDxiuwcmHA?usp=drive_link>
- CLIP: <https://drive.google.com/drive/folders/1XjchpD1gEnTeVDSOhYkAkNUxz6CI0l-G?usp=sharing>

This is the same data provided by the original work.

**Note:** Only the ResNet features were tested in our extension of FlexUOD.

Put all downloaded data into a folder named `data`. **You may need to rename the extracted ResNet dataset folder to have the correct spelling: `ResNet`.** If extracted correctly, you data directory should looks as follows:

```txt
./data/
    ResNet/
        cifar10/
        cifar100/
        ...
```

**Note:** There were some inconsistencies with what the original work's `dataloader.py` expected and what was provided in the data they linked. We believe we fixed any inconsistencies, but be wary of potential errors, particularly if you do not use the ResNet dataset.

## Running the Code

Our extensions to the original method are located in `new_method.py`. The function for outlier score is `predict_scores` and labeling is `predict_labels`. This differs from the original FlexUOD implementation, defined in `method.py`, which requires instantiating a `DaDTAnomalyDetector` object and calling the `dadt_` method. This method only returns outlier scores, as labeling was not implemented in the original code. `dadt_alt` is a method we defined, where the only modification is that the estimated contamination factor is also returned; otherwise, there are no modifications to the method.

Our experiment code is located within `experiments.ipynb`. The video experiments are located in `fake_video_test.py` and `real_video_test.py` and must be run in a terminal. `fake_video_test.py` is much more reliable to run since it generates its own data, whereas `real_video_test.py` requires a video input and has several aspects tuned to the specific video we used. For custom tests, refer to the docstrings of `predict_scores` and `predict_labels`.
