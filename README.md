# Combining Global and Local Attention with Positional Encoding for Video Summarization

## PyTorch Implementation of PGL-SUM [[Paper](https://www.iti.gr/~bmezaris/publications/ism2021a_preprint.pdf)] [[DOI](https://doi.org/10.1109/ISM52913.2021.00045)] [[Cite](https://github.com/e-apostolidis/PGL-SUM#citation)]
<div align="justify">

- From **"Combining Global and Local Attention with Positional Encoding for Video Summarization"**, Proc. of the IEEE Int. Symposium on Multimedia (ISM), Dec. 2021.
- Written by Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris and Ioannis Patras.
- This software can be used for training a deep learning architecture which estimates frames' importance after modeling their dependencies with the help of global and local multi-head attention mechanisms that integrate a positional encoding component. Training is performed in a supervised manner based on ground-truth data (human-generated video summaries). After being trained on a collection of videos, the PGL-SUM model is capable of producing representative summaries for unseen videos, according to a user-specified time-budget about the summary duration. </div>

## Main dependencies
Developed, checked and verified on an `Ubuntu 20.04.3` PC with a `NVIDIA TITAN Xp` GPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cuDNN Version` | `TensorBoard` | `TensorFlow` | `NumPy` | `H5py`
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
3.8(.8) | 1.7.1 | 11.0 | 8005 | 2.4.1 | 2.3.0 | 1.20.2 | 2.10.0

## Data
<div align="justify">

Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the [data](data) folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:
<pre>
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth importance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of subsampled frames in original video
    /n_steps                  number of subsampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
</pre>
Original videos and annotations for each dataset are also available in the dataset providers' webpages: 
- <a href="https://github.com/yalesong/tvsum"><img src="https://img.shields.io/badge/Dataset-TVSum-green"/></a> <a href="https://gyglim.github.io/me/vsum/index.html#benchmark"><img src="https://img.shields.io/badge/Dataset-SumMe-blue"/></a>
</div>

## Training
<div align="justify">

To train the model using one of the aforementioned datasets and for a number of randomly created splits of the dataset (where in each split 80% of the data is used for training and 20% for testing) use the corresponding JSON file that is included in the [data/splits](/data/splits) directory. This file contains the 5 randomly-generated splits that were utilized in our experiments.

For training the model using a single split, run:
```bash
python model/main.py --split_index N --n_epochs E --batch_size B --video_type 'dataset_name'
```
where, `N` refers to the index of the used data split, `E` refers to the number of training epochs, `B` refers to the batch size, and `dataset_name` refers to the name of the used dataset.

Alternatively, to train the model for all 5 splits, use the [`run_summe_splits.sh`](model/run_summe_splits.sh) and/or [`run_tvsum_splits.sh`](model/run_tvsum_splits.sh) script and do the following:
```shell-script
chmod +x model/run_summe_splits.sh    # Makes the script executable.
chmod +x model/run_tvsum_splits.sh    # Makes the script executable.
./model/run_summe_splits.sh           # Runs the script. 
./model/run_tvsum_splits.sh           # Runs the script.  
```
Please note that after each training epoch the algorithm performs an evaluation step, using the trained model to compute the importance scores for the frames of each video of the test set. These scores are then used by the provided [evaluation](evaluation) scripts to assess the overall performance of the model (in F-Score).

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a command line (cmd) and running: `tensorboard --logdir=/path/to/log-directory --host=localhost`
- opening a browser and pasting the returned URL from cmd. </div>

## Configurations
<div align="justify">

Setup for the training process:
 - In [`data_loader.py`](model/data_loader.py), specify the path to the h5 file of the used dataset, and the path to the JSON file containing data about the utilized data splits.
 - In [`configs.py`](model/configs.py), define the directory where the analysis results will be saved to. </div>
   
Arguments in [`configs.py`](model/configs.py): 
|Parameter name | Description | Default Value | Options
| :--- | :--- | :---: | :---:
`--mode` | Mode for the configuration. | 'train' | 'train', 'test'
`--verbose` | Print or not training messages. | 'false' | 'true', 'false'
`--video_type` | Used dataset for training the model. | 'SumMe' | 'SumMe', 'TVSum'
`--input_size` | Size of the input feature vectors. | 1024 | int > 0
`--seed` | Chosen number for generating reproducible random numbers. | 12345 | None, int
`--fusion` | Type of the used approach for feature fusion. | 'add' | None, 'add', 'mult', 'avg', 'max' 
`--n_segments` | Number of video segments; equal to the number of local attention mechanisms. | 4 | None, int ≥ 2
`--pos_enc` | Type of the applied positional encoding. | 'absolute' | None, 'absolute', 'relative'
`--heads` | Number of heads of the global attention mechanism. | 8 | int > 0
`--n_epochs` | Number of training epochs. | 200 | int > 0
`--batch_size` | Size of the training batch, 20 for 'SumMe' and 40 for 'TVSum'. | 20 | 0 < int ≤ len(Dataset)
`--clip` | Gradient norm clipping parameter. | 5 | float 
`--lr` | Value of the adopted learning rate. | 5e-5 | float
`--l2_req` | Value of the regularization factor. | 1e-5 | float
`--split_index` | Index of the utilized data split. | 0 | 0 ≤ int ≤ 4
`--init_type` | Weight initialization method. | 'xavier' | None, 'xavier', 'normal', 'kaiming', 'orthogonal'
`--init_gain` | Scaling factor for the initialization methods. | None | None, float

## Model Selection and Evaluation 
<div align="justify">

The utilized model selection criterion relies on the post-processing of the calculated losses over the training epochs and enables the selection of a well-trained model by indicating the training epoch. To evaluate the trained models of the architecture and automatically select a well-trained model, define the [`dataset_path`](evaluation/compute_fscores.py#L25) in [`compute_fscores.py`](evaluation/compute_fscores.py) and run [`evaluate_exp.sh`](evaluation/evaluate_exp.sh). To run this file, specify:
 - [`base_path/exp$exp_num`](evaluation/evaluate_exp.sh#L6-L7): the path to the folder where the analysis results are stored,
 - [`$dataset`](evaluation/evaluate_exp.sh#L8): the dataset being used, and
 - [`$eval_method`](evaluation/evaluate_exp.sh#L9): the used approach for computing the overall F-Score after comparing the generated summary with all the available user summaries (i.e., 'max' for SumMe and 'avg' for TVSum).
```bash
sh evaluation/evaluate_exp.sh $exp_num $dataset $eval_method
```
For further details about the adopted structure of directories in our implementation, please check line [#6](evaluation/evaluate_exp.sh#L6) and line [#11](evaluation/evaluate_exp.sh#L11) of [`evaluate_exp.sh`](evaluation/evaluate_exp.sh). </div>

## Trained models and Inference
<div align="justify">

We have released the [**trained models**](https://doi.org/10.5281/zenodo.5635735) for our main experiments -namely `Table III` and `Table IV`- of our ISM 2021 paper. The [`inference.py`](inference/inference.py) script, lets you evaluate the -reported- trained models, for our 5 randomly-created data splits. Firstly, download the trained models, with the following script:
``` bash
sudo apt-get install unzip wget
wget "https://zenodo.org/record/5635735/files/pretrained_models.zip?download=1" -O pretrained_models.zip
unzip pretrained_models.zip -d inference
rm -f pretrained_models.zip
```
Then, specify the PATHs for the [`model`](inference/inference.py#L57), the [`split_file`](inference/inference.py#L61) and the [`dataset`](inference/inference.py#L67) in use. Finally, run the script with the following syntax
```shell-script
python inference/inference.py --table ID --dataset 'dataset_name'
```
where, `ID` refers to the id of the reported table, and `dataset_name` refers to the name of the used dataset.
</div>

### Additional evaluation results
<div align="justify">

Given the above pre-trained models, we present some additional evaluation results following the rank-based evaluation protocol proposed [here](https://openaccess.thecvf.com/content_CVPR_2019/papers/Otani_Rethinking_the_Evaluation_of_Video_Summaries_CVPR_2019_paper), and the diversity measure described [here](https://ieeexplore.ieee.org/document/9275314). Moreover, we provide results with regards to the trainable parameters and the required time for training. The code for implementing the rank-based evaluation protocol is available at [CA-SUM](https://github.com/e-apostolidis/CA-SUM/blob/main/inference/evaluation_metrics.py#L40), and the summary diversity was measured using the relevant code from [DSNet](https://github.com/li-plus/DSNet/blob/1804176e2e8b57846beb063667448982273fca89/src/helpers/vsumm_helper.py#L122).
</div>

|Dataset | Spearman's ρ | Kendall's τ | Summary Diversity | Params (M) | Train time<br>(sec / epoch)
| :---: | :---: | :---: | :---: | :---: | :---:
SumMe | - | - | 0.631 | 9.44 | 0.63
TVSum | 0.206 | 0.157 | 0.488 | 9.44 | 1.17


## Citation
<div align="justify">
    
If you find our work, code or pretrained models, useful in your work, please cite the following publication:

E. Apostolidis, G. Balaouras, V. Mezaris, I. Patras, "<b>Combining Global and Local Attention with Positional Encoding for Video Summarization</b>", Proc. IEEE Int. Symposium on Multimedia (ISM), Dec. 2021.
</div>

BibTeX:

```
@INPROCEEDINGS{9666088,
    author    = {Apostolidis, Evlampios and Balaouras, Georgios and Mezaris, Vasileios and Patras, Ioannis},
    title     = {Combining Global and Local Attention with Positional Encoding for Video Summarization},
    booktitle = {2021 IEEE International Symposium on Multimedia (ISM)},
    month     = {December},
    year      = {2021},
    pages     = {226-234}
}
```

## License
<div align="justify">
    
Copyright (c) 2021, Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreement H2020-832921 MIRROR, and by EPSRC under grant No. EP/R026424/1. </div>

## Re-implementation in ModelScope
A re-implementation of PGL-SUM also appears in [ModelScope](https://www.modelscope.cn/models/damo/cv_googlenet_pgl-video-summarization/summary). Please note that we have not tested this re-implementation and therefore cannot confirm if it fully reproduces our original implementation.
