# Normal and Abnormal Pathology Knowledge-Augmented Vision-Language Model for Anomaly Detection in Pathology Images

Official code and resources repository for "Normal and Abnormal Pathology Knowledge-Augmented Vision-Language Model for Anomaly Detection in Pathology Images" [\[paper\]](https://arxiv.org/abs/2508.15256) (ICCV 2025) in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

```bibtex
@inproceedings{song2025normal,
  title={Normal and abnormal pathology knowledge-augmented vision-language model for anomaly detection in pathology images},
  author={Song, Jinsol and Wang, Jiamu and Nguyen, Anh Tien and Byeon, Keunho and Ahn, Sangjeong and Lee, Sung Hak and Kwak, Jin Tae},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22066--22076},
  year={2025}
}
```

## Trained Model Weights

The lymph node WSI dataset used for training cannot be publicly released at this time.  
To enable reproducibility, we provide the trained model weights as follows:

- [Original trained weights](https://drive.google.com/file/d/1ie7ch0Pcvdrf46NyPL2lyckzsuzyPoWI/view?usp=drive_link): Used to obtain the experimental results reported in the paper's tables (100 normal words and 50 abnormal words).  
- [Re-trained weights](https://drive.google.com/file/d/141a-w_ungtVT9e5vTszAvRonCQEGURSf/view?usp=drive_link): Re-trained using a cleaned vocabulary with duplicates removed, corresponding to the word counts reported in the paper (92 normal words and 48 abnormal words).

To load a pretrained model without further training, set `"train": 0` and `"update_centroids": 0`, and specify the path to the pretrained weights in `"trained_model_pth"`.

**Note on vocabulary sizes:**  
In the paper, we report 92 normal words and 48 abnormal words. However, the table results were generated using 100 normal words and 50 abnormal words (with duplicates). The cleaned-word model was trained later using a deduplicated vocabulary to match the reported counts. The normal and abnormal pathology terms used in training can be found in the `words` folder.

## Training and Validation

The paths specified by `"train_imgs_path"` and `"val_imgs_path"` should contain **processed patch images**, not raw WSI files. All images in these directories and their subfolders must be **normal images only**. The folder structure can be arbitrary.

Ideally, validation normal images should not overlap with the training data, as this provides a better estimate of model performance (as done in the paper). However, if sufficient new validation images are not available, it is acceptable to use a subset of the training dataset for validation.

For our experiments, we used a quite large training dataset, performing 1 epoch with a batch size of 100 and accumulating 100 batches before each weight update; you may adjust these settings depending on the size of your dataset.

## Test and Inference

### Patch-level Anomaly Scores

To compute patch-level anomaly scores, set `"test": 1` and `"test_patches_WSIs_path"`. The top path should contain subfolders, each named after a WSI's ID (`{WSI_id}`). Inside each WSI folder, include only the patch images corresponding to that WSI. The folder structure should look like:

```bibtex
your_test_WSI_patch_folders_top_path/
├── WSI_001/
│   ├── WSI_001_0_0_50.png
│   ├── WSI_001_512_0_40.png
│   └── ...
├── WSI_002/
│   ├── WSI_002_0_0_10.png
│   ├── WSI_002_1024_512_0.png
│   └── ...
└── ...
```

### WSI-level Anomaly Scores

To compute WSI-level anomaly scores after obtaining patch-level scores, set `"WSI_level": 1` and `"test_raw_WSIs_path"`. The top path should contain the raw WSI files named as `{WSI_id}.svs`, corresponding to the WSI IDs used for patch generation. The folder structure should look like:

```bibtex
your_test_raw_WSI_files_top_path/
├── WSI_001.svs
├── WSI_002.svs
├── WSI_003.svs
└── ...
```

To apply the 3×3 erosion described in the paper, the coordinates of each patch are required. Therefore, patch filenames extracted from each WSI should follow the format:
`{WSI_id}_{w_coord}_{h_coord}_{abnormal_percent(optional)}.png`. If you are only computing patch-level anomaly scores or WSI-level anomaly scores without applying erosion, following this filename convention is not required.
When computing WSI-level anomaly scores with the 3×3 erosion applied, make sure the `"WSI_downsampling_factor"` is set to the same value as the patch size specified in `"test_patch_size"` (both width and height of the patch). This setting matches the configuration used in the paper.
