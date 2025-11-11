# CAMP-VQA
![visitors](https://visitor-badge.laobi.icu/badge?page_id=xinyiW915/CAMP-VQA) 
![GitHub Repo stars](https://img.shields.io/github/stars/xinyiW915/CAMP-VQA?logo=github)
![Python](https://img.shields.io/badge/Python-3.8+-blue)

Official Code for the following paper:

**X. Wang, A. Katsenou, J.Shen and D. Bull**. [CAMP-VQA: Caption-Embedded Multimodal Perception for No-Reference Quality Assessment of Compressed Video](https://arxiv.org/abs/2511.07290)

[Our paper](https://arxiv.org/abs/2511.07290) was accepted by the IEEE/CVF Winter Conference on Applications of Computer Vision 2026. ([WACV 2026](https://wacv.thecvf.com/)).


Try our online demo on Hugging Face 🤗: [https://huggingface.co/spaces/xinyiW915/CAMP-VQA](https://huggingface.co/spaces/xinyiW915/CAMP-VQA)

---
## Performance
We evaluated the proposed model, CAMP-VQA, on the seven main-stream UGC benchmark datasets. The experimental testing included:
1. Training and testing were performed on each target dataset, referred to as intra-dataset experiments. 
2. Pre-training the model on LSVQ, followed by fine-tuning on the target datasets (denoted as w/ fine-tune), aimed at assessing the model’s transferability and adaptation capabilities.

### **Performance comparison of CAMP-VQA:** 
Spearman’s Rank Correlation Coefficient (SRCC)

| Model                   | Extra Training Data | CVD2014 | KoNViD-1k | LIVE-VQC | YouTube-UGC | LSVQ_test | LSVQ_1080p | FineVD | LIVE-YT-Gaming | KVQ       |
|-------------------------|-------------|--------|-----------|----------|-------------|-----------|------------|----------------|----------------|-----------|
| CAMP-VQA                | None | 0.933  | 0.927     | 0.922    | 0.901       | 0.920     | 0.908      | 0.919 | 0.903          | 0.956     |
| **CAMP-VQA (w/ fine-tune)** | LSVQ | **0.966**  | **0.930**     | **0.934**    | **0.912**       | **0.920**     | **0.908**      | **0.924** | **0.905**          | **0.967** |

Pearson’s Linear Correlation Coefficient (PLCC)

| Model                   | Extra Training Data | CVD2014 | KoNViD-1k | LIVE-VQC | YouTube-UGC | LSVQ_test | LSVQ_1080p | FineVD    | LIVE-YT-Gaming | KVQ       |
|-------------------------|-------------|-------|-------|----------|-------------|-----------|------------|-----------|-----|-----------|
| CAMP-VQA                | None | 0.944 | 0.936 | 0.940    | 0.920       | 0.933     | 0.920      | 0.923     | 0.922          | 0.958     |
| **CAMP-VQA (w/ fine-tune)** | LSVQ | **0.964** | **0.944** | **0.946**    | **0.928**       | **0.933**     | **0.920**      | **0.933** | **0.942**          | **0.967** |

### **Cross-dataset evaluation when trained on LSVQ**
| Model    | Correlation Metrics | CVD2014 | KoNViD-1k | LIVE-VQC | YouTube-UGC | FineVD | LIVE-YT-Gaming | KVQ   |
|----------|---------------------|---------|-------|----------|------------|--------|-------|-------|
| CAMP-VQA | SRCC                | 0.907   | 0.926 | 0.919    | 0.880      | 0.865  | 0.864          | 0.811 |
| CAMP-VQA | PLCC                | 0.933   | 0.932 | 0.937    | 0.898      | 0.890  | 0.884          | 0.810 |

More reported results can be found in **[correlation_result.ipynb](https://github.com/xinyiW915/CAMP-VQA/blob/main/src/correlation_result.ipynb)**.

## Proposed Model
The goal of the proposed framework is to evaluate visual quality without reliance on the uncompressed version of a video. This framework, as outlined in Fig, comprises three components: SVE, TME and SEE.

<img src=./CAMP-VQA.png alt="proposed_CAMP-VQA_framework" width="800"/>

## Usage
### 📌 Install Requirement
The repository is built with **Python 3.10** and can be installed via the following commands:

```shell
git clone https://github.com/xinyiW915/CAMP-VQA.git
cd CAMP-VQA
conda create -n campvqa python=3.10 -y
conda activate campvqa
pip install -r requirements.txt  
```

### 📥 Download UGC Datasets

The corresponding UGC video datasets can be downloaded from the following sources:  
[CVD2014](https://qualinet.github.io/databases/video/cvd2014_video_database/), [KoNViD-1k](https://database.mmsp-kn.de/konvid-1k-database.html), [LIVE-VQC](https://live.ece.utexas.edu/research/LIVEVQC/), [YouTube-UGC](https://media.withyoutube.com/), [LSVQ](https://github.com/baidut/PatchVQ), [FineVD](https://huggingface.co/datasets/IntMeGroup/FineVD), [LIVE-YT-Gaming](https://live.ece.utexas.edu/research/LIVE-YT-Gaming/index.html/), [KVQ](https://lixinustc.github.io/projects/KVQ/) 

The metadata for the NR-VQA UGC dataset is available under [`./metadata`](./metadata).  

Once downloaded, place the datasets in any other storage location of your choice. Ensure that the `videos_dir` in the [`load_dataset`](./src/main_camp-vqa.py) function inside `main_camp-vqa.py` is updated accordingly.


### 🎬 Test Demo  
Run the pre-trained models to evaluate the quality of a single video.  

The model weights provided in [`./model/fine_tune`](./model/fine_tune) for CAMP-VQA (w/ fine-tune) and [`./model/best_model`](./model/best_model/) for CAMP-VQA, contain the best-performing saved weights from training.
To evaluate the quality of a specific video, run the following command:
```shell
python camp-vqa_demo.py 
    -device <DEVICE> 
    -intra_cross_experiment <intra/cross>
    -is_finetune <True/False> 
    -save_model_path <MODEL_PATH> 
    -train_data_name <TRAIN_DATA_NAME> 
    -test_data_name <TEST_DATA_NAME>
    -test_video_path <DEMO_TEST_VIDEO>
```
Or simply try the default demo video by running:
```shell
python camp-vqa_demo.py 
```

## Training  
Steps to train CAMP-VQA from scratch on different datasets.  

See detailed prompt settings in [prompts.json](./src/config/prompts.json).

### Extract Features  
Run the following command to extract features from videos:   
```shell
python main_camp-vqa.py -database konvid_1k -num_workers 4 -feature_save_path ../features/
```

### Train Prediction Model
Train our model using extracted features:
```shell
python model_regression.py -data_name konvid_1k -feature_path ../features/camp-vqa/ -save_path ../model/
```

For **LSVQ**, train the model using:  
```shell
python model_regression_lsvq.py -data_name lsvq_train -feature_path ../features/camp-vqa/ -save_path ../model/
```

### Fine-Tuning on Trained Model
To fine-tune a pre-trained model on a new dataset:
1. Turn on the `-is_finetune` flag.  
2. Set [`-train_data_name`](./src/model_finetune.py) to the dataset used for training.  
3. Set [`-test_data_name`](./src/model_finetune.py) to the dataset you want to fine-tune on.
4. Make sure [`-feature_path`](./src/model_finetune.py) points correctly to your save path.

```shell
python model_finetune.py -train_data_name lsvq_train -test_data_name kvq -is_finetune
```

### Cross-dataset Evaluation on Trained Models
Results where models are trained on one dataset and tested on other datasets.
```shell
python model_finetune.py -train_data_name lsvq_train -test_data_name kvq
```

## Ablation Study
We explored the impact of key component on CAMP-VQA performance: semantic embeddings.

Ablation study on the effect of different component semantic embeddings on **KoNViD-1k** and **FineVD** datasets— including **image**, **quality**, **artifact**, and **content** embeddings (*ē<sub>img</sub>*, *ē<sub>qlt</sub>*, *ē<sub>art</sub>*, *content<sub>embs</sub>*).

| ē<sub>img</sub> | ē<sub>qlt</sub> | ē<sub>art</sub> | content<sub>embs</sub> | **KoNViD-1k**  (SRCC) | **KoNViD-1k** (PLCC) | **FineVD**  (SRCC) | **FineVD** (PLCC) |
|--------|--------------|-----------------|----------------|-----------------------|----------------------|--------------------|-------------------|
| ✓      |              |                 |                | 0.778                 | 0.804                | 0.804              | 0.817             |
|        | ✓            |                 |                | 0.631                 | 0.792                | 0.816              | 0.869             |
|        |              | ✓               |                | 0.735                 | 0.763                | 0.812              | 0.840             |
|        |              |                 | ✓              | 0.409                 | 0.451                | 0.401              | 0.409             |
| ✓      | ✓            |                 |                | 0.830                 | 0.871                | 0.899              | 0.911             |
| ✓      | ✓            | ✓               |                | **0.903**             | **0.922**            | **0.901**          | **0.919**         |
| ✓      | ✓            | ✓               | ✓              | 0.892                 | 0.919                | 0.896              | 0.917             |

### On Semantic Embeddings (e.g., image, quality, artifact, content):
  ```shell
  python main_semantic_embs_ablation.py -database konvid_1k -feat_name semantic_embs -feature_save_path ../features/semantic_embs/
  ```

## Acknowledgment
This work was funded by the UKRI MyWorld Strength in Places Programme (SIPF00006/1) as part of my PhD study.

## Citation
If you find this paper and the repo useful, please cite our paper 😊:

```bibtex
@article{wang2025camp,
      title={CAMP-VQA: Caption-Embedded Multimodal Perception for No-Reference Quality Assessment of Compressed Video},
      author={Wang, Xinyi and Katsenou, Angeliki, Shen, Junxiao and Bull, David},
      booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV2026)}, 
      year={2025},
}
@article{wang2025diva,
      title={DIVA-VQA: Detecting Inter-Frame Variations in UGC Video Quality},
      author={Wang, Xinyi and Katsenou, Angeliki and Bull, David},
      booktitle={IEEE International Conference on Image Processing (ICIP 2025)}, 
      year={2025},
}
@article{wang2024relax,
      title={ReLaX-VQA: Residual Fragment and Layer Stack Extraction for Enhancing Video Quality Assessment},
      author={Wang, Xinyi and Katsenou, Angeliki and Bull, David},
      year={2024},
      eprint={2407.11496},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.11496}, 
}
```

## Contact:
Xinyi WANG, ```xinyi.wang@bristol.ac.uk```
