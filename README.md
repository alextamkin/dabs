# DABS: A Domain Agnostic Benchmark for Self-Supervised Learning

This repository contains the code for DABS, a benchmark for domain-agnostic self-supervised learning algorithms. The basic components of the benchmark can be found in [datasets](#datasets), [encoders](#encoders), and [algorithms](#pretraining-algorithms). Training is implemented with the [PyTorch Lightning](https://www.pytorchlightning.ai/) framework, logging with [Weights and Biases](https://wandb.ai/), and configuration management with [Hydra](https://hydra.cc/).

## Usage
We provide support for Python >= 3.7. Install requirements with
```bash
python -m pip install -r requirements.txt
```
For instructions on how to install PyTorch versions compatible with your CUDA versions, see [pytorch.org](https://pytorch.org/).

### Datasets

We provide a set of dataset implementations (in `src/datasets`) from image, text, speech, sensor, medical imaging, and image-text domains. Preprocessing operations on these datasets are minimal and hard-coded as simple resizing (i.e. of images) and truncations (i.e. of text, audio). These should not be changed so as to maintain fair comparisons across other users of the benchmark.

> See `conf/datasets/*.yaml` for all dataset configs, including the loss, metrics, and batch size used for each dataset.

Almost all datasets will download automatically when the dataset class is instantiated. The exceptions are the CheXpert, ImageNet, and CU Birds datasets, where manual registration or download is required. See the respective dataset files for specific instructions.


| Pretraining Dataset (unlabeled) | Transfer Dataset (labeled) |
|:-----------------|:-----------------|
| [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) | [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [CU Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [Traffic Sign](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html), [VGG Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/) |
| [PAMAP2](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring) | [PAMAP2](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring) |
| [MSCOCO](https://cocodataset.org/#home) | [MSCOCO](https://cocodataset.org/#home) (mismatched detection), [VQA](https://visualqa.org/vqa_v1_download.html) (Binary classification)  |
| [Wikitext-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) | [GLUE](https://gluebenchmark.com/) (10 Tasks)  |
| [mC4](https://huggingface.co/datasets/mc4) | [PAWS-X](https://huggingface.co/datasets/paws-x) (7 Tasks)  |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) (atelectasis, cardiomegaly, consolidation, edema, and pleural effusion), [ChestX-ray8](https://nihcc.app.box.com/v/ChestXray-NIHCC) (atelectasis, cardiomegaly, effusion, infiltration, mass, nodule, pneumonia, pneumothorax) |
| [LibriSpeech](https://www.openslr.org/12) | [Audio MNIST](https://arxiv.org/abs/1807.03418v2), [Fluent Speech](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) (Action, Object, Location), [Google Speech Commands](https://arxiv.org/abs/1804.03209), [LibriSpeech](https://www.openslr.org/12), [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) |

### Pretraining
During the pretraining phase, self-supervised encoders are trained to learn good representations from unlabeled data. We currently support seven datasets for pretraining, one for each domain: MS COCO, ImageNet, CheXpert, PAMAP2, mC4, WikiText-103, and LibriSpeech. If the pretraining dataset has associated labels, an online linear evaluator is jointly trained with the encoder to provide a heuristic of transfer performance. 

Run pretraining with commands like
```bash
python pretrain.py exp.name=<experiment-name> dataset=<dataset> algorithm=<algorithm>
```
Each dataset and encoder has its own config file, so to train a Transformer on the CheXpert dataset with the *e*-Mix algorithm, run
```bash
python pretrain.py exp.name=emix-chexpert encoder=transformer dataset=chexpert algorithm=emix
```
> See `conf/pretrain.yaml` for all pretraining configuration fields.

For more information on the datasets, encoders, and algorithms, see the following section.


| Pretraining Dataset      | Modality         | Label type (unused)   | Input Type    |
|:-------------|:-----------------|:----------------------|:-------------:|
| CIFAR10      | Natural images   | Single label          | 2d            |
| PAMAP2       | Sensor           | Single label          | 2d            |
| MSCOCO       | Captioned images | Single label          | 2d +<br/>tokens |
| WikiText-103 | English Text             | No label              | tokens        |
| mC4 | Multilingual Text             | No label              | tokens        |
| CheXpert     | Medical images   | Multi label           | 2d            |
| LibriSpeech  | Speech           | No label              | 2d            |


### Transfer Learning
After pretraining, a small linear classifier is trained on top of the frozen encoder. Run transfer learning from a randomly initialized encoder with
```bash
python transfer.py exp.name=<experiment-name> dataset=<dataset> ckpt=null 
```

> See `conf/transfer.yaml` for all transfer learning configuration fields and optionally replace `null` with the path to your pretrained encoder checkpoint.

| Dataset                | Modality         | Label type   | Evaluation metric    | Input Type    |
|:-----------------------|:-----------------|:-------------|:---------------------|:-------------:|
| Aircraft               | Natural images   | Single label | Accuracy             | 2d            |
| CU Birds               | Natural images   | Single label | Accuracy             | 2d            |
| DTD                    | Natural images   | Single label | Accuracy             | 2d            |
| Traffic Sign           | Natural images   | Single label | Accuracy             | 2d            |
| VGG Flower             | Natural images   | Single label | Accuracy             | 2d            |
| Pamap2                 | Sensor           | Single label | Accuracy             | 2d            |
| MS COCO                | Captioned images | Binary label | Accuracy             | 2d +<br/>tokens |
| VQA                    | Captioned images | Binary label | Accuracy             | 2d +<br/>tokens |
| CheXpert               | Medical images   | Multi label  | AUROC                | 2d            |
| ChestX-ray8               | Medical images   | Multi label  | AUROC                | 2d            |
| PAWS-X                   | Multilingual Text             | Binary label | Accuracy  | tokens        |
| COLA                   | English Text             | Binary label | Pearson correlation  | tokens        |
| MNLI Matched           | English Text             | Single label | Accuracy             | tokens        |
| MNLI Mismatched        | English Text             | Single label | Accuracy             | tokens        |
| MRPC                   | English Text             | Binary label | Accuracy             | tokens        |
| QNLI                   | English Text             | Binary label | Accuracy             | tokens        |
| QQP                    | English Text             | Binary label | Accuracy             | tokens        |
| RTE                    | English Text             | Binary label | Accuracy             | tokens        |
| SST2                   | English Text             | Binary label | Accuracy             | tokens        |
| STSB                   | English Text             | Regression   | Spearman correlation | tokens        |
| WNLI                   | English Text             | Binary label | Accuracy             | tokens        |
| Audio MNIST            | Speech           | Single label | Accuracy             | 2d            |
| Fluent Speech          | Speech           | Single label | Accuracy             | 2d            |
| Google Speech Commands | Speech           | Single label | Accuracy             | 2d            |
| LibriSpeech            | Speech           | Single label | Accuracy             | 2d            |
| VoxCeleb1              | Speech           | Single label | Accuracy             | 2d            |

## Encoders
A domain-agnostic SSL method should have an encoder which remains as constant as possible across domains. We provide a general transformer encoder baseline (in `src/encoders`). The transformer operates on a sequence of vectors that are produced by a small set of embedding modules (e.g. patch or token embeddings).

## Pretraining algorithms
The pretraining algorithm is the framework and objective that the encoder is trained with. Examples of domain-specific algorithms include [SimCLR](https://arxiv.org/abs/2002.05709), [BYOL](https://arxiv.org/abs/2006.07733), and [MoCo](https://arxiv.org/abs/1911.05722), but these are not domain-agnostic methods as they depend on vision-specific augmentations. We provide our own domain-agnostic implementations of recent algorithms, including *e*-mix (a generalization of [*i*-mix](https://arxiv.org/abs/2010.08887)) and Shuffled Embedding Detection (ShED; a generalization of [ELECTRA](https://arxiv.org/abs/2003.10555)), which randomly permutes a subset of the input embeddings and trains the model to identify the permuted embeddings.

## Results
Below are results for algorithms trained on each dataset in DABS. The baseline performance is obtained via a randomly initialized encoder.

| Pretrain Dataset | Transfer Dataset       | Encoder     | Baseline Performance | *e*-mix Performance | ShED Performance |
|:-----------------|:-----------------------|:------------|:--------------------:|:-------------------:|:----------------:|
| ImageNet          | CIFAR10                | Transformer | 24.20%               | 39.43%              | **39.63%**       |
| ImageNet          | CU Birds               | Transformer | 1.62%                | **3.86%**           | 2.95%            |
| ImageNet          | VGG Flowers            | Transformer | 9.03%                | **25.96%**          | 13.03%           |
| ImageNet          | DTD                    | Transformer | 7.39%                | 8.83%               | **18.35%**       |
| ImageNet          | Traffic Sign           | Transformer | 14.33%               | **65.07%**          | 27.51%           |
| ImageNet          | Aircraft               | Transformer | 2.70%                | **10.15%**          | 5.60%            |
| PAMAP2           | PAMAP2                 | Transformer | 69.81%               | 	79.48%              | **88.69%**           |
| MSCOCO           | VQA                    | Transformer | **57.50%**           | 48.90%              | 54.30%           |
| CheXpert         | CheXpert               | Transformer | 68.14%               | **72.40%**          | **72.40%**       |
| CheXpert         | ChestX-ray8               | Transformer | 57.00%               | 63.00%          | **63.70%**       |
| Wikitext-103     | GLUE (average)         | Transformer | 42.29%               | 44.08%              | **48.37%**       |
| mC4     | PAWS-X (average)         | Transformer | 58.11%               | 56.16%              | **59.91%**       |
| LibriSpeech      | Audio MNIST            | Transformer | 33.13%               | **80.35%**          | 67.33%           |
| LibriSpeech      | Fluent Locations       | Transformer | **62.09%**           | 60.93%              | 60.24%           |
| LibriSpeech      | Fluent Actions         | Transformer | 26.15%               | 29.87%              | **30.53%**       |
| LibriSpeech      | Fluent Objects         | Transformer | 30.13%               | **39.89%**          | 39.36%           |
| LibriSpeech      | Google Speech Commands | Transformer | 4.87%                | 19.22%              | **20.73%**       |
| LibriSpeech      | LibriSpeech            | Transformer | 17.12%               | **60.18%**          | 34.77%           |
| LibriSpeech      | VoxCeleb1              | Transformer | 0.59%                | 2.43%               | **2.81%**        |
