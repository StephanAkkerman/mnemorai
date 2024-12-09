# FluentAI: Learn languages in a flash

![FluentAI Banner](img/banner-withbg.jpg)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Supported versions">
  <img src="https://img.shields.io/github/license/StephanAkkerman/FluentAI.svg?color=brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

---

> [!CAUTION]
> This project is currently under development, please see the issues to see everything that still needs to be done before this is ready to use.

## Introduction

FluentAI is inspired by the method detailed in the paper [SmartPhone: Exploring Keyword Mnemonic with Auto-generated Verbal and Visual Cues by Jaewook Lee and Andrew Lan](https://arxiv.org/pdf/2305.10436.pdf). The aim is to recreate their approach using accessible, open-source models.
The pipeline they propose, as shown below, serves as the blueprint for our project. It illustrates the process of automating language learning, blending cutting-edge AI techniques with proven language learning methodology.
For the architectural overview view our [Figma board](https://www.figma.com/board/zkIYtrCM3ri0ER62p4WiEt/Architectural-Overview?node-id=0-1&t=6vREjL5A8JitOAeG-1)

You can find the list of supported languages [here](supported-languages.md).

![image](https://github.com/StephanAkkerman/FluentAI/assets/45365128/c9ca3190-b136-453d-91cd-f785eac11fa3)

## Table of Contents 🗂

- [Mnemonic Word Generation](#mnemonic-word-generation-)
  - [Imageability](#imageability)
  - [Phonetic Similarity](#phonetic-similarity)
  - [Orthographic Similarity](#orthographic-similarity)
  - [Semantic Similarity](#semantic-similarity)
  - [Best Mnemonic Word](#best-mnemonic-word)
- [Prerequisites](#prerequisites-)
- [Installation](#installation-)
    - [Using `pip`](#using-pip)
    - [Building from Source](#building-from-source)
    - [Setting up the Frontend (Optional)](#setting-up-the-frontend-optional)
    - [GPU Support](#gpu-support)
- [Usage](#usage-)
- [Citation](#citation-)
- [Contributing](#contributing-)
- [License](#license-)

## Mnemonic Word Generation 🏭

In the image below you can see a more detailed process of deriving the mnemonic word, which is the core of the project. The mnemonic word is a word that is easy to remember and that is associated with the word you want to learn. This is done by using a pre-trained model to generate a sentence that is then used to generate a mnemonic word. In the image above this is referred to as "TransPhoner", as this is where the image below is derived from.

![image](https://github.com/user-attachments/assets/d6914bb2-308c-4612-ae7d-df04455bfeae)

### Imageability

The imageability of a word is a measure of how easily a word can be visualized. This is important for the mnemonic word, as it should be easy to visualize. To determine the imageability of a word, we train a model on this [dataset](https://huggingface.co/datasets/StephanAkkerman/imageability). It includes the embeddings for each word and their imageability score. The embeddings are generated by the FastText model and these embeddings can be used to predict the imageability of words that are not in the dataset.

### Phonetic Similarity

The phonetic similarity of a word is a measure of how similar the pronunciation of two words is. This is important for the mnemonic word, as it should be easy to remember. Therefore we use this to determine which English words should be considered for the mnemonic word. We use the CLTS and PanPhon models to generate the feature vectors of the IPA representation of the words. These feature vectors are then used to calculate the phonetic similarity between the words. We use [faiss](https://github.com/facebookresearch/faiss) to speed up the search for the most similar words.

### Orthographic Similarity

The orthographic similarity of a word is a measure of how similar the spelling of two words is. This is a very simple process and the user can select a few methods that they'd like to use.

### Semantic Similarity

The semantic similarity of a word is a measure of how similar the meaning of two words is. The FastText model is used to generate the embeddings of the words and these embeddings are used to calculate the semantic similarity between the words.

### Best Mnemonic Word

To determine the best mnemonic word, we use the methods described above. The results of each method are given as a score (between 0 and 1) and these scores are combined to determine the best mnemonic word. The user can select the weights of each method to determine how important each method is.

## Mnemonic Image Generation

TODO

## Prerequisites 📋

Before starting, make sure you have the following requirements:

- [Anki](https://apps.ankiweb.net/) installed on your device.
- [Anki-Connect](https://foosoft.net/projects/anki-connect/) this add-on allows you to add cards to Anki from the command line.
- Add the deck in `/deck/FluentAI.apkg` to your Anki application. You can do this by dragging and dropping the file into the Anki application.
- [Python](https://www.python.org/downloads/) 3.10 installed on your device.
- [React](https://react.dev) installed on your device (optional).

## Installation ⚙️

### Using `pip`

We have bundled all required dependencies into a package for easy installation. To get started, simply run one of the following commands:

```bash
pip install .
```

or install directly from the repository:

```bash
pip install git+https://github.com/StephanAkkerman/FluentAI.git
```

### Building from Source

If you prefer to build from source, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/StephanAkkerman/FluentAI.git
   ```

2. Navigate to the project directory:

   ```bash
   cd FluentAI
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Setting up the Frontend (Optional)

If you plan to use the API, you can set up the frontend by following these steps:

1. Navigate to the `frontend` directory:

   ```bash
   cd fluentai/frontend
   ```

2. Install the necessary frontend dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm start
   ```

This will launch the frontend and connect it to the API for a seamless user experience.

### Install with GPU Support (Recommended)

If you would like to use a GPU to run the code, you can install the `torch` package with the CUDA support.
After installing the required dependencies, run the following command:

```bash
pip install -r requirements/gpu-requirements.txt
```

## Usage ⌨️

TODO

## Citation ✍️

If you use this project in your research, please cite as follows:

```bibtex
@misc{FluentAI,
  author  = {Stephan Akkerman, Winston Lam},
  title   = {FluentAI},
  year    = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/FluentAI}}
}
```

## Contributing 🛠

Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.
If you would like to make code contributions yourself, please read [CONTRIBUTING.MD](CONTRIBUTING.md).\
![https://github.com/StephanAkkerman/FluentAI/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=StephanAkkerman/FluentAI)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
