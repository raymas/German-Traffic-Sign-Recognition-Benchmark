# German Traffic Sign Recognition Benchmark (GTSRB)

This repository aims to implement networks models from the GTSRB challenge.

## Getting started

Clone this repository:

```bash
git clone https://github.com/raymas/German-Traffic-Sign-Recognition-Benchmark.git
```

Prior training networks, there are prerequisites.

### Docker installation

Simply build using the provided [Dockerfile](Dockerfile):

```bash
docker build -t gtsrb-nn .
docker run -it gtsrb-nn
```

### Conda environnement

Create a new virtual environnement using the lastest tensorflow packages (GPU or not) from anaconda.

```bash
conda create -c anaconda tensorflow[-gpu] pip
```

PS: please remove the '[]' if you want to have to gpu acceleration or simply delete '[-gpu]' for cpu only tensorflow.

Install the required softwares:

```bash
pip install -r requirements.txt
```

## How to use

Train and test by launching one of the provided model:

```bash
python DeepKnowledgeSeville.py
```

## Contributing

Fork, publish a new branch, add your name to [CONTRIBUTORS.md](CONTRIBUTORS.md)
