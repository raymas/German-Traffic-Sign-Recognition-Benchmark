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
python main.py --model DKS --train
```

For help:

```bash
python main.py -h
```

## Results

### DeepKnowledgeSeville

<table>
    <tr>
        <th>Loss</th>
        <th>Accuracy</th>
    </tr>
    <tr>
        <td><img src="https://raw.githubusercontent.com/raymas/German-Traffic-Sign-Recognition-Benchmark/master/example-results/DKS/epoch_loss.svg?sanitize=true" width=300></td>
        <td><img src="https://raw.githubusercontent.com/raymas/German-Traffic-Sign-Recognition-Benchmark/master/example-results/DKS/epoch_acc.svg?sanitize=true" width=300></td>
    </tr>
</table>

## Contributing

Fork, publish a new branch, add your name to [CONTRIBUTORS.md](CONTRIBUTORS.md)

## Sources

[GTSRB](http://benchmark.ini.rub.de/) J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011.

[DeepKnowledge Seville]() CNN with 3 Spatial Transformers, DeepKnowledge Seville, Álvaro Arcos-García and Juan A. Álvarez-García and Luis M. Soria-Morillo, [Neural Networks](https://doi.org/10.1016/j.neunet.2018.01.005)

[Sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) Multi-Scale CNNs, sermanet , Traffic sign recognition with multi-scale Convolutional Networks, Traffic sign recognition with multi-scale Convolutional Networks, P. Sermanet, Y. LeCun, August 2011, [International Joint Conference on Neural Networks (IJCNN) 2011](http://dx.doi.org/10.1109/IJCNN.2011.6033589)
