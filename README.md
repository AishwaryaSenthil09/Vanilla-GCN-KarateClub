# Vanilla GCN on Karate Club Dataset

## Project Overview
This project implements a two-layer Vanilla Graph Convolutional Network (GCN) for node classification using the Zachary Karate Club dataset.

## Dataset
The Karate Club dataset represents the social relationships between 34 members of a university karate club.

- Nodes: 34
- Undirected edges: 78
- Node features: 34 per node
- Classes: 4
- Training nodes: 4

## Model Architecture
- Input layer: 34 features
- First graph convolution: 16 hidden features
- ReLU activation
- Dropout regularization
- Second graph convolution: 4 output classes
- Optimizer: Adam
- Learning rate: 0.01
- Training epochs: 200
- Loss function: Cross-Entropy Loss

## Results
- Training accuracy: 100.00%
- Overall dataset accuracy: 82.35%
- Final training loss: 0.0158
- Hidden embedding shape: 34 × 16

Note: Overall dataset accuracy is calculated over all 34 nodes, including the training nodes. It is not an independent test-set accuracy.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python vanilla_gcn.py
```

## Outputs
The outputs folder contains the Karate Club network visualization, training loss curve, t-SNE embedding plot, and numerical results.

## Conclusion
The implementation demonstrates graph-based feature aggregation, node embedding generation, and node classification using a Vanilla Graph Convolutional Network.
