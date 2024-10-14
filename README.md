# Chess AI with Monte Carlo Tree Search and Neural Networks

This project implements a Chess AI using Monte Carlo Tree Search (MCTS) and Deep Neural Networks. The AI learns to play chess through self-play and improves over time by training on the generated game data.

## Table of Contents

- [Chess AI with Monte Carlo Tree Search and Neural Networks](#chess-ai-with-monte-carlo-tree-search-and-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Install Dependencies](#install-dependencies)
  - [Usage](#usage)
    - [Training the Neural Network](#training-the-neural-network)
    - [Running MCTS Self-Play](#running-mcts-self-play)
    - [Evaluating the AI](#evaluating-the-ai)
    - [Running the Full Pipeline](#running-the-full-pipeline)
    - [Visualizing a Game](#visualizing-a-game)
    - [Analyzing Games](#analyzing-games)
  - [Modules Description](#modules-description)

## Features

- **Monte Carlo Tree Search (MCTS)**: Efficient search algorithm for making optimal decisions.
- **Deep Neural Networks**: Utilizes PyTorch for neural network implementation and training.
- **Self-Play Learning**: AI improves by playing against itself and learning from the outcomes.
- **Multiprocessing Support**: Accelerated training and evaluation using multiprocessing.
- **Data Visualization**: Tools to visualize the chessboard and analyze games.

## Project Structure
```
├── LICENSE
├── README.md
├── requirements.txt
└── alpha-zero
    ├── __init__.py
    ├── analysis
    │   ├── __init__.py
    │   └── analyze.py
    ├── board
    │   ├── __init__.py
    │   ├── board.py
    │   ├── encoder_decoder.py
    │   └── visualize.py
    ├── evaluator
    │   ├── __init__.py
    │   └── evaluator.py
    ├── mcts
    │   ├── __init__.py
    │   └── mcts.py
    ├── neural_net
    │   ├── __init__.py
    │   ├── neural_net.py
    │   ├── train.py
    │   └── train_multiprocessing.py
    └── pipeline
        ├── __init__.py
        └── pipeline.py
```

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch: ensure compatibility with your CUDA version if using GPU

### Clone the Repository

```bash
git clone https://github.com/yourusername/chess-ai.git
cd chess-ai
```

### Install Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage
### Training the Neural Network
To train the neural network using self-play data:

```bash
python alpha-zero/neural_net/train.py
```

For multiprocessing training to utilize multiple CPU cores:

```bash
python alpha-zero/neural_net/train_multiprocessing.py
```

### Running MCTS Self-Play
To generate self-play games using MCTS:

```bash
python alpha-zero/mcts/mcts.py
```

### Evaluating the AI
To evaluate the performance of the AI:

```bash
python alpha-zero/evaluator/evaluator.py
```

### Running the Full Pipeline
To run the entire training and evaluation pipeline:

```bash
python alpha-zero/pipeline/pipeline.py
```

### Visualizing a Game
To visualize a chess game or board state:

```bash
python alpha-zero/board/visualize.py
```

### Analyzing Games
To analyze games played by the AI:

```bash
python alpha-zero/analysis/analyze.py
```

## Modules Description
`mcts`: Implements the Monte Carlo Tree Search algorithm for decision-making during self-play.\
`neural_net`: Contains the neural network model definition and training scripts.\
`board`: Manages the chessboard representation, move generation, encoding/decoding, and visualization tools.\
`evaluator`: Provides functionality to evaluate the AI's performance against predefined benchmarks.\
`pipeline`: Integrates various components to streamline the training and evaluation process.\
`analysis`: Includes scripts for analyzing game data and AI performance metrics.
