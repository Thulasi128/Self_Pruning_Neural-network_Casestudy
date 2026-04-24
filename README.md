# Self_Pruning_Neural-network_Casestudy
Self_Pruning_Neural-network_Casestudy
Overview

This project implements a self-pruning neural network using PyTorch.
Instead of manually pruning weights, the model learns which connections to remove during training using trainable gate parameters.

The goal is simple:

Maintain good accuracy
Reduce unnecessary weights
Achieve an efficient and sparse model automatically
Key Idea

Each weight in the network is controlled by a learnable gate:

A gate value is passed through a sigmoid function → range (0,1)
The weight is multiplied by this gate
If the gate is close to 0 → weight is effectively removed

This allows the network to decide which connections matter.

Model Architecture

Feedforward Neural Network:

Input: 32×32×3 (CIFAR-10 images → flattened to 3072)
Layers:
3072 → 512 (PrunableLinear)
512 → 128 (PrunableLinear)
128 → 10 (Output layer)

Activation: ReLU

Pruning Mechanism

Each PrunableLinear layer contains:

weight
bias
gate_scores (trainable)

Forward pass:

gates = sigmoid(gate_scores)
pruned_weights = weight * gates
output = linear(input, pruned_weights)
Loss Function

Total loss:

Loss = CrossEntropyLoss + λ × SparsityLoss

Where:

CrossEntropyLoss → classification performance
SparsityLoss → encourages gates to go toward 0
SparsityLoss = sum(all gate values) / total gates
Lambda (λ)

Controls pruning strength:

λ Value	Effect
0.0	No pruning
0.5	Light pruning
2.0	Balanced
5.0	Aggressive pruning
Dataset
CIFAR-10
Automatically downloaded via torchvision

Classes:

Airplane, Automobile, Bird, Cat, Deer,
Dog, Frog, Horse, Ship, Truck
Training Details
Optimizer: Adam
Learning Rates:
Weights → 0.001
Gate scores → 0.005 (higher to push pruning)
Batch size: 256
Epochs: 10
Normalization: Mean = 0.5, Std = 0.5
Sparsity Calculation

A weight is considered pruned if:

gate < 0.5

Metrics printed:

Min / Max gate value
Mean gate value
% of gates below threshold
Final sparsity %
Results

The model evaluates different λ values and prints:

Lambda | Accuracy | Sparsity

This shows the trade-off:

Higher sparsity → lower accuracy (usually)
Goal → find balance
Visualization

Two plots are generated:

1. Gate Distribution
Histogram of gate values
Log scale
Red line at pruning threshold (0.5)
2. Accuracy vs Sparsity
Shows how λ affects:
Model accuracy
Sparsity %

Saved as:

gate_distribution_and_tradeoff.png
How to Run
1. Install dependencies
pip install torch torchvision matplotlib numpy
2. Run notebook or script
python your_script.py

GPU will be used automatically if available.

Project Structure
Self_Pruning_Neural-network_Casestudy/
│
├── model code (PrunableLinear, FeedForwardNet)
├── training loop
├── sparsity calculation
├── visualization
└── README.md
What Works Well
Fully differentiable pruning
No separate pruning step required
Simple and effective implementation
Clear control using λ
Limitations (Be Honest)

Your current setup has some weaknesses:

1. Weak Architecture

A fully connected network on CIFAR-10 is inefficient.
CNN would perform much better.

Fix: Replace with Conv layers + prunable filters.

2. Soft Pruning Only

Weights are not actually removed, just scaled down.

Fix: Apply hard thresholding after training:

if gate < 0.5 → set weight = 0 permanently
3. No Fine-Tuning Step

After pruning, model is not retrained.

Fix:

Prune
Freeze masks
Fine-tune for 5–10 epochs
4. Fixed Threshold (0.5)

This is arbitrary and may not be optimal.

Fix:

Try dynamic threshold
Or percentile-based pruning
Future Improvements
Use CNN instead of MLP
Structured pruning (neurons/filters instead of weights)
Compare with magnitude pruning
Export compressed model
Measure inference speed improvements
Conclusion

This project demonstrates a self-pruning neural network where pruning is integrated into training.

It clearly shows the accuracy vs sparsity trade-off, which is critical in model compression.
