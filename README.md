# Neuroevolution

Implementation of classical neuroevolution. 

Selection Algorithms:

- Truncation Selection
- Wheel Selection
- Tournament Selection

Crossover Algorithms:

- Single point crossover
- Two point crossover
- Uniform crossover

Mutation Algorithms:

- Add random value to weight
- Generate new weight value
- Change weight sign
- Scale weight by a value

```c
// Create neural network
NeuralNetwork network(30, 10, ActivationFunction::SIGMOID);
network.addLayer(25, ActivationFunction::RELU);
network.addLayer(15, ActivationFunction::RELU);
// Create neuroevolution
Neuroevolution manager(network, modelNumber, parentPairNumber, threadNumber);
// Set genetic operators
manager.setSelection<WheelSelection>(modelNumber);
manager.setCrossover<MPCCrossover>(crossoverProb);
manager.setMutation<NewMutation>(mutationProb, geneMutationProb);
// Run neuroevolution by sending results of neural network models (fitness function resutlts)
manager.run(fitnessVec);
```
