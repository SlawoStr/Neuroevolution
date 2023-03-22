#include "NeuralNetwork.h"
#include "SelectionAlgo/WheelSelection.h"
#include "Neuroevolution.h"
#include "SelectionAlgo/TournamentSelection.h"
#include "SelectionAlgo/TruncationSelection.h"
#include "SelectionAlgo/WheelSelection.h"
#include "CrossoverAlgo/SPCCrossover.h"
#include "CrossoverAlgo/MPCCrossover.h"
#include "CrossoverAlgo/UniformCrossover.h"
#include "MutationAlgo/AddMutation.h"
#include "MutationAlgo/NewMutation.h"
#include "Utility/Timer.h"
#include <iostream>

int main()
{
    size_t modelNumber{ 20000 };
    size_t parentPairNumber{ 9900 };
    for (int i = 1; i <= 20; ++i)
    {
        NeuralNetwork network(30, 10, ActivationFunction::SIGMOID);
        network.addLayer(25, ActivationFunction::RELU);
        network.addLayer(15, ActivationFunction::RELU);
        Neuroevolution manager(network, modelNumber, parentPairNumber, i);
        manager.setSelection<WheelSelection>(modelNumber);
        manager.setCrossover<MPCCrossover>(0.9f);
        manager.setMutation<NewMutation>(1.0f, 0.1f);
        std::vector<std::pair<int, double>>  fitnessVec;
        for (int i = 0; i < modelNumber; ++i)
        {
            fitnessVec.push_back({ i,i * 2.0f });
        }
        Timer t;
        t.start();
        manager.run(fitnessVec);
        t.stop();
        std::cout << t.measure() * 1000 << std::endl;
    }
    return 0;
}

