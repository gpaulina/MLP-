import numpy as np
import matplotlib.pyplot as plt
from csv import reader
from numpy import exp, array, random, dot
import math
from numpy import loadtxt
from sklearn.metrics import mean_squared_error
import csv


class NeuronLayer():
    def __init__(self, numberOfNeurons, numberOfInputs, bias):
        self.bias = bias
        self.weights =  2* random.random((numberOfInputs+bias, numberOfNeurons))

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1 #warstwa ukryta
        self.layer2 = layer2 #warstwa wyjścia

    # Sigmoidalna funkcja aktywacyjna
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Pochodna funkcji aktywacyjnej poprawiająca kształt krzywej
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Proces nauki sieci
    def train(self, trainingSetOfInputs, trainingSetOfOutputs, threshold, learningRate, momentum):
        prevlayer1AdjustmentOfWeights = 0 #Korekty wag warstw 1 i 2 z poprzednich iteracji
        prevlayer2AdjustmentOfWeights = 0
        if self.layer1.bias == 1:
            biasValue = np.ones((4, 1))
            trainingSetOfInputs = np.hstack((trainingSetOfInputs, biasValue))
        for iteration in range(threshold): # iterujemy po ilosci przejsc przez zestaw treningowy
            layer1Output, layer2Output = self.think(trainingSetOfInputs) # layer1/2Output = self.__sigmoid(dot(inputs, self.layer1/2.weights))

            # Oblicznaie błędu
            layer2Error = trainingSetOfOutputs - layer2Output #blad z warstwy 2 = podany - wyliczony przez neurony
            layer2MSE = mean_squared_error(trainingSetOfOutputs, layer2Output) #blad sredniokwadratowy wpisujemy do pliku mse.txt
            if (iteration % 1 == 0):
                with open('mse.txt', 'a') as f:
                    f.write("%s" % layer2MSE + "\n")

            layer2Delta = layer2Error * self.__sigmoid_derivative(layer2Output) #liczymy delte z layer2, jest to nasz koncowy blad

            layer1Error = layer2Delta.dot(self.layer2.weights.T) #error z layer1
            layer1_delta = layer1Error * self.__sigmoid_derivative(layer1Output)
            if self.layer1.bias == 1:
                layer1_delta = np.delete(layer1_delta,np.s_[-1:], axis=1)
# Momentum wpływa na aktualizowanie wag podczas uczenia.
# Momentum sprzyja zmianom wag w stałym kierunku. Ustawiamy wartosc 0 - 1
            layer1AdjustmentOfWeights = ((trainingSetOfInputs.T.dot(layer1_delta)) * (learningRate)) + ((prevlayer1AdjustmentOfWeights * (momentum)))
            layer2AdjustmentOfWeights = ((layer1Output.T.dot(layer2Delta)) * (learningRate)) + ((prevlayer2AdjustmentOfWeights * (momentum)))

            prevlayer1AdjustmentOfWeights = layer1AdjustmentOfWeights
            prevlayer2AdjustmentOfWeights = layer2AdjustmentOfWeights

            # Dopasowanie wag
            self.layer1.weights += layer1AdjustmentOfWeights
            self.layer2.weights += layer2AdjustmentOfWeights

    def think(self, inputs):
        layer1Output = self.__sigmoid(dot(inputs, self.layer1.weights))
        if self.layer2.bias == 1:
            biasValue2 = np.ones((4, 1))
            layer1Output = np.hstack((layer1Output, biasValue2))
        layer2Output = self.__sigmoid(dot(layer1Output, self.layer2.weights))

        return layer1Output, layer2Output

    def think2(self, inputs):
        biasValue3 = np.ones((4, 1))
        if self.layer1.bias == 1:
            inputs = np.hstack((inputs, biasValue3))
        layer1Output = self.__sigmoid(dot(inputs, self.layer1.weights))
        if self.layer2.bias == 1:
            layer1Output = np.hstack((layer1Output, biasValue3))
        layer2Output = self.__sigmoid(dot(layer1Output, self.layer2.weights))

        return layer1Output, layer2Output

    def print_weights(self):
        print("    Warstwa 1 (4 neurony, każdy z 4 inputami): ")
        print(self.layer1.weights)
        print("    Warstwa 2 (4 neurony, każdy z 4 inputami): ")
        print(self.layer2.weights)


if __name__ == "__main__":
    with open('mse.txt', 'a') as f:
        f.truncate(0)

    transformation = loadtxt("transformation.txt", comments="#", delimiter=" ", unpack=False)
    test = loadtxt("test.txt", comments="#", delimiter=" ", unpack=False)
    random.seed(1)

    # Tworzymy warstwę ukrytą sieci (ilosc neuronow, ilosc wejsc, bias)
    layer1 = NeuronLayer(2, 4, 0)

    # Tworzymy warstwę wyjść sieci
    layer2 = NeuronLayer(4, 2, 0)

    # Łączenie warstw w sieć
    neural_network = NeuralNetwork(layer1, layer2)

    print("Wagi początkowe: ")
    neural_network.print_weights()

    trainingSetOfInputs = transformation
    trainingSetOfOutputs = transformation

# (..., threshold, learningRate, momentum)
    neural_network.train(trainingSetOfInputs, trainingSetOfOutputs, 200, 0.9, 0.8)

    print("Wartości wag po procesie uczenia: ")
    neural_network.print_weights()

    print("Wyniki dla danych testowych(innych od użytych do nauki): ")

    hidden_state, output = neural_network.think2(test)
    print(output)

#Rysowanie wykresu

    x = []
    y = []

    with open('mse.txt', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:

            y.append(float(row[0]))
    x = [i for i in range(0, len(y))]
    plt.plot(x, y)
    plt.xlabel('cotysięczne przejście przez zestaw treningowy')
    plt.ylabel('błąd średniokwadratowy')
    plt.title('Wykres pokazujący błąd średniokwadratowy')
    plt.show()


