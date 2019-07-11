module NeuralNetwork_fsharp.NeuralNetwork


type activation_function = {
    func : double -> double
    dfunc : double -> double
}

type neural_network = {
    inputNodes : int
    hiddenNodes : int
    outputNodes : int
    weights_ih : Matrix.matrix
    weights_ho : Matrix.matrix
    bias_h : Matrix.matrix
    bias_o : Matrix.matrix
    learning_rate: double
    activation_func : activation_function
}


val create : int -> int -> int -> neural_network

val setLearningRate : neural_network -> double -> neural_network
val setActivationFunction : neural_network -> activation_function -> neural_network

val predict : neural_network -> double[] -> double[]
val train : neural_network -> double[] -> double[] -> neural_network
val mutate : neural_network -> (double -> double) -> neural_network

val copy : neural_network -> neural_network


