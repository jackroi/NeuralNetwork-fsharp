(*
 * NeuralNetwork.fs
 * Author: Giacomo Rosin
 * (based on Daniel Shiffman Toy-Neural-Network-JS, https://github.com/CodingTrain/Toy-Neural-Network-JS)
 * 
 * Simple NeuralNetwork library
*)

module NeuralNetwork_fsharp.NeuralNetwork

open System


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


// sigmoid activation function
let sigmoid = {
    func = (fun x -> 1.0 / (1.0 + Math.Exp -x))
    dfunc = (fun y -> y * (1.0 - y))
}

// tanh activation function
let tanh = {
    func = (fun x -> Math.Tanh x)
    dfunc = (fun y -> 1.0 - (y * y))
}

// flip the args of a function
let flip f x y = f y x

(*
 * create: create a neural network with 3 layer
 * - inputNodes: number of input nodes
 * - hiddenNodes: number of hidden nodes
 * - outputNodes: number of output nodes
 * return a new neural network
*)
let create (inputNodes : int) (hiddenNodes : int) (outputNodes : int) : neural_network =
    {
        inputNodes = inputNodes
        hiddenNodes = hiddenNodes
        outputNodes = outputNodes
        weights_ih = Matrix.create hiddenNodes inputNodes |> Matrix.randomize
        weights_ho = Matrix.create outputNodes hiddenNodes |> Matrix.randomize
        bias_h = Matrix.create hiddenNodes 1 |> Matrix.randomize
        bias_o = Matrix.create outputNodes 1 |> Matrix.randomize
        learning_rate = 0.1
        activation_func = sigmoid
    }

(*
 * setLearningRate: build a new neural network with a new learningRate value
 * - nn: the input neural network
 * - lr: the new learning rate value
 * return a new neural network
*)
let setLearningRate (nn : neural_network) (lr : double) : neural_network =
    { nn with learning_rate = lr }

(*
 * setActivationFunction: build a new neural network with a new activation function
 * - nn: the input neural network
 * - actFunc: the new activation function
 * return a new neural network
*)
let setActivationFunction (nn : neural_network) (actFunc : activation_function) : neural_network =
    { nn with activation_func = actFunc }

(*
 * predict: calculate the outputs using the given inputs
 * - nn: the input neural network
 * - inputArray: float array containing the input values
 * return a float array containing the outputs
*)
let predict (nn : neural_network) (inputArray : double[]) : double[] =
    let inputs = Matrix.fromArray inputArray

    let hidden =
        inputs
        |> Matrix.dotProduct nn.weights_ih
        |> Matrix.addMatrix nn.bias_h
        |> Matrix.map nn.activation_func.func

    let outputs =
        hidden
        |> Matrix.dotProduct nn.weights_ho
        |> Matrix.addMatrix nn.bias_o
        |> Matrix.map nn.activation_func.func

    Matrix.toArray outputs

(*
 * train: build a new neural network trained using inputArray and targetArray
 * - nn: the input neural network
 * - inputArray: float array containing the input values
 * - targetArray: float array containing the target values
 * return a new neural network
*)
let train (nn : neural_network) (inputArray : double[]) (targetArray : double[]) : neural_network =
    // Generating the Hidden outputs
    let inputs = Matrix.fromArray inputArray

    let hidden =
        inputs
        |> Matrix.dotProduct nn.weights_ih
        |> Matrix.addMatrix nn.bias_h
        |> Matrix.map nn.activation_func.func

    // Generating the Output's outputs
    let outputs =
        hidden
        |> Matrix.dotProduct nn.weights_ho
        |> Matrix.addMatrix nn.bias_o
        |> Matrix.map nn.activation_func.func

    // Convert array to a matrix object
    let targets = Matrix.fromArray targetArray

    // Calculate the error (error = targets - outputs)
    let outputs_errors =
        Matrix.subMatrix targets outputs

    // Calculate gradient
    let gradients =
        outputs
        |> Matrix.map nn.activation_func.dfunc
        |> Matrix.mulMatrix outputs_errors
        |> flip Matrix.mulScalar nn.learning_rate

    // Calculate deltas
    let weigths_ho_deltas = 
        hidden
        |> Matrix.transpose
        |> Matrix.dotProduct gradients

    // Adjust the weigths and the bias by the calculated deltas
    let new_weights_ho = Matrix.addMatrix nn.weights_ho weigths_ho_deltas
    let new_bias_o = Matrix.addMatrix nn.bias_o gradients

    // Calculate the hidden layer errors
    let hidden_errors = 
        new_weights_ho
        |> Matrix.transpose
        |> flip Matrix.dotProduct outputs_errors

    // Calculate hidden gradient
    let hidden_gradients =
        hidden
        |> Matrix.map nn.activation_func.dfunc
        |> Matrix.mulMatrix hidden_errors
        |> flip Matrix.mulScalar nn.learning_rate

    // Calculate input -> hidden deltas
    let weights_ih_deltas =
        inputs
        |> Matrix.transpose
        |> Matrix.dotProduct hidden_gradients

    // Adjust the weigths and the bias by the calculated deltas
    let new_weights_ih = Matrix.addMatrix nn.weights_ih weights_ih_deltas
    let new_bias_h = Matrix.addMatrix nn.bias_h hidden_gradients

    { nn with
        weights_ho = new_weights_ho
        weights_ih = new_weights_ih
        bias_o = new_bias_o
        bias_h = new_bias_h
    }

(*
 * mutate: build a new neural network with mutated weights (using a mutation function)
 * - nn: the input neural network
 * - mutationFunc (double -> double): mutation function
 * return a new neural network
*)
let mutate (nn : neural_network) (mutationFunc : double -> double) : neural_network =
    { nn with
        weights_ih = Matrix.map mutationFunc nn.weights_ih
        weights_ho = Matrix.map mutationFunc nn.weights_ih
        bias_h = Matrix.map mutationFunc nn.bias_h
        bias_o = Matrix.map mutationFunc nn.bias_o
    }

(*
 * mutate: build a new neural network copying the input one
 * - nn: the input neural network
 * return a new neural network
*)
let copy (nn : neural_network) : neural_network =
    { nn with
        weights_ih = Matrix.copy nn.weights_ih
        weights_ho = Matrix.copy nn.weights_ho
        bias_h = Matrix.copy nn.bias_h
        bias_o = Matrix.copy nn.bias_o
    }
