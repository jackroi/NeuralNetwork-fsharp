// XOR example

module NeuralNetwork_fsharp.Main

type td = {
    inputs : double[]
    targets : double[]
}

let trainingData = [|
    {
        inputs = [| 0.0; 1.0 |]
        targets = [| 1.0 |]
    };
    {
        inputs = [| 1.0; 0.0 |]
        targets = [| 1.0 |]
    };
    {
        inputs = [| 0.0; 0.0 |]
        targets = [| 0.0 |]
    };
    {
        inputs = [| 1.0; 1.0 |]
        targets = [| 0.0 |]
    }
|]


[<EntryPoint>]
let main _ = 
    let mutable nn = NeuralNetwork.create 2 4 1

    nn <- NeuralNetwork.setLearningRate nn 0.1

    printfn "%A" (NeuralNetwork.predict nn [| 0.0; 1.0 |])
    printfn "%A" (NeuralNetwork.predict nn [| 1.0; 0.0 |])
    printfn "%A" (NeuralNetwork.predict nn [| 0.0; 0.0 |])
    printfn "%A" (NeuralNetwork.predict nn [| 1.0; 1.0 |])
    printfn ""

    let rnd = System.Random()

    for _ in 0 .. 500000 do
        let index = rnd.Next (0, 4)
        let data = trainingData.[index]

        nn <- NeuralNetwork.train nn data.inputs data.targets

    
    printfn "%A" (NeuralNetwork.predict nn [| 0.0; 1.0 |])
    printfn "%A" (NeuralNetwork.predict nn [| 1.0; 0.0 |])
    printfn "%A" (NeuralNetwork.predict nn [| 0.0; 0.0 |])
    printfn "%A" (NeuralNetwork.predict nn [| 1.0; 1.0 |])


    printfn "press [enter] to exit"
    System.Console.ReadLine() |> ignore

    0
