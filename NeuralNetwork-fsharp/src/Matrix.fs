(*
 * Matrix.fs
 * Author: Giacomo Rosin
 * (based on Daniel Shiffman Toy-Neural-Network-JS, https://github.com/CodingTrain/Toy-Neural-Network-JS)
 * 
 * Matrix library (double)
*)

module NeuralNetwork_fsharp.Matrix

// Type definition
type matrix = Matrix of int * int * double[][]

(*
 * create: create a (rows x cols) matrix, initialized to 0.0
 * - rows: number of rows
 * - cols: number of cols
 * return a new matrix
*)
let create (rows : int) (cols : int) : matrix =
    let data = Array.init rows (fun _ -> Array.init cols (fun _ -> 0.0))
    Matrix (rows, cols, data)

(*
 * createFromArray: create a (rows x cols) matrix, initializing the values through an array of array of double
 * - rows: number of rows
 * - cols: number of cols
 * - arr: array of array of double, used to initialize the values of the matrix
 * return a new matrix
*)
let createFromArray (rows : int) (cols : int) (arr : double[][]) : matrix =
    let data = Array.map (Array.copy) arr       // deep copy
    Matrix (rows, cols, data)        // TODO: error checking

(*
 * createFromFlatArray: create a (rows x cols) matrix, initializing the values through an array of double
 * - rows: number of rows
 * - cols: number of cols
 * - arr: array of double, used to initialize the values of the matrix
 * return a new matrix
*)
let createFromFlatArray (rows : int) (cols : int) (arr : double[]) : matrix = 
    if rows * cols <> arr.Length then invalidArg "arr" "Array length should be rows * cols"

    let data = Array.init rows (fun r -> Array.init cols (fun c -> arr.[cols * r + c]))
    Matrix (rows, cols, data)

(*
 * map: build a new matrix whose elements are the results of applying the given function to each of the elements of the matrix
 * - fn (double -> double): the function to transform elements of the matrix
 * - m: the input matrix
 * return a new matrix
*)
let map (fn : double -> double) (m : matrix) : matrix =
    let (Matrix (rows, cols, data)) = m
    let new_data = Array.map (Array.map fn) data
    Matrix (rows, cols, new_data)

(*
 * mapi: build a new array whose elements are the results of applying the given function to each of the elements of the array;
 * the integer index passed to the function indicates the index of element being transformed
 * - fn (double -> int -> int -> double): the function to transform elements and their indices
 * - m: the input matrix
 * return a new matrix
*)
let mapi (fn : double -> int -> int -> double) (m : matrix) : matrix =
    let (Matrix (rows, cols, data)) = m
    let new_data = Array.mapi (fun i el1 -> (Array.mapi (fun j el2 -> fn el2 i j) el1)) data
    Matrix (rows, cols, new_data)

(*
 * randomizeRange: build a new matrix giving a random value, between min and max, to each elements
 * - m: the input matrix
 * - (min, max): range of the random numbers
 * return a new matrix
*)
let randomizeRange (m : matrix) (min : double, max : double) : matrix =
    let rnd = System.Random()
    let getRandom (min, max) = rnd.NextDouble() * (max - min) + min

    map (fun _ -> (getRandom (min, max))) m

(*
 * randomize: build a new matrix giving a random value, between -1.0 and 1.0, to each elements
 * - m: the input matrix
 * return a new matrix
*)
let randomize (m : matrix) : matrix =
    randomizeRange m (-1.0, 1.0)

(*
 * copy: build a new matrix that contains the elements of the given matrix
 * - m: the input matrix
 * return a new matrix
*)
let copy (m : matrix) : matrix =
    let (Matrix (rows, cols, data)) = m
    Matrix (rows, cols, Array.copy data)
    
(*
 * fromArray: build a new column matrix (arr.Length x 1) that contains the elements of the given array
 * - arr: the input array
 * return a new matrix
*)
let fromArray (arr : double[]) : matrix =
    createFromFlatArray arr.Length 1 arr

(*
 * toArray: build a new double array by flattening the matrix
 * - arr: the input matrix
 * return an array of double
*)
let toArray (m : matrix) : double[] =
    let (Matrix (rows, cols, data)) = m
    Array.init (rows * cols) (fun i -> data.[i / cols].[i % cols])

(*
 * addMatrix: build a new matrix obtained by the sum of the two given matrixes
 * - a: the first matrix
 * - b: the second matrix
 * return a new matrix
*)
let addMatrix (a : matrix) (b : matrix) : matrix =
    let (Matrix (a_rows, a_cols, _)) = a
    let (Matrix (b_rows, b_cols, b_data)) = b
    if a_rows <> b_rows || a_cols <> b_cols then failwith "Not compatible"

    mapi (fun el i j -> el + b_data.[i].[j]) a

(*
 * addScalar: build a new matrix obtained adding a matrix with a scalar
 * - a: the first matrix
 * - n: scalar value
 * return a new matrix
*)
let addScalar (a : matrix) (n : double) : matrix =
    map (fun el -> el + n) a

(*
 * subMatrix: build a new matrix obtained by the subtraction of the two given matrixes
 * - a: the first matrix
 * - b: the second matrix
 * return a new matrix
*)
let subMatrix (a : matrix) (b : matrix) : matrix =
    let (Matrix (a_rows, a_cols, _)) = a
    let (Matrix (b_rows, b_cols, b_data)) = b
    if a_rows <> b_rows || a_cols <> b_cols then failwith "Not compatible"

    mapi (fun el i j -> el - b_data.[i].[j]) a


(*
 * subScalar: build a new matrix obtained subtracting from a matrix a scalar
 * - a: the first matrix
 * - n: scalar value
 * return a new matrix
*)
let subScalar (a : matrix) (n : double) : matrix =
    map (fun el -> el - n) a

(*
 * mulMatrix: build a new matrix obtained by the multiplication of the two given matrixes (hadamard product)
 * - a: the first matrix
 * - b: the second matrix
 * return a new matrix
*)
let mulMatrix (a : matrix) (b : matrix) : matrix =
    let (Matrix (a_rows, a_cols, _)) = a
    let (Matrix (b_rows, b_cols, b_data)) = b
    if a_rows <> b_rows || a_cols <> b_cols then failwith "Not compatible"

    mapi (fun el i j -> el * b_data.[i].[j]) a

(*
 * mulScalar: build a new matrix obtained multiplicating a matrix with scalar
 * - a: the first matrix
 * - n: scalar value
 * return a new matrix
*)
let mulScalar (a : matrix) (n : double) : matrix =
    map (fun el -> el * n) a

(*
 * dotProduct: build a new matrix obtained by the multiplication of the two given matrixes (dot product)
 * - a: the first matrix
 * - b: the second matrix
 * return a new matrix
*)
let dotProduct (a : matrix) (b : matrix) : matrix =
    let (Matrix (a_rows, a_cols, a_data)) = a
    let (Matrix (b_rows, b_cols, b_data)) = b
    if a_cols <> b_rows then failwith "Not compatible"
    
    let new_data = Array.init a_rows (fun _ -> Array.init b_cols (fun _ -> 0.0))
    for i in 0 .. a_rows-1 do 
        for j in 0 .. b_cols-1 do
            let mutable sum = 0.0
            for k in 0 .. a_cols-1 do 
                sum <- sum + a_data.[i].[k] * b_data.[k].[j]
            new_data.[i].[j] <- sum

    Matrix (a_rows, b_cols, new_data)

(*
 * transpose: build a new matrix obtained transposing the given one
 * - m: the input matrix
 * return a new matrix
*)
let transpose (m : matrix) : matrix =
    let (Matrix (rows, cols, data)) = m
    let new_data = Array.init cols (fun i -> Array.init rows (fun j -> data.[j].[i]))       // Array.transpose data
    Matrix (cols, rows, new_data)

(*
 * print: print the given matrix
 * - m: the input matrix
 * return the input matrix
*)
let print (m : matrix) : matrix = 
    let (Matrix (rows, cols, data)) = m
    printfn "(%dx%d)" rows cols
    Array.iter (printfn "%A") data
    m
