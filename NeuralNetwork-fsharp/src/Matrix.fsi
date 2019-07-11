module NeuralNetwork_fsharp.Matrix


type matrix = Matrix of int * int * float[][]


val create : int -> int -> matrix
val createFromArray : int -> int -> float[][] -> matrix
val createFromFlatArray : int -> int -> float[] -> matrix

val map : (float -> float) -> matrix -> matrix
val mapi : (float -> int -> int -> float) -> matrix -> matrix
val copy : matrix -> matrix
val transpose : matrix -> matrix

val randomize : matrix -> matrix
val randomizeRange : matrix -> (float * float) -> matrix

val fromArray : float[] -> matrix
val toArray : matrix -> float[]

val addMatrix : matrix -> matrix -> matrix
val addScalar : matrix -> float -> matrix
val subMatrix : matrix -> matrix -> matrix
val subScalar : matrix -> float -> matrix
val mulMatrix : matrix -> matrix -> matrix
val mulScalar : matrix -> float -> matrix
val dotProduct : matrix -> matrix -> matrix

val print : matrix -> matrix