module  {
  func @main_graph(%arg0: memref<4xf32>) -> memref<3xf32> attributes {input_names = ["input.1"], output_names = ["18"]} {
    %cst = constant 0.000000e+00 : f32
    %cst_0 = constant 0xFF800000 : f32
    %c0_i64 = constant 0 : i64
    %c4_i64 = constant 4 : i64
    %c8_i64 = constant 8 : i64
    %c20_i64 = constant 20 : i64
    %c32_i64 = constant 32 : i64
    %c72_i64 = constant 72 : i64
    %c112_i64 = constant 112 : i64
    %0 = alloc() : memref<152xi8>
    %1 = "krnl.getref"(%0, %c0_i64) : (memref<152xi8>, i64) -> memref<f32>
    %2 = "krnl.getref"(%0, %c4_i64) : (memref<152xi8>, i64) -> memref<f32>
    %3 = alloc() : memref<3xf32>
    %4 = "krnl.getref"(%0, %c8_i64) : (memref<152xi8>, i64) -> memref<3xf32>
    %5 = "krnl.getref"(%0, %c20_i64) : (memref<152xi8>, i64) -> memref<3xf32>
    %6 = "krnl.getref"(%0, %c32_i64) : (memref<152xi8>, i64) -> memref<10xf32>
    %7 = "krnl.getref"(%0, %c72_i64) : (memref<152xi8>, i64) -> memref<10xf32>
    %8 = "krnl.getref"(%0, %c32_i64) : (memref<152xi8>, i64) -> memref<10xf32>
    %9 = "krnl.getref"(%0, %c112_i64) : (memref<152xi8>, i64) -> memref<10xf32>
    %10 = "krnl.getref"(%0, %c72_i64) : (memref<152xi8>, i64) -> memref<10xf32>
    %11 = "krnl.getref"(%0, %c32_i64) : (memref<152xi8>, i64) -> memref<10xf32>
    %12 = "krnl.global"() {name = "constant_0", shape = [4, 10], value = dense<[[-0.246135831, -0.092899084, 0.619207918, -0.424382687, -0.443481505, -0.928455293, 1.2941761, 0.302099407, -0.41976881, 0.269429982], [0.137399733, 0.227514327, 0.624374032, 0.130959868, 0.187045813, -0.788561403, 1.1330024, -0.682371795, -0.339631319, -0.402945668], [-0.0237659812, -0.4105407, -0.543239653, -1.333080e-01, 0.0517955422, 2.31133199, -0.525980234, 3.502840e-01, -0.388360798, -0.43208462], [-0.208887339, -0.242459774, -0.797830462, -0.0765706896, 0.176199973, 2.12913418, -1.08306837, 0.324207634, 0.0472493768, -0.356763303]]> : tensor<4x10xf32>} : () -> memref<4x10xf32>
    affine.for %arg1 = 0 to 10 {
      %20 = alloca() : memref<f32>
      affine.store %cst, %20[] : memref<f32>
      affine.for %arg2 = 0 to 4 {
        %22 = affine.load %arg0[%arg2] : memref<4xf32>
        %23 = affine.load %12[%arg2, %arg1] : memref<4x10xf32>
        %24 = affine.load %20[] : memref<f32>
        %25 = mulf %22, %23 : f32
        %26 = addf %24, %25 : f32
        affine.store %26, %20[] : memref<f32>
      }
      %21 = affine.load %20[] : memref<f32>
      affine.store %21, %11[%arg1] : memref<10xf32>
    }
    %13 = "krnl.global"() {name = "constant_1", shape = [10], value = dense<[-0.311646879, -0.431221306, 0.469752908, 0.481094539, 0.272461116, -0.248878419, 0.2797409, -0.332013369, 1.873110e-01, 0.382959753]> : tensor<10xf32>} : () -> memref<10xf32>
    affine.for %arg1 = 0 to 10 {
      %20 = affine.load %11[%arg1] : memref<10xf32>
      %21 = affine.load %13[%arg1] : memref<10xf32>
      %22 = addf %20, %21 : f32
      affine.store %22, %10[%arg1] : memref<10xf32>
    }
    affine.for %arg1 = 0 to 10 {
      %20 = affine.load %10[%arg1] : memref<10xf32>
      %21 = cmpf olt, %20, %cst : f32
      %22 = select %21, %cst, %20 : f32
      affine.store %22, %9[%arg1] : memref<10xf32>
    }
    %14 = "krnl.global"() {name = "constant_2", shape = [10, 10], value = dense<[[-0.0532949269, 0.228087217, 0.124331832, 0.274742514, 0.245896965, -0.101453468, -0.0413132906, 0.0188050568, -0.292971551, 0.171668559], [-0.0384894311, 0.265742451, -0.184304833, 0.16724506, -0.131404489, 0.280650884, 0.0990278422, -0.0755590498, 0.158147305, 0.146666676], [2.949350e-02, 0.122816294, 0.110189289, 0.0299925655, 0.936579823, -0.229376689, -0.0900963246, 0.212030262, -0.0886185616, 0.164182127], [0.215725154, 0.0559133291, -0.297151536, 0.197408885, -0.0643002986, -0.228644639, -0.0735795945, 0.0889647603, -0.0618865787, 0.251100391], [0.305684239, -0.142204314, -0.0708866566, 0.0292969048, -0.0850767791, -0.12107113, -0.211682558, -0.0204363465, -0.17885904, 0.226244897], [2.925020e-01, -0.128147095, 0.0107858777, -5.36204432E-4, -1.72136021, 1.63067079, -0.233031392, -0.0384222865, 0.0801345631, -0.210525796], [6.556150e-02, -0.156288207, -0.157138765, 0.0115601812, 1.6676966, 0.136670098, -0.249013394, -0.121973731, -0.146361321, -0.05800635], [-0.00140071847, 0.078114748, -0.207607672, 0.0298967641, -0.308185518, 0.481457978, 0.11561197, 0.251934975, 0.0759283155, 0.00228716619], [-0.210867584, 0.108702987, 0.300003022, -0.291363239, 6.310600e-02, 0.143540978, 0.110801876, 0.268966049, -7.130660e-02, -0.31040135], [0.276075035, -0.275792181, 0.197648436, 0.0460505784, 0.0715084373, 0.0426917635, 0.100050777, 0.0555913784, -0.2080051, -0.270867348]]> : tensor<10x10xf32>} : () -> memref<10x10xf32>
    affine.for %arg1 = 0 to 10 {
      %20 = alloca() : memref<f32>
      affine.store %cst, %20[] : memref<f32>
      affine.for %arg2 = 0 to 10 {
        %22 = affine.load %9[%arg2] : memref<10xf32>
        %23 = affine.load %14[%arg2, %arg1] : memref<10x10xf32>
        %24 = affine.load %20[] : memref<f32>
        %25 = mulf %22, %23 : f32
        %26 = addf %24, %25 : f32
        affine.store %26, %20[] : memref<f32>
      }
      %21 = affine.load %20[] : memref<f32>
      affine.store %21, %8[%arg1] : memref<10xf32>
    }
    %15 = "krnl.global"() {name = "constant_3", shape = [10], value = dense<[-0.176395148, -1.067030e-01, -0.236138284, -0.292516053, 1.02632129, -0.257343292, 0.148608744, 0.0442748666, 0.0406188555, 0.0537791885]> : tensor<10xf32>} : () -> memref<10xf32>
    affine.for %arg1 = 0 to 10 {
      %20 = affine.load %8[%arg1] : memref<10xf32>
      %21 = affine.load %15[%arg1] : memref<10xf32>
      %22 = addf %20, %21 : f32
      affine.store %22, %7[%arg1] : memref<10xf32>
    }
    affine.for %arg1 = 0 to 10 {
      %20 = affine.load %7[%arg1] : memref<10xf32>
      %21 = cmpf olt, %20, %cst : f32
      %22 = select %21, %cst, %20 : f32
      affine.store %22, %6[%arg1] : memref<10xf32>
    }
    %16 = "krnl.global"() {name = "constant_4", shape = [10, 3], value = dense<[[-0.335769773, -0.0299628582, 6.909720e-02], [-0.277574033, 0.00144463778, -0.00670164824], [-0.00624275208, 0.287015587, -0.0910358578], [-0.0627249256, 0.280814886, 0.0560187213], [1.32039201, 0.975252747, -1.82740545], [-1.53125393, 0.351987571, 1.0384022], [0.153964728, -0.205612421, -0.137824595], [0.147223398, 0.110498711, 0.250904888], [0.246240526, -0.0115929199, -0.165132374], [-0.234203637, -0.244696498, -0.110656291]]> : tensor<10x3xf32>} : () -> memref<10x3xf32>
    affine.for %arg1 = 0 to 3 {
      %20 = alloca() : memref<f32>
      affine.store %cst, %20[] : memref<f32>
      affine.for %arg2 = 0 to 10 {
        %22 = affine.load %6[%arg2] : memref<10xf32>
        %23 = affine.load %16[%arg2, %arg1] : memref<10x3xf32>
        %24 = affine.load %20[] : memref<f32>
        %25 = mulf %22, %23 : f32
        %26 = addf %24, %25 : f32
        affine.store %26, %20[] : memref<f32>
      }
      %21 = affine.load %20[] : memref<f32>
      affine.store %21, %5[%arg1] : memref<3xf32>
    }
    %17 = "krnl.global"() {name = "constant_5", shape = [3], value = dense<[0.256047785, 0.0399362482, -0.434387296]> : tensor<3xf32>} : () -> memref<3xf32>
    affine.for %arg1 = 0 to 3 {
      %20 = affine.load %5[%arg1] : memref<3xf32>
      %21 = affine.load %17[%arg1] : memref<3xf32>
      %22 = addf %20, %21 : f32
      affine.store %22, %4[%arg1] : memref<3xf32>
    }
    affine.store %cst, %2[] : memref<f32>
    affine.store %cst_0, %1[] : memref<f32>
    affine.for %arg1 = 0 to 3 {
      %20 = affine.load %1[] : memref<f32>
      %21 = affine.load %4[%arg1] : memref<3xf32>
      %22 = cmpf ogt, %20, %21 : f32
      %23 = select %22, %20, %21 : f32
      affine.store %23, %1[] : memref<f32>
    }
    %18 = affine.load %1[] : memref<f32>
    affine.for %arg1 = 0 to 3 {
      %20 = affine.load %2[] : memref<f32>
      %21 = affine.load %4[%arg1] : memref<3xf32>
      %22 = subf %21, %18 : f32
      %23 = exp %22 : f32
      %24 = addf %20, %23 : f32
      affine.store %24, %2[] : memref<f32>
      affine.store %23, %3[%arg1] : memref<3xf32>
    }
    %19 = affine.load %2[] : memref<f32>
    affine.for %arg1 = 0 to 3 {
      %20 = affine.load %3[%arg1] : memref<3xf32>
      %21 = divf %20, %19 : f32
      affine.store %21, %3[%arg1] : memref<3xf32>
    }
    dealloc %0 : memref<152xi8>
    return %3 : memref<3xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22float\22 , \22dims\22 : [4]  }\0A\0A]\00@[   { \22type\22 : \22float\22 , \22dims\22 : [3]  }\0A\0A]\00"} : () -> ()
}
