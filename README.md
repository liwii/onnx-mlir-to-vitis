# onnx-mlir-to-vitis
Hardware Synthesis Compiler for Deep Neural Network.
It converts MLIR code from [onnx-mlir](https://github.com/onnx/onnx-mlir) to C++ code that can be synthesized to RTL with [Vitis HLS](https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/introductionvitishls.html)

## Dependencies
- onnx-mlir (it might not only work for other commits than https://github.com/onnx/onnx-mlir/commit/76397990d4ef99a088f86e3615af9ca438d01fe0)
- Vitis HLS
- numpy

## Tutorial
Train your network with machine learning libraries, and export the result as ONNX.

```
python tutorial/mnist/mnist.py
```

Run onnx-mlir to emit MLIR representation of the network.

```
path/to/onnx-mlir --EmitMLIR tutorial/mnist/mnist.onnx
```

Run compiler.py to convert the MLIR code to C++ code synthesizable with Vitis HLS.

```
python compiler.py tutorial/mnist/mnist.onnx.mlir > tutorial/mnist/graph.cpp
```

Finally, run synthesis on Vitis HLS. This [blog](https://blog.n-hassy.info/2021/05/vitis-hls-to-fpga-1/) might be useful. (Written in Japanese)

## Note
Currently, the compiler only supports networks with
- softmax
- relu
- fully-connected

layers. It does not support batch-processing either, so you need to set single element as the input data for the network when you export ONNX network.