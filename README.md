# ONNX parser

Application parse ONNX model and create network in OpenCV.
You can see difference (norm) between the original PyTorch model and ONNX model.

## Command line arguments
argv[0]: ./onnx_parser
argv[1]: <path-to-model>/<model-name.onnx>
argv[2]: <path-to-input>/<input-data-name.npy>
argv[3]: <path-to-output>/<output-data-name.npy>

## Support layers
* Maxpooling
* Convolution
* Linear (FullyConnected)
* Sigmoid
