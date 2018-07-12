#include <iostream>
#include <fstream>
#include <string>

#include "onnx.pb.h"

using namespace std;

void get_weight (const onnx::GraphProto& graph_proto) {
  for(int i = 0; i < graph_proto.initializer_size(); i++) {
		const onnx::TensorProto& tensor_proto = graph_proto.initializer(i);
	  const onnx::TensorProto_DataType& datatype = tensor_proto.data_type();
		int tensor_size  = 1;

		for(int j = 0; j < tensor_proto.dims_size(); j++) {
			tensor_size *= tensor_proto.dims(j);
		}
    std::cout << "Tensor: " << tensor_proto.name() << '\n';
    std::cout << "Tensor size: " << tensor_size << '\n';

    if(datatype == onnx::TensorProto_DataType_FLOAT) {
      std::string raw_data_val = tensor_proto.raw_data();
      std::cout << "data in tensor: " << '\n';
      const char * val = raw_data_val.c_str();

      for(int k = 0; k < tensor_size*4 - 4; k+=4) {
				//float weight;
				char b[] = {val[k], val[k+1], val[k+2], val[k+3]};
        std::cout << "weight: " << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << '\n';
      }
    }
  }
}

void get_layer_params (const onnx::NodeProto& node_proto) {
  // for (int i = 0; i < node_proto.input_size(); i++) {
  //   std::cout << "Input: " << node_proto.input(i) << '\n';
  // }
  // for (int i = 0; i < node_proto.output_size(); i++) {
  //   std::cout << "Output: " << node_proto.output(i) << '\n';
  // }

  std::string layer_type = node_proto.op_type();
  if(layer_type == "Conv") {
    std::cout << "_______convolution layer_______" << '\n';
    int pad_h, pad_w;
    int stride_h, stride_w;
    int kernel_h, kernel_w;
    int dilation_h, dilation_w;

    int group = -1;

  std::cout << "quantity of attribute = " << node_proto.attribute_size() << '\n';
  for(int i = 0; i < node_proto.attribute_size(); i++) {
    const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
    std::string attribute_name = attribute_proto.name();
    std::cout << "attribute[" << i << "] = " << attribute_name << '\n';

// TO DO: add groups and bias to hyperparams

      if(attribute_name == "dilations") {
        dilation_h = attribute_proto.ints(0);
        std::cout << "dilation height: " << dilation_h << '\n';
        dilation_w = attribute_proto.ints(1);
        std::cout << "dilation widht: " << dilation_w << '\n';
      } else if(attribute_name == "group") {
       std::cout << attribute_proto.ByteSizeLong() << '\n';
       group = attribute_proto.i();
       std::cout << "group: " << group << '\n';
      } else if(attribute_name == "kernel_shape") {
        kernel_h = attribute_proto.ints(0);
        std::cout << "kernel height: " << kernel_h << '\n';
        kernel_w = attribute_proto.ints(1);
        std::cout << "kernel width: " << kernel_w << '\n';
      } else if(attribute_name == "pads") {
        pad_h = attribute_proto.ints(0);
        std::cout << "pad height: " << pad_h << '\n';
        pad_w = attribute_proto.ints(1);
        std::cout << "pad width: " << pad_w << '\n';
      } else if(attribute_name == "strides") {
        stride_h = attribute_proto.ints(0);
        std::cout << "stride height: " << stride_h << '\n';
        stride_w = attribute_proto.ints(1);
        std::cout << "stride width: " << stride_w << '\n';
      }

    }
  }
  else if(layer_type == "MaxPool") {
    std::cout << "_______maxpooling layer_______" << '\n';
    int pad_h, pad_w;
    int stride_h, stride_w;
    int kernel_h, kernel_w;

    std::cout << "quantity of attribute = " << node_proto.attribute_size() << '\n';
    for(int i = 0; i < node_proto.attribute_size(); i++) {
      const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
      std::string attribute_name = attribute_proto.name();
      std::cout << "attribute[" << i << "] = " << attribute_name << '\n';

      if(attribute_name == "strides") {
        stride_h = attribute_proto.ints(0);
        std::cout << "stride height: " << stride_h << '\n';
        stride_w = attribute_proto.ints(1);
        std::cout << "stride width: " << stride_w << '\n';
      }
      else if(attribute_name == "pads") {
        pad_h = attribute_proto.ints(0);
        std::cout << "pad height: " << pad_h << '\n';
        pad_w = attribute_proto.ints(1);
        std::cout << "pad width: " << pad_w << '\n';
      }
      else if(attribute_name == "kernel_shape") {
        kernel_h = attribute_proto.ints(0);
        std::cout << "kernel height: " << kernel_h << '\n';
        kernel_w = attribute_proto.ints(1);
        std::cout << "kernel width: " << kernel_w << '\n';
      }
    }
  }
}

void parse_onnx_model(const onnx::ModelProto& model_proto) {
  onnx::GraphProto graph_proto;
  onnx::NodeProto node_proto;
  if(model_proto.has_graph()) {
    std::cout << "Parsing the onnx model." << std::endl;
    graph_proto = model_proto.graph();
  }
  // if(graph_proto.has_name()) {
	// 	std::cout << "Extracting the weights for : " << graph_proto.name() << std::endl;
	// }
//  get_weight(graph_proto);

  for(int i = 0; i < graph_proto.node_size(); i++) {
      node_proto = graph_proto.node(i);
      get_layer_params(node_proto);
  }
}

int main(int argc, char const *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " ONNX_FILE" << '\n';
    return -1;
  }

  onnx::ModelProto model_proto;


  {
    std::fstream input(argv[1], ios::in | ios::binary);
    if (!input) {
      std::cout << argv[1] << ": file not found. Creating a new file." << '\n';
    } else if (!model_proto.ParseFromIstream(&input)) {
      std::cerr << "Failed to parse onnx model." << std::endl;
      return -1;
    }
  }

  parse_onnx_model(model_proto);

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
