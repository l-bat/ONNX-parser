#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <opencv2/dnn/all_layers.hpp>
#include "onnx.pb.h"

using namespace std;

static std::vector<int> getShape(const std::string& header) {
    std::string field = "'shape':";
    int idx = header.find(field);

    int from = header.find('(', idx + field.size()) + 1;
    int to = header.find(')', from);

    std::string shapeStr = header.substr(from, to - from);
    if (shapeStr.empty())
        return std::vector<int>(1, 1);

    // Remove all commas.
    shapeStr.erase(std::remove(shapeStr.begin(), shapeStr.end(), ','),
                   shapeStr.end());

    std::istringstream ss(shapeStr);
    int value;

    std::vector<int> shape;
    while (ss >> value)
    {
        shape.push_back(value);
    }
    return shape;
}

cv::Mat blobFromNPY(const std::string& path) {
    std::ifstream ifs(path.c_str(), std::ios::binary);

    std::string magic(6, '*');
    ifs.read(&magic[0], magic.size());
    ifs.ignore(1);  // Skip major version byte.
    ifs.ignore(1);  // Skip minor version byte.

    unsigned short headerSize;
    ifs.read((char*)&headerSize, sizeof(headerSize));

    std::string header(headerSize, '*');
    ifs.read(&header[0], header.size());

    std::vector<int> shape = getShape(header);

    cv::Mat blob(shape, CV_32F);
    ifs.read((char*)blob.data, blob.total() * blob.elemSize());
    return blob;
}

std::map<std::string, cv::Mat> get_weight(const onnx::GraphProto& graph_proto) {
  onnx::TensorProto tensor_proto;
  std::map<std::string, cv::Mat> layers_weights;
  onnx::TensorProto_DataType datatype;
  std::cout << "init size = " << graph_proto.initializer_size() <<'\n';
  for (int i = 0; i <  graph_proto.initializer_size(); i++) {
    tensor_proto = graph_proto.initializer(i);
    datatype = tensor_proto.data_type();

    if (datatype == onnx::TensorProto_DataType_FLOAT) {
       std::vector<int> sizes;
       for (int i=0; i < tensor_proto.dims_size(); i++) {
          sizes.push_back(tensor_proto.dims(i));
       }
       char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
       cv::Mat blob(tensor_proto.dims_size(), sizes.data(), CV_32FC1, val);
       std::cout << "blob size = " << blob.size << '\n';
       layers_weights.insert(std::pair<std::string, cv::Mat>(tensor_proto.name(), blob.clone()));
    }
  }
  return layers_weights;
}

cv::dnn::LayerParams get_lp(const onnx::NodeProto& node_proto) {
  cv::dnn::LayerParams lp;
  for(int i = 0; i < node_proto.attribute_size(); i++) {
    const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
    std::string attribute_name = attribute_proto.name();
    if (attribute_proto.has_i()) {
      lp.set(attribute_proto.name(), attribute_proto.i());
    } else if (attribute_proto.has_f()) {
      lp.set(attribute_proto.name(), attribute_proto.f());
    } else if (attribute_proto.has_s()) {
      lp.set(attribute_proto.name(), attribute_proto.s());
    } // else if (attribute_proto.has_t()) {
    //   lp.set(attribute_proto.name(), attribute_proto.t());
    // } else if (attribute_proto.has_g()) {
    //   lp.set(attribute_proto.name(), attribute_proto.g());
    // }
    for (int i = 0; i < attribute_proto.floats_size(); i++) {
      lp.set(attribute_proto.name(), attribute_proto.floats(i));
    }
    for (int i = 0; i < attribute_proto.ints_size(); i++) {
      if(attribute_name == "kernel_shape") {
        lp.set("kernel_h",  attribute_proto.ints(0));
        lp.set("kernel_w",  attribute_proto.ints(1));
      } else if(attribute_name == "strides") {
        lp.set("stride_h",  attribute_proto.ints(0));
        lp.set("stride_w",  attribute_proto.ints(1));
      } else if(attribute_name == "pads") {
        lp.set("pad_h",  attribute_proto.ints(0));
        lp.set("pad_w",  attribute_proto.ints(1));
      } else
      lp.set(attribute_proto.name(), attribute_proto.ints(i));
    }
    for (int i = 0; i < attribute_proto.strings_size(); i++) {
      lp.set(attribute_proto.name(), attribute_proto.strings(i));
    }
    // for (int i = 0; i < attribute_proto.tensors_size(); i++) {
    //   lp.set(attribute_proto.name(), attribute_proto.tensors(i));
    // }
    // for (int i = 0; i < attribute_proto.graphs_size(); i++) {
    //   lp.set(attribute_proto.name(), attribute_proto.graphs(i));
    // }
  }
  std::string layer_type = node_proto.op_type();
  if (layer_type == "MaxPool") {
    lp.type = "Pooling";
    lp.set("pool", "MAX");
  } else if (layer_type == "Gemm") {
    lp.type = "InnerProduct";
  } else if (layer_type == "Conv") {
    lp.type = "Convolution";
  } else {
    lp.type = node_proto.op_type();
  }
  return lp;
}

cv::dnn::Net create_net(const onnx::ModelProto& model_proto) {
  cv::dnn::Net net;
  if(model_proto.has_graph()) {
    onnx::GraphProto graph_proto = model_proto.graph();
    onnx::NodeProto node_proto;
    cv::dnn::LayerParams lp;
    std::map<std::string, cv::Mat> weights = get_weight(graph_proto);
    std::map<std::string, cv::Mat>::iterator weight, bias;

    for(int i = 0; i < graph_proto.node_size(); i++) {
      node_proto = graph_proto.node(i);

      lp = get_lp(node_proto);
      lp.name = node_proto.op_type() + "_" + std::to_string(i);
      std::cout << lp.name << '\n';

      std::cout << "input size = " << node_proto.input_size() << '\n';

      if (node_proto.input_size() == 2) {   // weights
        int num = std::stoi(node_proto.input(1));
        weight = weights.find(graph_proto.initializer(num - 1).name());
        if (weight != weights.end()) {
          lp.blobs.push_back(weight->second);
        }
        lp.set("bias_term", false);
      } else if(node_proto.input_size() > 2) {  //  weights + bias
          lp.set("bias_term", true);
          int num = std::stoi(node_proto.input(1));
          weight = weights.find(graph_proto.initializer(num - 1).name());
          if (weight != weights.end()) {
            lp.blobs.push_back(weight->second);
          }
          num = std::stoi(node_proto.input(2));
          bias =  weights.find(graph_proto.initializer(num - 1).name());
          if (bias != weights.end()) {
            lp.blobs.push_back(bias->second);
          }
          int num_output = static_cast<int> (lp.blobs[1].total());
          lp.set("num_output", num_output);
        }

     //std::cout << "______Layer Params______" << '\n' << lp << '\n';
     net.addLayerToPrev(lp.name, lp.type, lp);
    }
  }
  return net;
}

int main(int argc, char const *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " ONNX_FILE, input and output" << '\n';
    return -1;
  }

  onnx::ModelProto model_proto;
  {
    std::fstream input(argv[1], ios::in | ios::binary);
    if (!input) {
      std::cout << argv[1] << ": file not found." << '\n';
    } else if (!model_proto.ParseFromIstream(&input)) {
      std::cerr << "Failed to parse onnx model." << std::endl;
      return -1;
    }
  }


  cv::dnn::Net net;
  net = create_net(model_proto);
  cv::Mat input = blobFromNPY(argv[2]);
  std::cout << input.size << '\n';
  net.setInput(input);
  cv::Mat output = net.forward();
  std::cout << output.size << '\n';

  cv::Mat outputBlob = blobFromNPY(argv[3]);
  std::cout << outputBlob.size << '\n';
  double normL2 = cv::norm(output, outputBlob, cv::NORM_INF);
  std::cout << "norm = " << normL2 << '\n' << '\n'<< '\n' ;

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
