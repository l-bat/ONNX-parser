#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <opencv2/dnn/all_layers.hpp>
#include "onnx.pb.h"

using namespace std;

void print_matrix (cv::Mat mat, int* size) {
  for (int i = 0; i < size[0]; i++)
    for (int j = 0; j < size[1]; j++)
      for (int k = 0; k < size[2]; k++)
        for (int m = 0; m < size[3]; m++)
  std::cout << mat.at<cv::Vec4f>(i,j,k)[m] << ", ";
}

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



std::map<std::string, cv::Mat> get_weight (const onnx::GraphProto& graph_proto) {
  onnx::TensorProto tensor_proto;
  std::map<std::string, cv::Mat> map;
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
       std::cout << "blob size = " << blob.size() << '\n';
       map.insert(std::pair<std::string, cv::Mat>(tensor_proto.name(), blob.clone()));
    }
  }
    //int size[tensor.dims_size()];
    //for (int i = 0; i < tensor.dims_size(); i++) {
      //size[i] = tensor.dims(i);
    //}
    //char* val = const_cast<char*>(tensor.raw_data().c_str());
//    cv::Mat mat(tensor.dims_size(), size, tensor.raw_data().c_str());  // bug
    //cv::Mat mat(tensor.dims_size(), size, CV_32FC1, val);  // bug
  //}
  return map;
}

//  const onnx::TensorProto weights = graph_proto.initializer(0);
  //std::cout << "weights size = " << weights.dims_size() <<'\n';
  //std::cout << weights.dims(0) << '\n';
  //std::cout << weights.dims(1) << '\n';
  //std::cout << weights.data_type() << '\n';

   //std::cout << "size = " << weights.has_raw_data() << '\n';
   //std::string data = weights.raw_data();
   //std::cout << data.size() << '\n';
   //std::cout << "name = " << weights.name() << '\n';

  // std::cout << "inp size = " << graph_proto.input_size() << '\n';
  // for(int i = 0; i < graph_proto.input_size(); i++) {
  //   const onnx::ValueInfoProto& value_info_proto = graph_proto.input(i);
	// 	std::string layer_input = value_info_proto.name();
	// 	std::vector<int> dims;
  //
  //   const onnx::TypeProto& type_proto = value_info_proto.type();
  //   const onnx::TypeProto::Tensor& tensor = type_proto.tensor_type();
  //   const onnx::TensorShapeProto& tensor_shape = tensor.shape();
  //
  //   std::cout << "ten shape size = " << tensor_shape.dim_size() << '\n';
  //
  //   for(int j=0; j < tensor_shape.dim_size(); j++) {
  //   	const onnx::TensorShapeProto::Dimension& dimension = tensor_shape.dim(j);
  //     std::cout << "dim val = "<<  dimension.dim_value() << '\n';
  //   	dims.push_back(dimension.dim_value());
  // 	}
  //
  //   input_tensor_dim_map[layer_input] = dims;
  //
  //   // node_proto = graph_proto.node(i);
  //   // if(node_proto.input_size() > 1) {
  //   //   std::string layer_weights = node_proto.input(1);
  //   //   std::cout << "weights: " << layer_weights << '\n';
  //   // }
  //   // if(node_proto.input_size() > 2) {
  //   //   std::string layer_bias = node_proto.input(2);
  //   //   std::cout << "bias: " << layer_bias << '\n';
  //   // }
  // }

  // std::string layer_type = node_proto.op_type();
  // if(layer_type == "Gemm") {
  //   std::string layer_weights = " ";
  //   std::vector<int> weight_dims, bias_dims;
  //   std::vector<int> weight_dims_gemm;
  //   if(layer_details.size() > 4) {
  //     layer_weights = layer_details.find("weights")->second;
  //     weight_dims = input_tensor_dim_map.find(layer_weights)->second;
  //     weight_dims_gemm.push_back(in_w);
  //     weight_dims_gemm.push_back(in_h);
  //     weight_dims_gemm.push_back(in_c);
  //     weight_dims_gemm.push_back(weight_dims[0]);
  //
  //   }
  //   std::string layer_bias = " ";
  //   if(layer_details.size() > 5) {
  //     layer_bias = layer_details.find("bias")->second;
  //     bias_dims = input_tensor_dim_map.find(layer_bias)->second;
  //   }
  //
  //   out_n = 1;
  //   out_c = weight_dims[0];
  //   out_h = 1;
  //   out_w = 1;
  //
  //   if(layer_details.size() > 4) {
  //     in_out_map[layer_weights] = weight_dims_gemm;
  //   }
  //
  //   if(layer_details.size() > 5) {
  //     in_out_map[layer_bias] = bias_dims;
  //   }
  // }
  //
//}

void print_layer_params(const onnx::NodeProto& node_proto) {
  // for (int i = 0; i < node_proto.input_size(); i++) {
  //   std::cout << "Input: " << node_proto.input(i) << '\n';
  // }
  // for (int i = 0; i < node_proto.output_size(); i++) {
  //   std::cout << "Output: " << node_proto.output(i) << '\n';
  // }

  std::string layer_type = node_proto.op_type();
  std::cout << "layer type: " << layer_type << '\n';

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


      if(attribute_name == "dilations") {
        dilation_h = attribute_proto.ints(0);
        std::cout << "dilation height: " << dilation_h << '\n';
        dilation_w = attribute_proto.ints(1);
        std::cout << "dilation widht: " << dilation_w << '\n';
      } else if(attribute_name == "group") {
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
  // else if(layer_type == "Sigmoid") {
  //
  //   //int lrn_size;
  //
  //   std::cout << "quantity of attribute = " << node_proto.attribute_size() << '\n';
  //   for(int i=0; i < node_proto.attribute_size(); i++) {
  //     const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
  //     std::string attribute_name = attribute_proto.name();
  //     std::cout << "attribute[" << i << "] = " << attribute_name << '\n';
  //
  //     // if(attribute_name == "size") {
  //     //   lrn_size = attribute_proto.i();
  //     // }
  //   }
    else {
      std::cout << "quantity of attribute = " << node_proto.attribute_size() << '\n';
      for(int i=0; i < node_proto.attribute_size(); i++) {
        const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
        std::string attribute_name = attribute_proto.name();
        std::cout << "attribute[" << i << "] = " << attribute_name << '\n';
    }
  }
}

std::unordered_map<std::string, int> get_layer_params(const onnx::NodeProto& node_proto) {
  std::string layer_type = node_proto.op_type();
  std::unordered_map<std::string, int> params;

  if(layer_type == "MaxPool") {
    int pad_h, pad_w;
    int stride_h, stride_w;
    int kernel_h, kernel_w;

    for(int i = 0; i < node_proto.attribute_size(); i++) {
      const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
      std::string attribute_name = attribute_proto.name();

      if(attribute_name == "strides") {
        stride_h = attribute_proto.ints(0);
        params.emplace("stride_h", stride_h);
        stride_w = attribute_proto.ints(1);
        params.emplace(std::make_pair("stride_w", stride_w));
      }
      else if(attribute_name == "pads") {
        pad_h = attribute_proto.ints(0);
        params.emplace(std::make_pair("pad_h", pad_h));
        pad_w = attribute_proto.ints(1);
        params.emplace(std::make_pair("pad_w", pad_w));
      }
      else if(attribute_name == "kernel_shape") {
        kernel_h = attribute_proto.ints(0);
        params.emplace(std::make_pair("kernel_h", kernel_h));
        kernel_w = attribute_proto.ints(1);
        params.emplace(std::make_pair("kernel_w", kernel_w));
      }
    }
  }
  else {
    std::cout << "quantity of attribute = " << node_proto.attribute_size() << '\n';

      for(int i = 0; i < node_proto.attribute_size(); i++) {
        const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
        std::string attribute_name = attribute_proto.name();
        if (attribute_proto.has_i()) {
          params.emplace(attribute_proto.name(), attribute_proto.i());
        } //else if (attribute_proto.has_f()) {
        //     params.emplace(attribute_proto.name(), attribute_proto.f());
        // }  // else if (attribute_proto.has_s()) {
        //   params.emplace(attribute_proto.name(), attribute_proto.s());
        // } else if (attribute_proto.has_t()) {
        //   params.emplace(attribute_proto.name(), attribute_proto.t());
        // } else if (attribute_proto.has_g()) {
        //   params.emplace(attribute_proto.name(), attribute_proto.g());
        // }
        // for (int i = 0; i < attribute_proto.floats_size(); i++) {
        //   params.emplace(attribute_proto.name(), attribute_proto.floats(i));
        // }
        for (int i = 0; i < attribute_proto.ints_size(); i++) {
          params.emplace(attribute_proto.name(), attribute_proto.ints(i));
        }
        // for (int i = 0; i < attribute_proto.strings_size(); i++) {
        //   params.emplace(attribute_proto.name(), attribute_proto.strings(i));
        // }
        // for (int i = 0; i < attribute_proto.tensors_size(); i++) {
        //   params.emplace(attribute_proto.name(), attribute_proto.tensors(i));
        // }
        // for (int i = 0; i < attribute_proto.graphs_size(); i++) {
        //   params.emplace(attribute_proto.name(), attribute_proto.graphs(i));
        // }
    }
  }
  return params;
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
      }
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
  }
  else {
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
    std::map<std::string, cv::Mat>::iterator weight;

    for(int i = 0; i < graph_proto.node_size(); i++) {
      node_proto = graph_proto.node(i);
      lp = get_lp(node_proto);
      std::cout << "1" << '\n';
      std::cout << "input size = " << node_proto.input_size() << '\n';

    //  layer_params = get_layer_params(node_proto);

    if(node_proto.input_size() > 1) {   // weights
			//       node_proto.input(1);  // = 1
      std::cout << "num node input = " << node_proto.input(1) << '\n';
      int num = std::stoi(node_proto.input(1));
      weight = weights.find(graph_proto.initializer(num -1).name());  // bug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if (weight != weights.end()) {
        lp.blobs.push_back(weight->second);
      }
		}

		if(node_proto.input_size() > 2) {  // bias
		// node_proto.input(2);  // = 2
      int num = std::stoi(node_proto.input(2));
      std::cout << "num node input = " << node_proto.input(2) << '\n';
     weight = weights.find(graph_proto.initializer(num - 1).name());  // bug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     if (weight != weights.end()) {
       lp.blobs.push_back(weight->second);
     }
		}
      // int num = std::stoi(node_proto.input(i));
      // weight = weights.find(graph_proto.initializer( num ).name());  // bug
      // if (weight != weights.end()) {
      //   lp.blobs.push_back(weight->second);
      // }
      std::cout << "2" << '\n';

      lp.set("num_output", node_proto.output_size());


    //  std::cout << "num output = " << node_proto.output_size() << '\n';
    //  lp.type = (node_proto.attribute(i).name() == "MaxPool")? "Pooling" : node_proto.attribute(i).name();
      lp.name = node_proto.op_type() + "_" + std::to_string(i);
      // if (node_proto.op_type() == "MaxPool") {
      //   lp.type = "Pooling";
      //   lp.set("pool", "MAX");
      // } else if (node_proto.op_type() == "Gemm") {
      //   lp.type = "InnerProduct";
      // } else if (node_proto.op_type() == "Conv") {
      //   lp.type = "Convolution";
      // }
      // else {
      //   lp.type = node_proto.op_type();
      // }
      net.addLayerToPrev(lp.name, lp.type, lp);
    }
  }
  return net;
}

void parse_onnx_model(const onnx::ModelProto& model_proto) {
  onnx::GraphProto graph_proto;
  onnx::NodeProto node_proto;
  if(model_proto.has_graph()) {
    std::cout << "Parsing the onnx model." << std::endl;
    graph_proto = model_proto.graph();
  }
  std::cout << "node size = " << graph_proto.node_size() << '\n';
  get_weight(graph_proto);
  for(int i = 0; i < graph_proto.node_size(); i++) {
      node_proto = graph_proto.node(i);
      std::cout << "node input = " << node_proto.input_size() <<'\n';
      print_layer_params(node_proto);
  }
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
      std::cout << argv[1] << ": file not found. Creating a new file." << '\n';
    } else if (!model_proto.ParseFromIstream(&input)) {
      std::cerr << "Failed to parse onnx model." << std::endl;
      return -1;
    }
  }

  parse_onnx_model(model_proto);
  google::protobuf::ShutdownProtobufLibrary();

  onnx::GraphProto graph_proto = model_proto.graph();
  std::unordered_map<std::string, int> params;
  cv::dnn::LayerParams lp;
  cv::dnn::Net net;
  net = create_net(model_proto);

  net.setInput(blobFromNPY(argv[2]));

  cv::Mat output = net.forward();
  std::cout << output.size << '\n';
  cv::Mat outputBlob = blobFromNPY(argv[3]);
  std::cout << outputBlob.size << '\n';
  double normL2 = cv::norm(output, outputBlob, cv::NORM_INF);
  std::cout << "norm = " << normL2 << '\n';


  // cv::dnn::LayerParams lp;
  // lp.name = "MaxPool";
  // lp.type = "Pooling";
  // lp.set("kernel_h", params["kernel_h"]);
  // lp.set("kernel_w", params["kernel_w"]);
  // lp.set("pad_h", params["pad_h"]);
  // lp.set("pad_w", params["pad_w"]);
  // lp.set("stride_h", params["stride_h"]);
  // lp.set("stride_w", params["stride_w"]);
  // lp.set("pool", "MAX");
  //
  // cv::dnn::Net net;
  // net.addLayerToPrev(lp.name, lp.type, lp);
  //
  // int dim = 4;
  // int size[dim] = {20, 3, 50, 100};
  // cv::Mat inp(dim, size, CV_32FC1, cv::Scalar(10.5));
  // net.setInput(inp);
  // cv::Mat out = net.forward();
  // std::cout << out.size << '\n';
  // print_matrix(out, size);
  // cv::Mat inputBlob = blobFromNPY("input.npy");
  // net.setInput(inputBlob);
  // cv::Mat output = net.forward();

  return 0;
}
