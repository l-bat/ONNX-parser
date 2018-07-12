#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/dnn/all_layers.hpp>
//#include </home/liubov/work_spase/opencv/modules/dnn/test/npy_blob.cpp>
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

void print_layer_params (const onnx::NodeProto& node_proto) {
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
        //std::cout << "stride height: " << stride_h << '\n';
      //  std::string ("stride_h");
        params.emplace("stride_h", stride_h);
        stride_w = attribute_proto.ints(1);
        //std::cout << "stride width: " << stride_w << '\n';
        params.emplace(std::make_pair("stride_w", stride_w));
      }
      else if(attribute_name == "pads") {
        pad_h = attribute_proto.ints(0);
      //  std::cout << "pad height: " << pad_h << '\n';
        params.emplace(std::make_pair("pad_h", pad_h));
        pad_w = attribute_proto.ints(1);
        //std::cout << "pad width: " << pad_w << '\n';
        params.emplace(std::make_pair("pad_w", pad_w));
      }
      else if(attribute_name == "kernel_shape") {
        kernel_h = attribute_proto.ints(0);
        //std::cout << "kernel height: " << kernel_h << '\n';
        params.emplace(std::make_pair("kernel_h", kernel_h));
        kernel_w = attribute_proto.ints(1);
        //std::cout << "kernel width: " << kernel_w << '\n';
        params.emplace(std::make_pair("kernel_w", kernel_w));
      }
    }
  }
  return params;
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
      print_layer_params(node_proto);
  }
}


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

  onnx::GraphProto graph_proto = model_proto.graph();
  onnx::NodeProto node_proto;
  std::unordered_map<std::string, int> params;
  cv::dnn::LayerParams lp;
  cv::dnn::Net net;
cv::Mat inputBlob = blobFromNPY("input.npy");
  for(int i = 0; i < graph_proto.node_size(); i++) {
      node_proto = graph_proto.node(i);
      params = get_layer_params(node_proto);
      std::cout << "params contains: " << '\n';
      for (auto& x: params)
        std::cout << x.first << ": " << x.second << std::endl;

      lp.name = "MaxPool";
      lp.type = "Pooling";
      lp.set("kernel_h", params["kernel_h"]);
      lp.set("kernel_w", params["kernel_w"]);
      lp.set("pad_h", params["pad_h"]);
      lp.set("pad_w", params["pad_w"]);
      lp.set("stride_h", params["stride_h"]);
      lp.set("stride_w", params["stride_w"]);
      lp.set("pool", "MAX");
      net.addLayerToPrev(lp.name, lp.type, lp);
      net.setInput(inputBlob);
      cv::Mat output = net.forward();
  }


  //
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
  // // int dim = 4;
  // // int size[dim] = {20, 3, 50, 100};
  // // cv::Mat inp(dim, size, CV_32FC1, cv::Scalar(10.5));
  // // net.setInput(inp);
  // // cv::Mat out = net.forward();
  // //std::cout << out.size << '\n';
  // //print_matrix(out, size);
  //
  // cv::Mat inputBlob = blobFromNPY("input.npy");
  // net.setInput(inputBlob);
  // cv::Mat output = net.forward();
  std::cout << output.size << '\n';
  cv::Mat outputBlob = blobFromNPY("output.npy");
  std::cout << outputBlob.size << '\n';
  double normL2 = cv::norm(output, outputBlob, cv::NORM_INF);
  std::cout << "norm = " << normL2 << '\n';
  return 0;
}
