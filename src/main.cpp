
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <string>
#include <utility>
#include <vector>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "common.h"
#include "logger.h"
#include "buffers.h"



#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"


//#include <string>


#define INPUT_WIDTH 640
#define INPUT_HEIGHT 384


using namespace std;

static const char onxx_model_name[512] = "../384.onnx";
//static const char onxx_model_name[512] = "/home/suhyung/work/ROBO/g42_640_640.onnx";
//static const char onxx_model_name[512] = "/home/suhyung/work/MyONNX/ssd.pytorch/test2.onnx";

static const char file_name[256] = "/home/suhyung/Data/0917_valid_images/images/00100.png";

int m_branch_single_tensor_size = 4;
//int m_anchor[3][6] = {{12, 16, 19, 36, 40, 28}, {36, 75, 76, 55, 72, 146}, {142, 110, 192, 243, 459, 401}};
float m_anchor[3][6] = {{13.23438, 9.28906, 11.28125, 26.76562, 5.30078, 75.56250}, 
                        {32.46875, 15.81250, 25.71875, 40.09375, 60.18750, 24.75000}, 
                        {35.50000,  82.37500, 95.31250,  45.34375, 172.87500, 105.62500}};


int m_x_grid_size[3] = {80, 40, 20};
int m_y_grid_size[3] = {48, 24, 12};
int m_stride[3] = {8, 16, 32};
int m_output_layers = 3;
int m_total_boxes = 0;
float *anchor_box;
int m_single_boxes[3];
void *buffers[4]; // need fix
float *inf_raw;
float *inf_output;


#define CHECK(status)                       \
  do                                        \
  {                                         \
    auto ret = (status);                    \
    if (ret != 0)                           \
    {                                       \
      std::cout << "Cuda failure: " << ret; \
      abort();                              \
    }                                       \
  } while (0)

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
  Logger(Severity severity = Severity::kWARNING)
      : reportableSeverity(severity)
  {
  }

  void log(Severity severity, const char *msg) override
  {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity)
      return;

    switch (severity)
    {
    case Severity::kINTERNAL_ERROR:
      std::cerr << "INTERNAL_ERROR: ";
      break;
    case Severity::kERROR:
      std::cerr << "ERROR: ";
      break;
    case Severity::kWARNING:
      std::cerr << "WARNING: ";
      break;
    case Severity::kINFO:
      std::cerr << "INFO: ";
      break;
    default:
      std::cerr << "UNKNOWN: ";
      break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportableSeverity;
};

static Logger gLogger;
static int gUseDLACore{-1};

//PriorBoxParameters

struct BBox
{
  float x1, y1, x2, y2;
};
std::vector<std::vector<uint32_t>> parse_int_double_array(const std::string key, rapidjson::Document &doc)
{
  std::vector<std::vector<uint32_t>> results;
  return results;
}

std::string readBuffer(std::string const &path)
{
  string buffer;
  ifstream stream(path.c_str(), ios::binary);

  if (stream)
  {
    stream >> noskipws;
    copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
  }

  return buffer;
}
void ImageLoad(float *data, int width, int height, cv::Mat dst) //, const std::string& imgFile)
{
  unsigned char *line = NULL;
  int offset_g = INPUT_WIDTH * INPUT_HEIGHT;
  int offset_r = INPUT_WIDTH * INPUT_HEIGHT * 2;
  for (int i = 0; i < INPUT_HEIGHT; ++i)
  {
    line = dst.ptr<unsigned char>(i);
    for (int j = 0; j < INPUT_WIDTH; ++j)
    {
      data[i * INPUT_WIDTH + j + offset_r] = (float)line[j * 3] /255.f;
      data[i * INPUT_WIDTH + j + offset_g] = (float)line[j * 3 + 1] /255.f;
      data[i * INPUT_WIDTH + j] = (float)line[j * 3 + 2] /255.f;
    }
  }
}
void onnxToTRTModel(const std::string &modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    IHostMemory *&trtModelStream) // output buffer for the TensorRT model
{
  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

  // create the builder
  IBuilder *builder = createInferBuilder(gLogger);
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  //nvinfer1::INetworkDefinition *network = builder->createNetwork();
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    

  auto parser = nvonnxparser::createParser(*network, gLogger);

  //Optional - uncomment below lines to view network layer information
  //config->setPrintLayerInfo(true);
  //parser->reportParsingInfo();

  if (!parser->parseFromFile(onxx_model_name, verbosity))
  {
    string msg("failed to parse onnx file");
    gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }

  // Build the engine
  builder->setMaxBatchSize(2);
  builder->setFp16Mode(1);
  builder->setMaxWorkspaceSize(1 << 20);
  if (!builder->platformHasFastInt8())
  {
    printf("INT8 impossible in this platform \n");
  }
  else
  {
    printf("INT8 possible in this platform \n");
  }

  ICudaEngine *engine = builder->buildCudaEngine(*network);
  assert(engine);

  //string input_name = network->getInput(0)->getName();
  //std::cout << "Hello, world!" << input_name << std::endl;
  //network->getInput(0)->setDynamicRange(-tensorMap.at(input_name), tensorMap.at(input_name));

  // we can destroy the parser
  parser->destroy();

  // serialize the engine, then close everything down
  trtModelStream = engine->serialize();
  engine->destroy();
  network->destroy();
  builder->destroy();
}


       

void doInference(IExecutionContext &context, float *input_data)
{
  //char file_name[256];

  const ICudaEngine &engine = context.getEngine();
  assert(engine.getNbBindings() == 3 + 1);
  int numBindings = engine.getNbBindings();

	for (int i = 0; i < numBindings; i++)
	{		
    const char *bind_name = engine.getBindingName(i);
		const nvinfer1::Dims bind_dims = engine.getBindingDimensions(i);

    cout << i << " " << bind_name ;
    size_t data_size = sizeof(float);
    for (int j = 0; j < bind_dims.nbDims;j++)
    {
      data_size *= bind_dims.d[j];
    }
    cout << " " << data_size << endl;    ;
  }


  //sprintf(file_name, "/home/suhyung/Data/0917_valid_images/images/00100.png");


  //for (int i = 0; i < 15;i++)
  //  cout << "Pixel data " << ImgData[i*640] << endl;
  CHECK(cudaMemcpyAsync(buffers[0], input_data, INPUT_HEIGHT * INPUT_WIDTH * 3 * sizeof(float), cudaMemcpyHostToDevice, NULL));
  context.enqueue(1, buffers, NULL, nullptr);

  int offset[3];
  offset[0] = 3 * 48 * 80 * 19;
  offset[1] = 3 * 24 * 40 * 19;
  offset[2] = 3 * 12 * 20 * 19;

  CHECK(cudaMemcpyAsync(inf_raw, buffers[1],  3 * 48 * 80 * 19 * sizeof(float), cudaMemcpyDeviceToHost, NULL));
  CHECK(cudaMemcpyAsync(inf_raw+offset[0], buffers[2],  3 * 24 * 40 * 19 * sizeof(float), cudaMemcpyDeviceToHost, NULL));
  CHECK(cudaMemcpyAsync(inf_raw+offset[0]+offset[1], buffers[3],  3 * 12 * 20 * 19 * sizeof(float), cudaMemcpyDeviceToHost, NULL));
  //for (int i = 0; i < 19;i++)
  //  cout << "inf     " << inf_raw[i] << endl;

  cout << "inference completed" << endl;

}



void calculate_size()
{
  for (int i = 0; i < m_output_layers; i++)
  {
    uint32_t num_boxes = m_x_grid_size[i] * m_y_grid_size[i] * 3;
    m_total_boxes += num_boxes;
    m_total_boxes *= 4;
    m_single_boxes[i] = num_boxes;
  }
  m_total_boxes = 48 * 80 * 3 * 4 + 24 * 40 * 3 * 4 + 12 * 20 * 3 * 4;
}

void make_anchor_box()
{
  uint32_t index = 0;
  for (int o = 0; o < m_output_layers; ++o)         // 3
  {
    for (int a = 0; a < 6; a+=2)   // 3
    {
      for (int y = 0; y < m_y_grid_size[o]; ++y)      // 48 -> 24 -> 12
      {
        for (int x = 0; x < m_x_grid_size[o]; ++x)    // 80 -> 40 -> 20
        {
          anchor_box[index++] = x;                  // X
          anchor_box[index++] = y;                  // Y
          anchor_box[index++] = m_anchor[o][a];     // W
          anchor_box[index++] = m_anchor[o][a+1];   // H
        }
      }
    }
  }
}
void decode_wholebox(float *output, float *input, float *anchor, int box_count, float stride, int tensor_size)
{

  for (int i = 0; i < box_count; ++i)
  {
    input[tensor_size * i + 0] = 1.0f / (1 + exp(input[tensor_size * i + 0] * -1));  
    input[tensor_size * i + 1] = 1.0f / (1 + exp(input[tensor_size * i + 1] * -1));  
    input[tensor_size * i + 2] = 1.0f / (1 + exp(input[tensor_size * i + 2] * -1));  
    input[tensor_size * i + 3] = 1.0f / (1 + exp(input[tensor_size * i + 3] * -1));  

    float center_x = (input[tensor_size * i + 0] * 2 - 0.5 + anchor[4 * i + 0]) * stride;
    float center_y = (input[tensor_size * i + 1] * 2 - 0.5 + anchor[4 * i + 1]) * stride;
    float width = (input[tensor_size * i + 2] * 2) * (input[tensor_size * i + 2] * 2) * anchor[4 * i + 2];
    float height = (input[tensor_size * i + 3] * 2) * (input[tensor_size * i + 3] * 2) * anchor[4 * i + 3];

    output[tensor_size * i + 0] = (center_x - width / 2.) / INPUT_WIDTH;
    output[tensor_size * i + 1] = (center_y - height / 2.) / INPUT_HEIGHT;
    output[tensor_size * i + 2] = (center_x + width / 2.) / INPUT_WIDTH;
    output[tensor_size * i + 3] = (center_y + height / 2.) / INPUT_HEIGHT;


    /*float cx = pred_boxes[i][0]; 
		float cy = pred_boxes[i][1]; 

		pred_boxes[i][0] = cx - pred_boxes[i][2] * 0.5;
		pred_boxes[i][1] = cy - pred_boxes[i][3] * 0.5;
		pred_boxes[i][2] = cx + pred_boxes[i][2] * 0.5;
		pred_boxes[i][3] = cy + pred_boxes[i][3] * 0.5;*/


  }
}

void decode_confidence(float *output, float *input, int box_count, int tensor_size, float *anchor, int stride, int offset)
{

  for (int i = 0; i < box_count; i++)
  {
    output[tensor_size * i + 0] = (input[tensor_size * i + 0] * 2 - 0.5 + anchor[4 * i + 0 + offset]) * stride;
    output[tensor_size * i + 1] = (input[tensor_size * i + 1] * 2 - 0.5 + anchor[4 * i + 1 + offset]) * stride;
    output[tensor_size * i + 2] = (input[tensor_size * i + 2] * 2) * (input[tensor_size * i + 2] * 2) * anchor[4 * i + 2 + offset];
    output[tensor_size * i + 3] = (input[tensor_size * i + 3] * 2) * (input[tensor_size * i + 3] * 2) * anchor[4 * i + 3 + offset];

    for (int j = 4; j < 19;j++)
      output[tensor_size * i + j] = input[tensor_size * i + j];
  }
 
}
void do_sigmoid(float *output, float *input)
{
  for (int i = 0; i < (80*48+40*24+20*12)*(19*3);i++)
  {
    output[i] = 1.0f / (1 + exp(input[i] * -1));
    if(i%19==4 && output[i%19+4]>0.4)
    {
      cout << "detected " << output[i % 19 + 4] << endl;
    }
  }
    
}
struct DetectedObject
{
  float x1;
  float x2;
  float y1;
  float y2;
  int cls;
  float conf;
};
cv::Mat od_label_image[14];
void loadLabelImages() {
    string label_od_class[14] = { 
        "person", "car", "pillar", "desk", "conf_booth", "street_lamp", 
"outdoor_unit", "pattern_panel", "flag", "manhole", "person_hand"
"chair", "sofa", "downspout" };
    for (int i = 0; i < 14; i++) {
        string image_path = "../label/"+label_od_class[i] + ".png";
        od_label_image[i] = cv::imread(image_path, cv::IMREAD_COLOR);
    }
}

int main(int argc, char **argv)
{
  float *ImgData = new float[3 * INPUT_HEIGHT * INPUT_WIDTH];
  cv::Mat cv_image;

  cv::VideoCapture cap(0); //카메라 생성
  if (!cap.isOpened())
  {
    cout << "Can't open the CAM" << endl;    ;
    return 1;
  }

 /* cv::Mat img;
	while (1)
	{
		cap >> img;
		cv::imshow("camera img", img);
		if (cv::waitKey(1) == 27)
			break;
	}*/



  char cache_path[512] = "serialized_engine.cache";
  calculate_size();
  loadLabelImages();
  anchor_box = new float[m_total_boxes];
  make_anchor_box();
  

  IHostMemory *trtModelStream{nullptr};
  ICudaEngine *engine;
  std::ifstream cache(cache_path);
  IRuntime *runtime = createInferRuntime(gLogger);
  runtime = createInferRuntime(gLogger);

  if (!cache)
  {
    cout << "make a cache" << endl;
    onnxToTRTModel(onxx_model_name, 1, trtModelStream);
    std::ofstream ofs(cache_path, std::ios::out | std::ios::binary);
    ofs.write((char *)(trtModelStream->data()), trtModelStream->size());
    ofs.close();
    engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    }
    else
    {
      string buffer = readBuffer(cache_path);
      engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
    }
    printf("convert completed\n");

    CHECK(cudaMalloc(&buffers[0], INPUT_WIDTH * INPUT_HEIGHT * 3 * sizeof(float)));

    CHECK(cudaMalloc(&buffers[1], 3 * 48 * 80 * 19 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[2], 3 * 24 * 40 * 19 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[3], 3 * 12 * 20 * 19 * sizeof(float)));

    int output_size = sizeof(float) * 3 * 48 * 80 * 19;
    output_size += sizeof(float) * 3 * 24 * 40 * 19;
    output_size += sizeof(float) * 3 * 12 * 20 * 19;

    inf_output = (float *)malloc(output_size*sizeof(float));
    memset(inf_output, 0x00, output_size * sizeof(float));
    

    inf_raw = (float *)malloc(output_size*sizeof(float));
    memset(inf_raw, 0x00, output_size * sizeof(float));
    IExecutionContext *context = engine->createExecutionContext();
    
    while(1)
    {
    
    cap >> cv_image;
    


//    cv::Mat cv_image = cv::imread(file_name, cv::IMREAD_COLOR);
    cv::Mat dst;
    printf("input image original resolution %d %d \n", cv_image.cols, cv_image.rows);
    if (cv_image.cols == 0)
    {
    // continue;
    cout << "check the input image " << endl;
    }

    cv::resize(cv_image, dst, cv::Size(INPUT_WIDTH, 360), (0.0), (0.0), cv::INTER_LINEAR);
    cv::copyMakeBorder(dst, dst, 12, 12, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114) );
    cout << "imput resolution " << dst.cols  << "x" << dst.rows << endl;

    ImageLoad(ImgData, INPUT_WIDTH, INPUT_HEIGHT, dst);

    doInference(*context, ImgData);


    do_sigmoid(inf_raw, inf_raw);
    uint32_t offset = 0;
    uint32_t anchor_offset = 0;
    decode_confidence(inf_output, inf_raw, 48 * 80 * 3, 19, anchor_box, m_stride[0], 0);
    decode_confidence(inf_output+ 48*80*3*19, inf_raw + 48*80*3*19, 24 * 40 * 3, 19, anchor_box, m_stride[1], 48 * 80 * 3 * 4);
    decode_confidence(inf_output+ 48*80*3*19+24*40*3*19, inf_raw+ 48*80*3*19+24*40*3*19 , 12 * 20 * 3, 19, anchor_box, m_stride[2],(48*80+24*40)*12 );
    vector<DetectedObject> array_obj;
    for (int i = 0; i < output_size; i += 19)
    {
        if(inf_output[i+4]>0.1)
        {
          float x1 = (inf_output[i+0] - inf_output[i+2] / 2.) / INPUT_WIDTH;
          float y1 = (inf_output[i+1] - inf_output[i+3] / 2.) / INPUT_HEIGHT;
          float x2 = (inf_output[i+0] + inf_output[i+2] / 2.) / INPUT_WIDTH;
          float y2 = (inf_output[i+1] + inf_output[i+3] / 2.) / INPUT_HEIGHT;
          DetectedObject obj;
          obj.x1 = x1;
          obj.x2 = x2;
          obj.y1 = y1;
          obj.y2 = y2;

          float max_conf=0;
          int max_cls=5;
          for (int j = 5; j < 19; j++)
          {
            if(max_conf<inf_output[i + j])
            {
              max_cls = j;
              max_conf = inf_output[i + j];
            }
            
          }
          obj.cls = max_cls;
          obj.conf = inf_output[i + 4]*max_conf;
          array_obj.push_back(obj);
        }
    }

    vector<int> indices;
    vector<cv::Rect> boxes;
    vector<float> confidences;
    vector<int> cls_id;
    std::vector<cv::Rect> srcRects;

    for (int i = 0; i < array_obj.size(); i++)
    {
      confidences.push_back(array_obj[i].conf);
      cls_id.push_back(array_obj[i].cls);
      boxes.push_back(cv::Rect(cv::Point(array_obj[i].x1 * 1920, array_obj[i].y1 * 1080), cv::Point(array_obj[i].x2 * 1920, array_obj[i].y2 * 1080)));
    }
    cv::dnn::NMSBoxes(boxes, confidences, 0.4, 0.5, indices);
    for (int i = 0; i < indices.size();i++)
    {
      int idx = indices[i];
      srcRects.push_back(boxes[idx]);
    }

    //cv::Mat cv_image = cv::imread(file_name, cv::IMREAD_COLOR);
    for (int i = 0; i < srcRects.size();i++)
    {
       /* int idx = indices[i];
        int cls_num = cls_id[idx] - 5;
        cv::Rect outRect;
        outRect.x = srcRects[i].tl().x;
        outRect.y = srcRects[i].tl().y;
        outRect.width = min(srcRects[i].br().x - srcRects[i].tl().x, 1920 - outRect.x);
        outRect.height = min(srcRects[i].br().y - srcRects[i].tl().y, 1080 - outRect.y);*/
        cv::rectangle(cv_image, srcRects[i], cv::Scalar(255, 0, 0), 5);
      /*  if (0)//od_label_image[cls_num].cols > 0 && od_label_image[cls_num].rows > 0)

        {
          cv::Rect rect_od_label(outRect.x, outRect.y - 27,
                                  od_label_image[cls_num].cols, od_label_image[cls_num].rows);
          rect_od_label.x = min(rect_od_label.x, 1920 - od_label_image[cls_num].cols);
          rect_od_label.y = min(rect_od_label.y, 1080 - od_label_image[cls_num].rows);
          rect_od_label.x = max(rect_od_label.x, 0);
          rect_od_label.y = max(rect_od_label.y, 0);
  //        od_label_image[cls_num].copyTo(cv_image(rect_od_label));
        }*/
    }

    
      

      cv::imshow("after", cv_image);
      if (cv::waitKey(1) == 27)
			break;
    
    }
   cv::waitKey(-1);


    cudaEvent_t m_start;
    cudaEvent_t m_stop;
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);

    for (int i = 0; i < 4; i++)
    {
      CHECK(cudaFree(buffers[i]));
    }
    free(inf_output);
    free(inf_raw);
    delete (anchor_box);
    delete[] ImgData;

    return 1;
}
