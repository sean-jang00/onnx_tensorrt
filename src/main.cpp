
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
#include "yolov4.h"
#include "util.h"




using namespace std;


static const char onxx_model_name[512] = "../384_sigmoid.onnx";
static const char file_name[256] = "/home/suhyung/Data/0917_valid_images/images/00100.png";

int m_branch_single_tensor_size = 4;
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


       

void doInference(IExecutionContext &context)//, float *input_data, int input_width, int inpupt_height)
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

  context.enqueue(1, buffers, NULL, nullptr);


  cout << "inference completed" << endl;

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
  int origin_width;
  int origin_height;
  YoloV4 yolov4;
  yolov4.make_anchor_box();
  float *ImgData = new float[3 * yolov4.m_width * yolov4.m_height];
  cv::Mat cv_image;

  cv::VideoCapture cap(0); //카메라 생성
  if (!cap.isOpened())
  {
    cout << "Can't open the CAM" << endl;    ;
    //return 1;
  }
  char cache_path[512] = "serialized_engine.cache";
  loadLabelImages();
  

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

    CHECK(cudaMalloc(&buffers[0], yolov4.m_width * yolov4.m_height * 3 * sizeof(float)));

    CHECK(cudaMalloc(&buffers[1], yolov4.m_branch_size[0] * yolov4.m_anchor_size * sizeof(float)));
    CHECK(cudaMalloc(&buffers[2], yolov4.m_branch_size[1] * yolov4.m_anchor_size  * sizeof(float)));
    CHECK(cudaMalloc(&buffers[3], yolov4.m_branch_size[2] * yolov4.m_anchor_size * sizeof(float)));

    int output_size = (yolov4.m_branch_size[0] + yolov4.m_branch_size[0] + yolov4.m_branch_size[0]) * yolov4.m_anchor_size * sizeof(float);


    inf_output = (float *)malloc(output_size*sizeof(float));
    memset(inf_output, 0x00, output_size * sizeof(float));
  
    inf_raw = (float *)malloc(output_size*sizeof(float));
    memset(inf_raw, 0x00, output_size * sizeof(float));
    IExecutionContext *context = engine->createExecutionContext();
    
    //while(1)
    //{
    
    //cap >> cv_image;
    cv_image = cv::imread(file_name, cv::IMREAD_COLOR);
    cv::Mat dst;

    origin_width = cv_image.cols;
    origin_height = cv_image.rows;
    printf("input image original resolution %d %d \n", cv_image.cols, cv_image.rows);
    if (cv_image.cols == 0)
    {
      cout << "check the input image " << endl;
    }

    cv::resize(cv_image, dst, cv::Size(yolov4.m_width, 360), (0.0), (0.0), cv::INTER_LINEAR);
    cv::copyMakeBorder(dst, dst, 12, 12, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114) );  // zero padding
    cout << "imput resolution " << dst.cols  << "x" << dst.rows << endl;

    ImageLoad(ImgData, yolov4.m_width , yolov4.m_height, dst);


    CHECK(cudaMemcpyAsync(buffers[0], ImgData, yolov4.m_width* yolov4.m_height * 3 * sizeof(float), cudaMemcpyHostToDevice, NULL));


    doInference(*context);


    int dst_offset = 0;
    for (int i = 0; i < 3;i++)
    {
      CHECK(cudaMemcpyAsync(inf_raw+dst_offset, buffers[i+1], yolov4.m_branch_size[i] * yolov4.m_anchor_size * sizeof(float), cudaMemcpyDeviceToHost, NULL));
      dst_offset += yolov4.m_branch_size[i] * yolov4.m_anchor_size;
    }


    //do_sigmoid(inf_raw, inf_raw);
    uint32_t offset = 0;
    uint32_t anchor_offset = 0;


    decode_confidence(inf_output, inf_raw, 48 * 80 * 3, 19, yolov4.anchor_box, m_stride[0], 0);
    decode_confidence(inf_output+ 48*80*3*19, inf_raw + 48*80*3*19, 24 * 40 * 3, 19, yolov4.anchor_box, m_stride[1], 48 * 80 * 3 * 4);
    decode_confidence(inf_output+ 48*80*3*19+24*40*3*19, inf_raw+ 48*80*3*19+24*40*3*19 , 12 * 20 * 3, 19, yolov4.anchor_box, m_stride[2],(48*80+24*40)*12 );
    vector<DetectedObject> array_obj;
    for (int i = 0; i < output_size; i += 19)
    {
        if(inf_output[i+4]>0.1)
        {
          float x1 = (inf_output[i+0] - inf_output[i+2] / 2.) / yolov4.m_width;
          float y1 = (inf_output[i+1] - inf_output[i+3] / 2.) / yolov4.m_height;
          float x2 = (inf_output[i+0] + inf_output[i+2] / 2.) / yolov4.m_width;
          float y2 = (inf_output[i+1] + inf_output[i+3] / 2.) / yolov4.m_height;
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
      boxes.push_back(cv::Rect(cv::Point(array_obj[i].x1 * origin_width, array_obj[i].y1 * origin_height), cv::Point(array_obj[i].x2 * origin_width, array_obj[i].y2 * origin_height)));
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
        outRect.width = min(srcRects[i].br().x - srcRects[i].tl().x, origin_width - outRect.x);
        outRect.height = min(srcRects[i].br().y - srcRects[i].tl().y, origin_height - outRect.y);*/
        cv::rectangle(cv_image, srcRects[i], cv::Scalar(255, 0, 0), 5);
      /*  if (0)//od_label_image[cls_num].cols > 0 && od_label_image[cls_num].rows > 0)

        {
          cv::Rect rect_od_label(outRect.x, outRect.y - 27,
                                  od_label_image[cls_num].cols, od_label_image[cls_num].rows);
          rect_od_label.x = min(rect_od_label.x, origin_width - od_label_image[cls_num].cols);
          rect_od_label.y = min(rect_od_label.y, origin_height - od_label_image[cls_num].rows);
          rect_od_label.x = max(rect_od_label.x, 0);
          rect_od_label.y = max(rect_od_label.y, 0);
  //        od_label_image[cls_num].copyTo(cv_image(rect_od_label));
        }*/
    }
    cv::Mat cv_vis;
    cv::resize(cv_image, cv_vis, cv::Size(origin_width/2, origin_height/2), (0.0), (0.0), cv::INTER_LINEAR);

    cv::imshow("after", cv_vis);
    //if (cv::waitKey(1) == 27)
    //break;
    //}
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
    delete[] ImgData;

    return 1;
}
