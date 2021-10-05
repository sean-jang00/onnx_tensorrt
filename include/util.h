#ifndef UTIL_H_
#define UTIL_H_


//#define INPUT_WIDTH 640
//#define INPUT_HEIGHT 384



#include <iostream>
#include <string>

using namespace std;

void ImageLoad(float *data, int width, int height, cv::Mat dst) //, const std::string& imgFile)
{
  unsigned char *line = NULL;
  int offset_g = width * height;
  int offset_r = width * height * 2;
  for (int i = 0; i < height; ++i)
  {
    line = dst.ptr<unsigned char>(i);
    for (int j = 0; j < width; ++j)
    {
      data[i * width + j + offset_r] = (float)line[j * 3] /255.f;
      data[i * width + j + offset_g] = (float)line[j * 3 + 1] /255.f;
      data[i * width + j] = (float)line[j * 3 + 2] /255.f;
    }
  }
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


#endif
