#ifndef YOLOV4_H
#define YOLOV4_H

#include <string.h>
#include <iostream>

enum
{
  CAM,
  IMAGE,
  VIDEO
};

class YoloV4
{
public:
  YoloV4();
  ~YoloV4();

  void make_anchor_box();
  float *anchor_box;
  int m_output_layers;
  int m_anchor_size = 19; // x,y,w,h,objectness, class(14)
  float m_anchor[3][6] = {{13.23438, 9.28906, 11.28125, 26.76562, 5.30078, 75.56250},
                          {32.46875, 15.81250, 25.71875, 40.09375, 60.18750, 24.75000},
                          {35.50000, 82.37500, 95.31250, 45.34375, 172.87500, 105.62500}};

  int m_x_grid_size[3] = {80, 40, 20};
  int m_y_grid_size[3] = {48, 24, 12};
  int m_stride[3] = {8, 16, 32};

  int m_branch_size[3];

  int m_width;
  int m_height;
};

#endif
