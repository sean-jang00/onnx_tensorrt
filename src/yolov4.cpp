#include "yolov4.h"

YoloV4::YoloV4()
{
  m_output_layers = 3;
  m_width = 640;
  m_height = 384;


  m_branch_size[0] = 48 * 80 * 3;
  m_branch_size[1] = 24 * 40 * 3;
  m_branch_size[2] = 12 * 20 * 3;

  int m_total_boxes = (m_branch_size[0] + m_branch_size[0] + m_branch_size[0]) * 4;
  //48 * 80 * 3 * 4 + 24 * 40 * 3 * 4 + 12 * 20 * 3 * 4;

  anchor_box = new float[m_total_boxes];
  make_anchor_box();
}
void YoloV4::make_anchor_box()
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

YoloV4::~YoloV4()
{
  delete anchor_box;
}
