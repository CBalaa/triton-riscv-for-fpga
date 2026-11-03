#include "support.h"
extern void _mlir_ciface_kernel_attention_score_12x128x128(MemRef3 *, MemRef3 *,
                                                           MemRef3 *, MemRef2 *,
                                                           MemRef3 *);
int launch(void) {
  size_t offset = 0;
  float *Content = workspace(offset);
  offset = vit_align64(offset + 196608u * sizeof(*Content));
  for (size_t i = 0; i < 196608u; ++i)
    Content[i] = (float)((int)(i % 41) - 20) * 0.03125f;
  float *Position = workspace(offset);
  offset = vit_align64(offset + 196608u * sizeof(*Position));
  for (size_t i = 0; i < 196608u; ++i)
    Position[i] = (float)((int)(i % 41) - 20) * 0.03125f;
  float *Segment = workspace(offset);
  offset = vit_align64(offset + 196608u * sizeof(*Segment));
  for (size_t i = 0; i < 196608u; ++i)
    Segment[i] = (float)((int)(i % 41) - 20) * 0.03125f;
  int32_t *Mask = workspace(offset);
  offset = vit_align64(offset + 16384u * sizeof(*Mask));
  for (size_t i = 0; i < 16384u; ++i)
    Mask[i] = (int32_t)(i % 7);
  float *Out = workspace(offset);
  offset = vit_align64(offset + 196608u * sizeof(*Out));
  for (size_t i = 0; i < 196608u; ++i)
    Out[i] = -999.0f;
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      Mask[i * 128 + j] = (j >= 117 && i != j);
  MemRef3 mContent = make_3(Content, 12, 128, 128);
  MemRef3 mPosition = make_3(Position, 12, 128, 128);
  MemRef3 mSegment = make_3(Segment, 12, 128, 128);
  MemRef2 mMask = make_2(Mask, 128, 128);
  MemRef3 mOut = make_3(Out, 12, 128, 128);
  _mlir_ciface_kernel_attention_score_12x128x128(&mContent, &mPosition,
                                                 &mSegment, &mMask, &mOut);
  unsigned errors = 0;
  float maximum = 0.0f;
  for (int h = 0; h < 12; ++h)
    for (int i = 0; i < 128; ++i)
      for (int j = 0; j < 128; ++j) {
        int k = (h * 128 + i) * 128 + j;
        {
          float want = ((Content[k] + Position[k]) + Segment[k]) * 0.125f -
                       (Mask[i * 128 + j] ? 1.0e30f : 0.0f);
          float delta = __builtin_fabsf(Out[k] - want);
          errors += !check_close(Out[k], want, 1e-06f, 1e-06f);
          if (delta > maximum)
            maximum = delta;
        }
      }
  return print_check("attention_score_12x128x128", errors, maximum);
}
