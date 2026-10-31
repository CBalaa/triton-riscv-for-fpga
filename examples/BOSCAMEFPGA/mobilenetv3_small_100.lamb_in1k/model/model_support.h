#ifndef MOBILENET_MODEL_SUPPORT_H
#define MOBILENET_MODEL_SUPPORT_H
#include <stdint.h>
void model_progress(unsigned index);
void model_timing(unsigned index, uint64_t elapsed);
int model_verify(const float *actual, const float *golden, unsigned count,
                 unsigned expected_top1, uint64_t elapsed);
#endif
