#ifndef MOBILENET_MODEL_SUPPORT_H
#define MOBILENET_MODEL_SUPPORT_H
#include <stdint.h>
void model_start(unsigned total);
void model_op_begin(unsigned index, const char *name, const char *case_name,
                    const char *symbol);
void model_op_end(uint64_t elapsed);
/* Optional NH runtime trap hook; reads only this core's execution state. */
void model_report_failure(void);
int model_verify(const float *actual, const float *golden, unsigned count,
                 unsigned expected_top1, uint64_t elapsed);
#endif
