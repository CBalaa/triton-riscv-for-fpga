#ifndef E5_MODEL_BOARD_H
#define E5_MODEL_BOARD_H
#include "support.h"
#include "nh_e6_platform.h"
#include "bosc_ame_dma.h"
uint32_t e5_model_mmio_read32(uintptr_t address);
uint64_t e5_model_kernel_begin(unsigned case_id);
void e5_model_kernel_end(unsigned case_id, uint64_t begin);
void e5_model_checkpoint(unsigned layer, const float *hidden);
int e5_model_launch(void);
extern float e5_sentence_output[384];
extern float e5_hidden_output[128 * 384];
extern float e5_layer_outputs[13 * 128 * 384];
extern volatile uint64_t e5_model_cycles, e5_kernel_cycles, e5_kernel_calls;
extern volatile uint64_t e5_kernel_case, e5_checkpoints, e5_launch_result;
#endif
