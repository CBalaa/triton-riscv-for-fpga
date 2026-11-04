#ifndef EFFICIENTNET_MODEL_BOARD_SUPPORT_H
#define EFFICIENTNET_MODEL_BOARD_SUPPORT_H
#include "graph_runner.h"
#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"
uint32_t efficientnet_mmio_read32(uintptr_t address);
void efficientnet_install_trap(void);
void efficientnet_trace(unsigned event);
void efficientnet_export_results(const float *logits);
void efficientnet_before(unsigned index, const char *name, const EfficientNetB3Span *spans, unsigned count);
void efficientnet_after(unsigned index, const char *name, const EfficientNetB3Span *spans, unsigned count);
extern unsigned char model_workspace_blob[];
extern float model_export_logits[];
extern uint64_t model_export_cycles[];
extern const unsigned char model_params_blob[], model_input_blob[], model_reference_blob[];
extern volatile unsigned model_returned,model_checked,model_errors,model_top1,model_call_count,model_call_inflight;
extern volatile uint32_t model_max_abs_bits;
extern volatile uint64_t model_cycles,model_kernel_cycles,model_logits_hash,model_logits_address;
extern volatile uint64_t model_call_cycles[];
extern const EfficientNetB3Hooks efficientnet_hooks;
#endif
