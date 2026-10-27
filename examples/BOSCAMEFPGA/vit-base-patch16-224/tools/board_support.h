#ifndef VIT_BOARD_SUPPORT_H
#define VIT_BOARD_SUPPORT_H

#include "support.h"
#include "nh_e6_platform.h"
#include "nr_runtime.h"

unsigned vit_board_kernel_begin(void);
void vit_board_kernel_end(unsigned slot, uint64_t cycles);
uint32_t vit_board_mmio_read32(uintptr_t address);
float fabsf(float value);

#endif
