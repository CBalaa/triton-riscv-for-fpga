#ifndef ELECTRA_MODEL_BOARD_H
#define ELECTRA_MODEL_BOARD_H
#include <stdint.h>
#include "nh_e6_platform.h"
uint64_t nr_cycles(void);
uint32_t electra_board_mmio_read32(uintptr_t address);
void board_init_uart(uint32_t frequency, uint32_t baud);
unsigned electra_operator_begin(unsigned kind);
void electra_operator_end(unsigned slot, uint64_t cycles);
#endif
