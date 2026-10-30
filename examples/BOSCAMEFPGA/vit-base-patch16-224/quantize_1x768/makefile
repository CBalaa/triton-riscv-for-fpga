# This case is sensitive to the .rodata placement: with .rodata directly
# after .text.e6 the first kernel call never returns. Hold .rodata at the
# board-tested +40-byte offset and keep the shared board.ld unchanged.
BOARD_LINK_SCRIPT := $(CURDIR)/rodata-shifted.ld
include ../common.mk
include ../board.mk
