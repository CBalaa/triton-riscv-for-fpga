#include "support.h"
#include "model_support.h"

#ifdef HOST_TEST
#include <stdio.h>
#include <stdlib.h>
#endif

void model_progress(unsigned index) {
  /* The board implementation can supply a mailbox/UART progress hook. */
#ifndef HOST_TEST
  extern void model_board_progress(unsigned) __attribute__((weak));
  if (model_board_progress)
    model_board_progress(index);
#else
  (void)index;
#endif
}

void model_timing(unsigned index, uint64_t elapsed) {
  nr_puts("model op=");
  nr_hex32(index);
  nr_puts(" ticks=");
  nr_hex64(elapsed);
  nr_puts("\r\n");
}

int model_verify(const float *actual, const float *golden, unsigned count,
                 unsigned expected_top1, uint64_t elapsed) {
  unsigned errors = 0;
  float maximum = 0.0f;
  unsigned top[5] = {0, 0, 0, 0, 0};
  for (unsigned i = 0; i < count; ++i) {
    if (!__builtin_isfinite(actual[i]) || !__builtin_isfinite(golden[i])) {
      ++errors;
      maximum = __builtin_inff();
      continue;
    }
    float difference = actual[i] - golden[i];
    if (difference < 0.0f)
      difference = -difference;
    if (difference > maximum)
      maximum = difference;
    if (!check_close(actual[i], golden[i], 0.002f, 0.001f))
      ++errors;
  }
  for (unsigned rank = 0; rank < 5; ++rank) {
    unsigned best = count;
    for (unsigned i = 0; i < count; ++i) {
      int used = 0;
      for (unsigned j = 0; j < rank; ++j)
        used |= top[j] == i;
      if (!used && (best == count || actual[i] > actual[best]))
        best = i;
    }
    top[rank] = best;
  }
  nr_puts("model total_ticks=");
  nr_hex64(elapsed);
  nr_puts(" top5=");
  for (unsigned rank = 0; rank < 5; ++rank) {
    nr_hex32(top[rank]);
    nr_puts(rank == 4 ? "\r\n" : ",");
  }
  if (top[0] != expected_top1) {
    nr_puts("model top1 mismatch expected=");
    nr_hex32(expected_top1);
    nr_puts("\r\n");
    ++errors;
  }
#ifdef HOST_TEST
  const char *path = getenv("MODEL_LOGITS_FILE");
  if (path) {
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(actual, sizeof(float), count, file) != count) {
      fputs("cannot write model logits\n", stderr);
      if (file)
        fclose(file);
      return 2;
    }
    if (fclose(file))
      return 2;
  }
#endif
  return print_check("mobilenetv3_small_100.lamb_in1k", errors, maximum);
}

#ifdef HOST_TEST
int main(void) { return launch(); }
#endif
