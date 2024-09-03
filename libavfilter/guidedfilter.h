#ifndef GUIDEDFILTER_H

#define GUIDEDFILTER_H

#include <stdint.h>


int guidedfilter(const uint8_t *I, const uint8_t *p, uint8_t *q, size_t r, size_t n, size_t m, size_t ld, float eps);


#endif
