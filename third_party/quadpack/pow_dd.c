#include "f2c.h"
#include <math.h>

/* libf2c power helper for `x ** y` with double base and exponent. f2c emits a
   call to `pow_dd` for the one fractional power in dqk15i; libf2c is not linked,
   so we provide the trivial definition. Renamed to a magmaan-private symbol at
   compile time (see cmake/QuadpackVendor.cmake). */
double pow_dd(doublereal *ap, doublereal *bp)
{
    return pow(*ap, *bp);
}
