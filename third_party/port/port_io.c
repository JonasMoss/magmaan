/* port_io.c — no-op replacements for libf2c's libI77 Fortran-I/O entry points.
 *
 * PORT calls libf2c's `s_wsfe` / `e_wsfe` / `do_fio` / `do_lio` only to print
 * iteration traces (`ditsum_`), parameter-check error messages (`dparck_`),
 * and machine-constant out-of-range diagnostics (`d1mach_` / `i1mach_`).
 * magmaan never wants these prints — verbosity, if needed, is recovered
 * post-fit from the IV/V structured solver-state arrays, which capture the
 * same information PORT would otherwise print.
 *
 * Providing these stubs lets us avoid linking libI77 entirely (~500 KB of
 * Fortran formatted-I/O state machine). The PORT algorithmic code is
 * unchanged; only its print sites become silent.
 *
 * Signatures follow libf2c's libI77 conventions (return 0 on success). The
 * cilist / icilist / olist / inlist / cllist structs are defined in
 * cport/f2c.h alongside the AMPL/ASL vendored sources.
 */

/* stdio + stdlib must come BEFORE f2c.h: f2c.h defines `abs` as a macro,
 * which collides with stdlib.h's `int abs(int)` declaration. Including the
 * system headers first lets f2c.h's macro shadow them without rewriting the
 * declarations themselves. */
#include <stdio.h>
#include <stdlib.h>

#include "f2c.h"

/* Sequential formatted external write — start. PORT call shape:
 *   io___11 = { unit=0, err=0, ciunit=0, cifmt=fmt_30, cirec=0 };
 *   s_wsfe(&io___11);
 *   do_fio(&c__1, ...);
 *   e_wsfe(); */
integer s_wsfe(cilist *a)
{
    (void)a;
    return 0;
}

integer e_wsfe(void)
{
    return 0;
}

/* Emit one formatted I/O item. `number` items of `len` bytes from `ptr`. */
integer do_fio(integer *number, char *ptr, ftnlen len)
{
    (void)number;
    (void)ptr;
    (void)len;
    return 0;
}

/* List-directed I/O (used by d1mach / i1mach error paths). */
integer do_lio(integer *type, integer *number, char *ptr, ftnlen len)
{
    (void)type;
    (void)number;
    (void)ptr;
    (void)len;
    return 0;
}

/* List-directed sequential external write. PORT's d1mach.c uses these for
 * the "machine constant out of range" diagnostic before falling through to
 * s_stop. We silence the message; if s_stop fires the caller will see a
 * clear abort instead. */
integer s_wsle(cilist *a)
{
    (void)a;
    return 0;
}

integer e_wsle(void)
{
    return 0;
}

/* Fortran STOP. Only reachable from d1mach / i1mach if a machine-constant
 * request is malformed (PORT's own internal logic; not driven by user
 * input). In that scenario PORT cannot return a sensible value, so silent
 * no-op would corrupt downstream arithmetic. Abort with a diagnostic
 * instead — visible in any test failure trace.
 *
 * Signature: `s_stop(const char *msg, ftnlen msg_len)`; the message is
 * NOT NUL-terminated, only `msg_len` characters wide. */
void s_stop(char *msg, ftnlen len)
{
    if (msg != NULL && len > 0) {
        fprintf(stderr, "magmaan: vendored PORT routine invoked s_stop(\"%.*s\"); "
                        "this is a PORT internal abort (typically a malformed "
                        "machine-constant request in d1mach/i1mach) and is not "
                        "recoverable.\n", (int)len, msg);
    } else {
        fputs("magmaan: vendored PORT routine invoked s_stop(); PORT internal "
              "abort, not recoverable.\n", stderr);
    }
    abort();
}
