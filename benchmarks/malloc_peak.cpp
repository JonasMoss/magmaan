// malloc-layer heap counter, installed via linker `--wrap`.
//
// `-Wl,--wrap=malloc` redirects every reference to `malloc` (in the harness
// objects, in libmagmaan.a, and in inlined Eigen) to `__wrap_malloc`, with the
// genuine allocator reachable as `__real_malloc`. Measuring at the malloc layer
// is necessary because Eigen's dynamic matrices allocate through
// `aligned_malloc -> std::malloc`, bypassing `operator new` entirely.
//
// `--wrap` needs no `dlsym` bootstrap (the real symbols are provided by the
// linker), so the wrappers are reentrancy-free.

#include "malloc_peak.hpp"

#include <atomic>
#include <cstddef>
#include <malloc.h>  // malloc_usable_size

extern "C" {
void* __real_malloc(std::size_t);
void __real_free(void*);
void* __real_calloc(std::size_t, std::size_t);
void* __real_realloc(void*, std::size_t);
int __real_posix_memalign(void**, std::size_t, std::size_t);
void* __real_aligned_alloc(std::size_t, std::size_t);
}

namespace {

std::atomic<std::size_t> g_live{0};
std::atomic<std::size_t> g_peak{0};

void note_add(std::size_t n) {
  const std::size_t cur =
      g_live.fetch_add(n, std::memory_order_relaxed) + n;
  std::size_t prev = g_peak.load(std::memory_order_relaxed);
  while (cur > prev && !g_peak.compare_exchange_weak(
                           prev, cur, std::memory_order_relaxed)) {
  }
}

void note_sub(std::size_t n) {
  g_live.fetch_sub(n, std::memory_order_relaxed);
}

}  // namespace

extern "C" {

void* __wrap_malloc(std::size_t n) {
  void* p = __real_malloc(n);
  if (p != nullptr) note_add(malloc_usable_size(p));
  return p;
}

void __wrap_free(void* p) {
  if (p != nullptr) note_sub(malloc_usable_size(p));
  __real_free(p);
}

void* __wrap_calloc(std::size_t nmemb, std::size_t size) {
  void* p = __real_calloc(nmemb, size);
  if (p != nullptr) note_add(malloc_usable_size(p));
  return p;
}

void* __wrap_realloc(void* p, std::size_t n) {
  const std::size_t old = (p != nullptr) ? malloc_usable_size(p) : 0;
  void* q = __real_realloc(p, n);
  if (q != nullptr) {
    note_sub(old);
    note_add(malloc_usable_size(q));
  } else if (n == 0) {
    note_sub(old);  // realloc(p, 0) freed p
  }
  return q;
}

int __wrap_posix_memalign(void** memptr, std::size_t alignment,
                          std::size_t size) {
  const int rc = __real_posix_memalign(memptr, alignment, size);
  if (rc == 0 && *memptr != nullptr) note_add(malloc_usable_size(*memptr));
  return rc;
}

void* __wrap_aligned_alloc(std::size_t alignment, std::size_t size) {
  void* p = __real_aligned_alloc(alignment, size);
  if (p != nullptr) note_add(malloc_usable_size(p));
  return p;
}

}  // extern "C"

namespace memprof {

void reset_peak() {
  g_peak.store(g_live.load(std::memory_order_relaxed),
               std::memory_order_relaxed);
}

std::size_t current_bytes() {
  return g_live.load(std::memory_order_relaxed);
}

std::size_t peak_bytes() {
  return g_peak.load(std::memory_order_relaxed);
}

}  // namespace memprof
