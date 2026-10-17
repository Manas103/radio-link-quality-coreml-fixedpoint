/* Hand-written integer-only forward pass for the link-degradation CNN.
 *
 * Every conv and the linear layer accumulate in int32, requantize with a
 * fixed-point (multiplier, shift) pair using only int64 multiply and integer
 * right-shift (the TFLite-style scheme), and clip to int8 after each ReLU.
 * The only floating point in this file is the very last step: dequantizing
 * the final int32 accumulator with OUTPUT_SCALE and applying sigmoid, which
 * is disclosed in the README as the one place float re-enters.
 *
 * Usage:
 *   forward.exe parity   data/parity_windows.f32 data/parity_reference_probs.f32
 *   forward.exe timing   data/parity_windows.f32 <n_iters>
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#include "weights.h"

static int32_t requantize(int64_t acc, int32_t mult, int32_t shift) {
    int64_t prod = acc * (int64_t)mult;
    int64_t rounding = (shift > 0) ? (1LL << (shift - 1)) : 0;
    return (int32_t)((prod + rounding) >> shift);
}

static int8_t clip_i8(int32_t v) {
    if (v < -128) return -128;
    if (v > 127) return 127;
    return (int8_t)v;
}

#define MAX_PAD 4
#define MAX_CIN 64

/* 1D convolution, "same" padding (pad = (k-1)/2), stride 1, ReLU + requant
 * fused in. in: [Cin][WINDOW] int8. w: [Cout][Cin][K] int8. b, mult, shift:
 * [Cout] (per-output-channel quantization). out: [Cout][WINDOW] int8.
 *
 * in is zero-padded into a scratch buffer once per call (attempt 2 on the
 * latency claim, see README Findings) so the inner (ci, kk) loop below is a
 * branch-free contiguous scan instead of bounds-checking every tap; that
 * change alone is what brought the measured median down from ~730us. */
static void conv1d_relu(const int8_t *in, int cin, const int8_t *w, const int32_t *b,
                         int cout, int k, const int32_t *mult, const int32_t *shift, int8_t *out) {
    int pad = (k - 1) / 2;
    static int8_t padded[MAX_CIN][WINDOW + 2 * MAX_PAD];
    for (int ci = 0; ci < cin; ci++) {
        memset(padded[ci], 0, sizeof(padded[ci]));
        memcpy(padded[ci] + pad, in + (size_t)ci * WINDOW, (size_t)WINDOW);
    }
    /* acc fits int32 with headroom: at most WINDOW*MAX_CIN*k terms of
     * +-127*+-127 << 2^31 (attempt 3 on the latency claim, see README
     * Findings: int32 instead of int64 accumulation let the compiler keep
     * the inner loop in plain scalar registers instead of 64-bit pairs). */
    for (int co = 0; co < cout; co++) {
        const int8_t *__restrict w_co = w + (size_t)co * cin * k;
        for (int t = 0; t < WINDOW; t++) {
            int32_t acc = b[co];
            for (int ci = 0; ci < cin; ci++) {
                const int8_t *__restrict in_row = padded[ci] + t;
                const int8_t *__restrict w_row = w_co + (size_t)ci * k;
                for (int kk = 0; kk < k; kk++) {
                    acc += (int32_t)in_row[kk] * (int32_t)w_row[kk];
                }
            }
            int32_t req = requantize(acc, mult[co], shift[co]);
            if (req < 0) req = 0; /* ReLU */
            out[(size_t)co * WINDOW + t] = clip_i8(req);
        }
    }
}

static void global_avg_pool(const int8_t *in, int ch, int8_t *out) {
    for (int c = 0; c < ch; c++) {
        int64_t sum = 0;
        for (int t = 0; t < WINDOW; t++) sum += in[(size_t)c * WINDOW + t];
        int32_t avg = (int32_t)(sum >= 0 ? (sum + WINDOW / 2) / WINDOW : -((-sum + WINDOW / 2) / WINDOW));
        out[c] = clip_i8(avg);
    }
}

/* Returns the raw int32 accumulator (not requantized): caller dequantizes
 * with OUTPUT_SCALE because this is the network's final layer. */
static int64_t fc_forward(const int8_t *in, int cin, const int8_t *w, const int32_t *b) {
    int64_t acc = b[0];
    for (int ci = 0; ci < cin; ci++) acc += (int64_t)in[ci] * (int64_t)w[ci];
    return acc;
}

/* Quantizes a float window [IN_CH][WINDOW] to int8 using INPUT_SCALE. */
static void quantize_input(const float *x, int8_t *out) {
    for (int c = 0; c < IN_CH; c++) {
        for (int t = 0; t < WINDOW; t++) {
            float v = x[(size_t)c * WINDOW + t] / INPUT_SCALE;
            long r = lroundf(v);
            if (r < -128) r = -128;
            if (r > 127) r = 127;
            out[(size_t)c * WINDOW + t] = (int8_t)r;
        }
    }
}

static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

/* Full forward pass on one window; returns the sigmoid probability. */
static double forward_one(const float *x_f32) {
    static int8_t qin[IN_CH * WINDOW];
    static int8_t a1[C1 * WINDOW];
    static int8_t a2[C2 * WINDOW];
    static int8_t a3[C3 * WINDOW];
    static int8_t pooled[C3];

    quantize_input(x_f32, qin);
    conv1d_relu(qin, IN_CH, CONV1_W, CONV1_B, C1, K1, CONV1_MULT, CONV1_SHIFT, a1);
    conv1d_relu(a1, C1, CONV2_W, CONV2_B, C2, K2, CONV2_MULT, CONV2_SHIFT, a2);
    conv1d_relu(a2, C2, CONV3_W, CONV3_B, C3, K3, CONV3_MULT, CONV3_SHIFT, a3);
    global_avg_pool(a3, C3, pooled);
    int64_t acc = fc_forward(pooled, C3, FC_W, FC_B);
    double logit = (double)acc * (double)OUTPUT_SCALE;
    return sigmoid(logit);
}

static float *load_f32(const char *path, long *n_floats) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *buf = (float *)malloc((size_t)bytes);
    size_t got = fread(buf, 1, (size_t)bytes, f);
    if ((long)got != bytes) { fprintf(stderr, "short read on %s\n", path); exit(1); }
    fclose(f);
    *n_floats = bytes / (long)sizeof(float);
    return buf;
}

static int run_parity(const char *windows_path, const char *ref_probs_path) {
    long n_floats, n_ref;
    float *windows = load_f32(windows_path, &n_floats);
    float *ref = load_f32(ref_probs_path, &n_ref);
    long window_floats = IN_CH * WINDOW;
    long n_windows = n_floats / window_floats;
    if (n_windows != n_ref) {
        fprintf(stderr, "window/ref count mismatch: %ld vs %ld\n", n_windows, n_ref);
        return 1;
    }
    double max_abs_diff = 0.0, sum_abs_diff = 0.0;
    long argmax = -1, n_within_1e2 = 0, n_within_1e4 = 0;
    FILE *out = fopen("docs/parity_c_probs.f32", "wb");
    for (long i = 0; i < n_windows; i++) {
        const float *x = windows + i * window_floats;
        double p = forward_one(x);
        float pf = (float)p;
        fwrite(&pf, sizeof(float), 1, out);
        double diff = fabs(p - (double)ref[i]);
        sum_abs_diff += diff;
        if (diff <= 1e-2) n_within_1e2++;
        if (diff <= 1e-4) n_within_1e4++;
        if (diff > max_abs_diff) { max_abs_diff = diff; argmax = i; }
    }
    fclose(out);
    printf("n_windows=%ld max_abs_diff=%.9g at index=%ld mean_abs_diff=%.9g within_1e-2=%ld within_1e-4=%ld\n",
           n_windows, max_abs_diff, argmax, sum_abs_diff / n_windows, n_within_1e2, n_within_1e4);
    free(windows);
    free(ref);
    return 0;
}

#ifdef _WIN32
static double qpc_freq_hz(void) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return (double)f.QuadPart;
}
#endif

static int run_timing(const char *windows_path, int n_iters) {
    long n_floats;
    float *windows = load_f32(windows_path, &n_floats);
    long window_floats = IN_CH * WINDOW;
    long n_windows = n_floats / window_floats;

#ifdef _WIN32
    double freq = qpc_freq_hz();
#endif

    double *samples = (double *)malloc(sizeof(double) * (size_t)n_iters);
    volatile double sink = 0.0;
    for (int i = 0; i < n_iters; i++) {
        const float *x = windows + (i % n_windows) * window_floats;
        double us;
#ifdef _WIN32
        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        double p = forward_one(x);
        QueryPerformanceCounter(&t1);
        sink += p;
        us = (double)(t1.QuadPart - t0.QuadPart) * 1e6 / freq;
#else
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        double p = forward_one(x);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        sink += p;
        us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
#endif
        samples[i] = us;
    }
    /* median */
    for (int i = 1; i < n_iters; i++) {
        double key = samples[i];
        int j = i - 1;
        while (j >= 0 && samples[j] > key) { samples[j + 1] = samples[j]; j--; }
        samples[j + 1] = key;
    }
    double median = samples[n_iters / 2];
    double p10 = samples[n_iters / 10];
    double p90 = samples[(n_iters * 9) / 10];
    printf("n_iters=%d median_us=%.3f p10_us=%.3f p90_us=%.3f sink=%f\n", n_iters, median, p10, p90, sink);
    free(samples);
    free(windows);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: forward.exe parity|timing ...\n");
        return 1;
    }
    if (strcmp(argv[1], "parity") == 0) {
        if (argc < 4) { fprintf(stderr, "parity needs windows_path ref_probs_path\n"); return 1; }
        return run_parity(argv[2], argv[3]);
    } else if (strcmp(argv[1], "timing") == 0) {
        if (argc < 4) { fprintf(stderr, "timing needs windows_path n_iters\n"); return 1; }
        return run_timing(argv[2], atoi(argv[3]));
    }
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 1;
}
