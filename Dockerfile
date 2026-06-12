# =============================================================================
# KINg — multi-stage slim build (CPU-only, no CUDA)
#
# ARG EVAL controls which evaluation to compile:
#   EVAL=NNUE (default) → Open / Deep Learning category build (net embedded)
#   EVAL=HCE            → No Deep Learning category build (no net, no python)
# =============================================================================

# Build stage
FROM ubuntu:22.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
ARG EVAL=NNUE
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake make python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src
COPY third_party ./third_party
COPY tests ./tests
COPY nets ./nets
COPY tools ./tools
# NOTE: CMakeLists.txt builds a PORTABLE x86-64 baseline (-march=x86-64, no -mavx2).
# AVX2 is selected at runtime in src/nnue.cpp (init_simd_dispatch via
# __builtin_cpu_supports), so the binary runs on ANY x86-64 CPU and never SIGILLs
# if the eval hardware lacks AVX2 — critical because crash == lost game.
# JOBS controls build parallelism. Default (empty) = all cores via $(nproc) —
# what the organizer wants on a fast machine. Override with --build-arg JOBS=4
# to cap parallelism when building on a shared/busy host, without changing the
# shipped full-speed default.
ARG JOBS=
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DEVAL=${EVAL} \
    && cmake --build build --target engine -j${JOBS:-$(nproc)}

# Runtime stage (slim, CPU-only — no python, no nets, just the binary)
FROM ubuntu:22.04 AS runtime
COPY --from=build /src/build/engine /usr/local/bin/engine
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD echo "uci" | timeout 5 /usr/local/bin/engine | grep -q "uciok"
ENTRYPOINT ["/usr/local/bin/engine"]
