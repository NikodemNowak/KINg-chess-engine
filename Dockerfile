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
COPY trainer/nnue_samples.txt ./trainer/nnue_samples.txt
# NOTE: CMakeLists.txt builds with -mavx2 -mpopcnt (see CMAKE_CXX_FLAGS_RELEASE).
# Exact target arch will be confirmed when the organizer publishes the eval hardware
# (~07.06); a later task may switch to a portable baseline or add runtime dispatch.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DEVAL=${EVAL} \
    && cmake --build build --target engine -j

# Runtime stage (slim, CPU-only — no python, no nets, just the binary)
FROM ubuntu:22.04 AS runtime
COPY --from=build /src/build/engine /usr/local/bin/engine
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD echo "uci" | timeout 5 /usr/local/bin/engine | grep -q "uciok"
ENTRYPOINT ["/usr/local/bin/engine"]
