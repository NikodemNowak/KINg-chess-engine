# =============================================================================
# KINg — multi-stage slim build (CPU-only, no CUDA)
# =============================================================================

# Build stage
FROM ubuntu:22.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src
COPY third_party ./third_party
COPY tests ./tests
# NOTE: CMakeLists.txt builds with -mavx2 -mpopcnt (see CMAKE_CXX_FLAGS_RELEASE).
# Exact target arch will be confirmed when the organizer publishes the eval hardware
# (~07.06); a later task may switch to a portable baseline or add runtime dispatch.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target engine -j

# Runtime stage (slim, CPU-only — no CUDA)
FROM ubuntu:22.04 AS runtime
COPY --from=build /src/build/engine /usr/local/bin/engine
# Plan 2/3 will also COPY syzygy tablebases and the NNUE net here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD echo "uci" | timeout 5 /usr/local/bin/engine | grep -q "uciok"
ENTRYPOINT ["/usr/local/bin/engine"]
