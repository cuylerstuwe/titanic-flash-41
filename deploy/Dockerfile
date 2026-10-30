# Pinned Arm64 CUDA/compiler environment. Its Python model server is not run.
FROM lmsysorg/sglang:dev-dsv41@sha256:3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential cmake nasm pkg-config libssl-dev libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*
ARG RUST_VERSION=1.99.0
RUN test "$(uname -m)" = aarch64 \
    && cd /tmp \
    && curl --fail --location --retry 4 -O "https://static.rust-lang.org/dist/rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.xz" \
    && curl --fail --location --retry 4 -O "https://static.rust-lang.org/dist/rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.xz.sha256" \
    && sha256sum -c "rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.xz.sha256" \
    && tar -xf "rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.xz" \
    && "./rust-${RUST_VERSION}-aarch64-unknown-linux-gnu/install.sh" --prefix=/opt/rust --without=rust-docs \
    && rm -rf /tmp/rust-*
ENV PATH=/opt/rust/bin:/usr/local/cuda/bin:/opt/sglang/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
ENV CANADA_EVENT_LOG_MODE=memory
WORKDIR /project
HEALTHCHECK NONE
ENTRYPOINT ["/data/runtime/titanic-flash"]
