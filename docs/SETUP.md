# Four-DGX-Spark installation recipe

This recipe targets **four Arm64 NVIDIA GB10 machines**, each with 128 GB unified memory, local NVMe storage and two 100 Gb/s RoCE interfaces. Use an isolated deployment directory and one unprivileged account with Docker access and key-only SSH from the head to all three peers. The model uses all four GPUs together. Stop other GPU services explicitly before weight staging or inference; the scripts refuse to take over a busy GPU.

Commands below run on the head unless another location is specified. Do not run Python with `-O`: setup assertions enforce model, artifact and deployment contracts.

## 1. Host and fabric prerequisites

On **every node**, install a supported DGX OS / Ubuntu Arm64 NVIDIA driver, Docker Engine, NVIDIA Container Toolkit and RDMA userspace (`rdma-core`, `ibverbs-utils`). Install `python3`, `rsync`, `git`, `openssh-client`, `nvme-cli` and systemd. Verify:

```sh
uname -m
nvidia-smi
docker info
ibv_devinfo
rdma link show
ip -br address
```

`uname -m` must report `aarch64`; the GPU must be GB10 / SM121. CUDA 13.0.88 inside the pinned container must work with the host driver. Keep swap available for the OS, but do not rely on swap for model allocations. At startup the portable launcher requires at least 110 GiB available RAM and 35 GiB free SSD space per node; containers are capped at 96 GiB with no extra swap allowance.

Configure two independent reachable RoCE fabrics, one port from every node on each fabric. Both switched fabrics and an equivalently reachable supported fabric are possible; the recipe assumes all four nodes can address every other node on both rails. Do not assume arbitrary point-to-point ring wiring satisfies that requirement. Confirm matching MTUs end to end and working bidirectional RDMA before loading the model. Logical NCCL ring algorithms do not require a physically ring-shaped network.

The checked profile uses IP traffic class **106** on both NCCL and the custom RoCE proxy. Account for it in switch QoS configuration; changing only `NCCL_IB_TC` does not change the separately compiled proxy. The recipe does not change switch settings automatically.

Record the HCA names, IP addresses, socket interface and **RoCEv2 GID index** on each node. For example:

```sh
ibdev2netdev
ls /sys/class/infiniband
cat /sys/class/infiniband/REPLACE_HCA/ports/1/gid_attrs/types/REPLACE_INDEX
cat /sys/class/infiniband/REPLACE_HCA/ports/1/gids/REPLACE_INDEX
```

The GID must encode that node's selected IPv4 address and be type `RoCE v2`. The runtime checks this correspondence and refuses a mismatch. HCA names and indices are not portable between installations. Reserve TCP ports 18041–18045 for the four-rank control/communication cohort, plus the chosen API port. Restrict those ports to the trusted cluster; the API requires a bearer key. Do not expose raw HTTP on the public Internet.

## 2. Allocate directories and clone

Use the same absolute root on all four nodes. The following examples use `/srv/titanic-flash-41`. An administrator should create that directory owned by the deployment account, with mode 0700. Put the head checkout at `source/` under it:

```sh
git clone https://github.com/cuylerstuwe/titanic-flash-41.git /srv/titanic-flash-41/source
cd /srv/titanic-flash-41/source
mkdir -p /srv/titanic-flash-41/data /srv/titanic-flash-41/model /srv/titanic-flash-41/state
chmod 700 /srv/titanic-flash-41/state
cp deploy/fleet.example.json /srv/titanic-flash-41/state/fleet.json
chmod 600 /srv/titanic-flash-41/state/fleet.json
```

Edit `state/fleet.json` with the four actual `hostname` outputs, the three SSH aliases, both HCA/IP pairs per node, the socket interface and GID indices. Rank 0 has `ssh: null`; the inventory must run on that head. The example's `192.0.2.*` / `198.51.100.*` addresses and `spark-*` hostnames are documentation placeholders, not an automatic network setup. Use a root without spaces or shell metacharacters.

The example explicitly selects a **2350 MHz** maximum GPU clock and temporary APST disablement on `/dev/nvme0` on each Spark. Before using the NVMe option, confirm that this controller backs the local Engram files and supports feature 12 with a 256-byte APST table (`sudo nvme get-feature /dev/nvme0 -f 12 -H`). Set a node's `disable_nvme0_apst` to `false` if that option is inapplicable. The helper does not alter another NVMe controller. Set `gpu_clock_cap_mhz` to `0` to leave clocks under host control; for a stock-clock comparison, restore stock clocks before startup. No lower automatic thermal cap is enabled.

Host-policy helpers run as the deployment account and need noninteractive administrative access to the documented `nvidia-smi`, `nvme`, runtime-directory provisioning and systemd timer operations. Have an administrator establish that access on the selected nodes; do not rely on an expiring cached sudo password. If unavailable, select the explicit host-policy overrides above. Arming fails before model startup if independent restoration cannot be established. Keep the original host settings and do not run a second host-policy manager over the same devices.

Leave `http_bind` at `127.0.0.1:8000` for an on-head client or SSH tunnel. To serve a trusted LAN, choose one specific private interface address and set `allow_remote: true`; wildcard binds are rejected. The API port must not overlap the five reserved cohort ports.

```sh
python3 deploy/fleet.py validate /srv/titanic-flash-41/state/fleet.json
ssh spark-1 hostname
ssh spark-2 hostname
ssh spark-3 hostname
```

Replace the SSH aliases in all commands with the inventory's actual aliases. Keep inventory, keys, model files and runtime state outside the checkout. Do not commit them.

## 3. Build the pinned Arm64 environment and engine

Read [the third-party notices](../THIRD_PARTY_NOTICES.md), including the separately fetched AGPL operator. The base container is pinned by registry digest. Its SGLang Python inference server is not used.

```sh
cd /srv/titanic-flash-41/source
docker build -f deploy/Dockerfile -t titanic-flash-41:toolchain .
docker run --rm --cpus=2 --memory=8g --memory-swap=8g \
  -v /srv/titanic-flash-41/source:/project:ro \
  -v /srv/titanic-flash-41/data:/data \
  --entrypoint /bin/bash titanic-flash-41:toolchain /project/tools/build.sh
docker run --rm --cpus=2 --memory=3g --memory-swap=3g \
  -v /srv/titanic-flash-41/source:/project:ro \
  -v /srv/titanic-flash-41/data:/data \
  --entrypoint /opt/sglang/bin/python titanic-flash-41:toolchain \
  /project/tools/prepare_runtime.py /data --with-external-woa
```

These steps need no GPU. They build the serving-only Rust executable, compile the project CUDA kernels, verify/copy the checked SM121 operators and compile all 18 external WO-A variants, including the compact 16/32-row paths. The build deliberately does not enable storage fault injection. Do not replace artifact files in a running deployment. Compiler identity and compiled source fingerprint are recorded in `data/runtime/provenance.json`.

Verify the complete prepared runtime before weight staging:

```sh
docker run --rm --cpus=2 --memory=3g --memory-swap=3g \
  -v /srv/titanic-flash-41/source:/project:ro \
  -v /srv/titanic-flash-41/data:/data:ro \
  --entrypoint /opt/sglang/bin/python titanic-flash-41:toolchain \
  /project/tools/verify_runtime.py /data
```

Export this exact image to each peer, so every rank uses the same image ID:

```sh
docker save titanic-flash-41:toolchain | ssh spark-1 docker load
docker save titanic-flash-41:toolchain | ssh spark-2 docker load
docker save titanic-flash-41:toolchain | ssh spark-3 docker load
```

## 4. Download and verify the canonical checkpoint

Allow **at least 1.25 TB of free space on the staging head** for the canonical download, all four staged shards, Engram backing and build files. Reserve about **200 GB per worker** for its runtime, local rank weights, Engram shard, container image and working room. A four-4-TB-Spark installation has comfortable room. Do not duplicate the full checkpoint on all workers unnecessarily.

```sh
python3 tools/download_model.py /srv/titanic-flash-41/model
```

The public checkpoint revision is pinned to `1bae0e1088993b9b82de08f59de5be5a34e42170`. There are 48 safetensors shards and small metadata files, approximately 510 GB total. The downloader verifies each file against immutable Hugging Face metadata, checks available disk space and resumes incomplete `.part` downloads. It never imports checkpoint code or silently accepts a damaged existing file. Retain the generated `download-manifest.json`.

Copy the verified tokenizer into runtime metadata. The checked model config and Engram metadata already came from `prepare_runtime.py`:

```sh
docker run --rm \
  -v /srv/titanic-flash-41/model:/model:ro \
  -v /srv/titanic-flash-41/data:/data \
  --entrypoint /bin/cp titanic-flash-41:toolchain \
  /model/tokenizer.json /data/model/tokenizer.json
```

The tokenizer SHA-256 must be `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`.

## 5. Build lossless Engram backing and GPU weight atlases

Pack the two large Engram tables into rank-local SSD files. This copies canonical FP8 row bytes and their E8M0 scale bytes without numerical conversion. Run each rank serially to keep memory and SSD pressure bounded:

```sh
for rank in 0 1 2 3; do
  docker run --rm --cpus=2 --memory=3g --memory-swap=3g \
    -v /srv/titanic-flash-41/source:/project:ro \
    -v /srv/titanic-flash-41/model:/model:ro \
    -v /srv/titanic-flash-41/data:/data \
    --entrypoint /opt/sglang/bin/python titanic-flash-41:toolchain \
    /project/tools/pack_engram.py /model /data/engram --rank "$rank"
done
```

Stop any model already using the head GPU, then stage all four ranks serially on that GPU. This does not require running inference simultaneously on all ranks. Numeric weight preparation uses the Rust GPU loader; the Python wrapper only orchestrates files and processes.

```sh
docker run --rm --gpus all --ipc=host --memory=96g --memory-swap=96g \
  -v /srv/titanic-flash-41/source:/project:ro \
  -v /srv/titanic-flash-41/model:/model:ro \
  -v /srv/titanic-flash-41/data:/data \
  --entrypoint /opt/sglang/bin/python titanic-flash-41:toolchain \
  /project/tools/stage_weights.py --rank all
```

This writes 40 layer atlases per rank, model-I/O and MTP atlases per rank, the canonical `vision-v1/r0` atlas, and a checkpoint manifest. **Rank 3 owns image placement in the current profile, and its split vision encoder (`CANADA_SCRATCH_VISION_SPLIT=1`) encodes each image on all four ranks.** Distribution therefore places that unchanged canonical vision atlas on every rank; its `r0` metadata describes the atlas, not the serving host. Each output records its rank, tensor selection, byte count and SHA-256. Existing complete outputs are verified before reuse. A partial staging directory is preserved for inspection; do not rename it into place or edit hashes to bypass a failed check.

Then derive the full projection matrices that the profile's [prefill options](PREFILL.md) load. This CPU-only step reads the four ranks' staged atlases on the head and copies their shard bytes, in rank order, into `data/kernels/claude-{wob,shared,engram-wkv}-full-v1` (about 1.9 GB). It converts no values, records each source atlas's digest, and verifies existing outputs before reuse:

```sh
docker run --rm --cpus=2 --memory=4g --memory-swap=4g \
  -v /srv/titanic-flash-41/source:/project:ro \
  -v /srv/titanic-flash-41/data:/data \
  --entrypoint /opt/sglang/bin/python titanic-flash-41:toolchain \
  /project/tools/stage_full_matrices.py
```

At load, each rank checks every derived file's digest and that its own part equals the shard it serves, so a wrong or stale file stops startup rather than changing outputs. Rerun this step whenever the atlases are re-staged. If the head no longer holds every rank's atlases, run `stage_full_matrices.py --data /data extract --rank R --parts DIR` on each node, gather the four `DIR/rR` folders on the head and run `stage_full_matrices.py --data /data assemble --parts DIR` there.

Copy only the correct rank's data to each worker:

```sh
python3 tools/sync_fleet.py /srv/titanic-flash-41/state/fleet.json
```

This copies the runtime, kernels (including the derived matrices) and model metadata to every worker, plus each rank's own shards. The controller's startup checks confirm that every rank holds the same derived matrices, built from the atlases it serves. The worker deployment roots must be writable by the SSH account. Generated container files can be root-owned; they only need to be readable for distribution/runtime. If administrative ownership adjustment is needed, scope it to this new deployment directory. Keep the original checkpoint until staging, distribution and a full startup have succeeded. Engram backing and the rank's staged atlases must stay on each node's local SSD during serving.

## 6. Start and validate

Ensure all four GPUs are idle. Start the controller on the head in a persistent terminal:

```sh
python3 deploy/fleet.py serve /srv/titanic-flash-41/state/fleet.json
```

It checks node identities, available memory/disk, image identity, runtime hashes and required files; installs immutable host-policy helpers; arms the selected temporary policies with node-local restoration timers; writes a fresh private rendezvous ticket; launches four owned containers; and waits for Ready. Initial load and graph setup can take several minutes. A failed rank stops the entire owned cohort. The controller never stops another project's containers.

In a second head terminal:

```sh
python3 /srv/titanic-flash-41/source/deploy/fleet.py status /srv/titanic-flash-41/state/fleet.json
python3 /srv/titanic-flash-41/source/tools/smoke_http.py \
  --base-url http://127.0.0.1:8000 \
  --key-file /srv/titanic-flash-41/state/keys/http.key
```

Require Ready, four running workers, successful fresh/cached requests and consistent timing/cache counts. The smoke uses temperature 1, top-p 1, top-k -1 and high reasoning. It is a serving check, not a broad quality benchmark.

Stop the foreground controller with Ctrl-C, or from another terminal:

```sh
python3 deploy/fleet.py stop /srv/titanic-flash-41/state/fleet.json
```

After that succeeds, install the example systemd unit if unattended supervision is desired. Replace its `User=` and paths, then have an administrator install it as `/etc/systemd/system/titanic-flash.service`, run `systemd-analyze verify`, `systemctl daemon-reload`, and `systemctl enable --now titanic-flash.service`. Do not leave the foreground controller running when starting systemd. Check Ready again. To keep a systemd-managed service stopped, use `systemctl stop titanic-flash.service`.

## 7. Preserve a rollback

Keep the known-good source commit, image ID, runtime provenance, kernel inventory and private fleet inventory together. Updates should use a separate data/runtime directory and be hash-verified before replacing a stopped service. Stop all four owned ranks before changing an image, kernel, source profile or inventory. Restore the complete previous set on failure; never mix ranks from different versions or reuse a rendezvous ticket between cohorts. A successful build is not sufficient qualification for a new kernel, quantization or protocol change.

The selected clock/APST policies are temporary, restored on stop, and protected by independent node-local lease expiry. Only transitions write durable state; renewals use private tmpfs. Retain `state/host-policy-code/` and the saved restoration records until the associated generations have been restored. Firmware, networking, router configuration and other inference services remain host-administration responsibilities. See [operations](OPERATIONS.md), [validation scope](VALIDATION.md) and [benchmark reproduction](BENCHMARKS.md).
