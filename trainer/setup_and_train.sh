#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Krea 2 Character LoRA - remote setup and training (ai-toolkit)
# =============================================================================
# Fully parameterised - nothing about one particular character is baked in -
# and driven by train_runpod.py over SSH. It assumes no operator is watching:
# the person who started this is not technical and may well close their laptop
# halfway through.
#
# Hence self-termination. This script owns shutting the pod down if the
# controlling laptop disappears, because a runaway bill is the worst possible
# outcome for the audience it serves.
#
# Environment variables (all set by train_runpod.py):
#   HF_TOKEN          HuggingFace token with access to gated krea/Krea-2-Raw
#   RUN_NAME          Run name, must match `name` in config.yaml
#   TRIGGER_WORD      The word that summons the character
#   SETUP_ONLY=1      Validate everything, do not launch training
#   SELF_TERMINATE=1  Delete this pod when finished (default 1)
#   MAX_TRAIN_HOURS   Hard cap on training time (default 6)
#   GRACE_HOURS       How long to wait after training for the client to
#                     collect results before shutting down anyway (default 2)
#
# Training runs under nohup so the SSH session can drop without killing it.
# Never buffer hours of output over SSH.
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_header()  { echo -e "\n${PURPLE}=== $1 ===${NC}"; }

WORK_DIR="/workspace/krea2-training"
REPO_DIR="$WORK_DIR/ai-toolkit"
DATASET_DIR="$WORK_DIR/dataset"
OUTPUT_DIR="$WORK_DIR/output"
CONFIG_FILE="$WORK_DIR/config.yaml"
# Optional. Uploaded by the cloud client, which trains with sampling disabled
# and renders previews afterwards from the finished LoRA instead.
PREVIEW_CONFIG="$WORK_DIR/preview_config.yaml"
TRAIN_LOG="$WORK_DIR/training.log"
CAPTION_LOG="$WORK_DIR/captioning.log"
CAPTION_SCRIPT="$WORK_DIR/caption_dataset.py"
WRAPPER="$WORK_DIR/run_training.sh"
TERMINATOR="$WORK_DIR/terminate_pod.sh"
TERMINATOR_PY="$WORK_DIR/terminate_pod.py"

# The pod cannot identify itself. Verified live on two pods (A4000, A5000):
# RUNPOD_POD_ID is absent from the interactive shell, the non-interactive
# shell, /etc/environment and /root/.bashrc; /etc/rc.local does not exist at
# all. So the id has to come from the client via POD_ID. The RUNPOD_POD_ID
# read below costs nothing and would pick it up if RunPod ever starts setting
# it, but nothing depends on that happening.
POD_ID="${RUNPOD_POD_ID:-${POD_ID:-}}"
# Needed so the pod can terminate itself. runpodctl ships unauthenticated and
# the pod is given no credentials of its own, so this must come from the
# client. It is the user's own key, on the user's own short-lived pod.
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"

HF_TOKEN="${HF_TOKEN:-}"
RUN_NAME="${RUN_NAME:-character_krea2}"
TRIGGER_WORD="${TRIGGER_WORD:-character}"
SETUP_ONLY="${SETUP_ONLY:-0}"
SELF_TERMINATE="${SELF_TERMINATE:-1}"
MAX_TRAIN_HOURS="${MAX_TRAIN_HOURS:-6}"
GRACE_HOURS="${GRACE_HOURS:-2}"

# The model download is ~35GB and HuggingFace keeps a second transient copy
# while it verifies. The default cache lives under /root, which is on the
# small container disk; /workspace is the big volume. Getting this wrong
# fills the disk 30 minutes into the run.
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "$WORK_DIR" "$OUTPUT_DIR" "$HF_HOME"

# =============================================================================
# PHASE 1: Environment
# =============================================================================

print_header "Phase 1: Environment setup"

if [ -n "$POD_ID" ]; then
    print_status "Pod id: $POD_ID (self-terminate enabled: $SELF_TERMINATE)"
else
    print_warning "No pod id available. This pod cannot shut itself down;"
    print_warning "the client and the terminateAfter deadline still can."
fi

apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq git python3-pip python3-venv curl >/dev/null 2>&1 || true

# A clone that dies part-way still leaves a directory behind, and a plain
# -d test cannot tell that apart from a finished one, so a retry would build
# a venv on top of a repo missing half its files. Seen live on an L40S: the
# checkout stopped at 64% with "unable to create file ...: File exists" on a
# genuinely fresh pod, having downloaded all 30MB without complaint. The
# transfer was fine; writing the working tree was not. So completion gets
# recorded explicitly and anything lacking that marker is thrown away.
#
# Deleting REPO_DIR is safe: the dataset, config and output all live beside
# it in WORK_DIR, not inside it. Only .venv is lost, which is worthless
# without the repo it belongs to.
CLONE_MARKER="$REPO_DIR/.download_complete"

if [ -f "$CLONE_MARKER" ]; then
    print_status "Training software already present"
else
    if [ -e "$REPO_DIR" ]; then
        print_warning "Found an unfinished earlier download. Starting it again."
    fi
    ATTEMPT=1
    while [ "$ATTEMPT" -le 3 ]; do
        print_status "Downloading the training software (ai-toolkit), try $ATTEMPT of 3..."
        rm -rf "$REPO_DIR"
        if git clone --depth 1 https://github.com/ostris/ai-toolkit.git "$REPO_DIR" \
            && git -C "$REPO_DIR" rev-parse --verify HEAD >/dev/null 2>&1 \
            && [ -f "$REPO_DIR/run.py" ]; then
            touch "$CLONE_MARKER"
            break
        fi
        print_warning "That try did not finish cleanly."
        ATTEMPT=$((ATTEMPT + 1))
        sleep 10
    done
    if [ ! -f "$CLONE_MARKER" ]; then
        print_error "Could not download the training software after 3 tries."
        print_error "This is usually a bad pod rather than anything you did."
        print_error "Please start the run again to get a different computer."
        exit 1
    fi
    print_success "Downloaded"
fi

cd "$REPO_DIR" || exit 1

if [ ! -d ".venv" ]; then
    print_status "Creating Python environment..."
    python3 -m venv .venv
fi

DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
print_status "NVIDIA driver: ${DRIVER_VER:-unknown}"

if .venv/bin/python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    print_success "PyTorch with CUDA already working, skipping reinstall"
else
    # Unpinned on purpose: RunPod images set PIP_CONSTRAINT pinning the torch
    # stack, so an explicit version pin that differs is guaranteed to fail
    # resolution.
    # pip runs unquieted so a stalled download names the file it stalled on.
    # The progress bar stays off: it redraws with \r, and the client only
    # prints complete lines, so it would arrive as one garbled line at the end.
    print_status "Installing PyTorch (this takes a few minutes)..."
    .venv/bin/pip install --upgrade pip --progress-bar off
    if ! .venv/bin/pip install --no-cache-dir torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu128 --progress-bar off; then
        print_warning "cu128 index failed, falling back to default wheels"
        .venv/bin/pip install --no-cache-dir torch torchvision torchaudio --progress-bar off
    fi
    print_success "PyTorch installed"
fi

if .venv/bin/python3 -c "import diffusers, transformers, optimum.quanto" 2>/dev/null \
        && [ -f ".deps_installed" ]; then
    print_success "Dependencies already installed"
else
    print_status "Installing dependencies (10-20 minutes on a fresh pod)..."
    if ! .venv/bin/pip install -r requirements.txt --progress-bar off; then
        print_error "Dependency install failed."
        exit 1
    fi
    touch .deps_installed
    print_success "Dependencies installed"
fi

# requirements.txt resolution can quietly swap the CUDA build for a CPU one.
if ! .venv/bin/python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    print_status "Re-forcing CUDA PyTorch (dependencies overwrote it)..."
    .venv/bin/pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --progress-bar off \
        || .venv/bin/pip install --no-cache-dir torch torchvision torchaudio --force-reinstall --progress-bar off
fi

# =============================================================================
# PHASE 2: HuggingFace access
# =============================================================================

print_header "Phase 2: HuggingFace access"

if [ -z "$HF_TOKEN" ]; then
    print_error "No HuggingFace token was provided."
    exit 1
fi
export HF_TOKEN

.venv/bin/python3 - <<'PYEOF'
import os, sys
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
try:
    info = HfApi(token=token).model_info("krea/Krea-2-Raw", files_metadata=False)
    print(f"Access OK: krea/Krea-2-Raw (gated={info.gated})")
except Exception as exc:
    print(f"CANNOT ACCESS krea/Krea-2-Raw: {exc}")
    print("The HuggingFace account for this token has not accepted the Krea 2")
    print("licence. Open https://huggingface.co/krea/Krea-2-Raw and accept it.")
    sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then
    exit 1
fi
print_success "Model access verified"

# =============================================================================
# PHASE 3: Dataset and captions
# =============================================================================

print_header "Phase 3: Dataset and captions"

IMG_COUNT=$(find "$DATASET_DIR" -maxdepth 1 \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \) 2>/dev/null | wc -l)
print_status "Dataset: $IMG_COUNT images"

if [ "$IMG_COUNT" -lt 10 ]; then
    print_error "Only $IMG_COUNT images found. At least 10 are required."
    exit 1
fi

CAP_COUNT=$(find "$DATASET_DIR" -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)

if [ "$CAP_COUNT" -ge "$IMG_COUNT" ]; then
    print_success "Captions already present ($CAP_COUNT), skipping captioning"
elif [ -f "$CAPTION_SCRIPT" ]; then
    print_status "Describing the photos automatically (a few minutes)..."
    if .venv/bin/python3 "$CAPTION_SCRIPT" \
            --dataset "$DATASET_DIR" \
            --trigger "$TRIGGER_WORD" > "$CAPTION_LOG" 2>&1; then
        print_success "Captions written"
        tail -3 "$CAPTION_LOG" || true
    else
        print_warning "Automatic captioning failed. Using simple captions instead."
        print_warning "Training still works; see $CAPTION_LOG for details."
        tail -5 "$CAPTION_LOG" || true
    fi
else
    print_warning "No captioning script uploaded, using simple captions"
fi

# Backstop: every image must end up with a caption containing the trigger
# word. With cache_text_embeddings enabled ai-toolkit ignores the config's
# trigger_word entirely, so the word has to be in the caption text itself.
print_status "Checking every photo has a caption..."
.venv/bin/python3 - "$DATASET_DIR" "$TRIGGER_WORD" <<'PYEOF'
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
trigger = sys.argv[2].strip()
exts = {".jpg", ".jpeg", ".png", ".webp"}

created = fixed = 0
for image in sorted(dataset.iterdir()):
    if image.suffix.lower() not in exts:
        continue
    caption_path = image.with_suffix(".txt")
    if not caption_path.exists() or not caption_path.read_text(encoding="utf-8").strip():
        caption_path.write_text(f"{trigger}, a photograph of {trigger}", encoding="utf-8")
        created += 1
        continue
    text = caption_path.read_text(encoding="utf-8").strip()
    if not text.lower().startswith(trigger.lower()):
        caption_path.write_text(f"{trigger}, {text}", encoding="utf-8")
        fixed += 1

print(f"Captions: {created} created, {fixed} had the trigger word added")
PYEOF
print_success "Captions ready"

# =============================================================================
# PHASE 4: Config and hardware checks
# =============================================================================

print_header "Phase 4: Config and hardware checks"

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file missing at $CONFIG_FILE"
    exit 1
fi

.venv/bin/python3 - "$CONFIG_FILE" "$RUN_NAME" <<'PYEOF'
import sys, yaml

with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)

expected_name = sys.argv[2]
proc = cfg["config"]["process"][0]
model, train, net = proc["model"], proc["train"], proc["network"]

assert model["arch"] == "krea2", f"arch must be krea2, got {model['arch']}"
assert model["name_or_path"] == "krea/Krea-2-Raw", model["name_or_path"]
assert cfg["config"]["name"] == expected_name, (
    f"config name {cfg['config']['name']!r} != expected {expected_name!r}"
)

# ai-toolkit raises if these disagree, and it does so only after the model
# has loaded - many minutes in. Catch it here instead.
ds = proc["datasets"][0]
assert train.get("cache_text_embeddings", False) == ds.get("cache_text_embeddings", False), (
    "cache_text_embeddings must match between train and dataset blocks"
)

print(f"Config OK: {cfg['config']['name']}")
print(f"  steps={train['steps']} lr={train['lr']} batch={train['batch_size']}")
print(f"  lora rank={net['linear']} alpha={net['linear_alpha']}")
print(f"  resolution={ds['resolution']} quantize={model.get('quantize')}")
print(f"  layer_offloading={model.get('layer_offloading', False)}")
PYEOF
if [ $? -ne 0 ]; then
    print_error "The training config is not valid."
    exit 1
fi
print_success "Config validated"

.venv/bin/python3 - <<'PYEOF'
import sys, torch

if not torch.cuda.is_available():
    print("CUDA: NOT AVAILABLE - training cannot run")
    sys.exit(1)
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
print(f"GPU: {name}, {vram}GB VRAM")
if vram < 40:
    print(f"WARNING: {vram}GB is below what this config expects")
PYEOF
if [ $? -ne 0 ]; then
    exit 1
fi

DISK_FREE=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
print_status "Free disk on /workspace: ${DISK_FREE}GB (about 40GB needed)"
if [ -n "$DISK_FREE" ] && [ "$DISK_FREE" -lt 40 ] 2>/dev/null; then
    print_warning "Disk space is tight. The model download may fail."
fi

if [ "$SETUP_ONLY" = "1" ]; then
    print_header "Setup-only mode - stopping before training"
    print_success "Everything checks out. Nothing was started."
    exit 0
fi

# =============================================================================
# PHASE 5: Launch training with a self-terminating wrapper
# =============================================================================

print_header "Phase 5: Launch training"

if pgrep -f "run.py $CONFIG_FILE" > /dev/null 2>&1; then
    print_warning "Training is already running for this config:"
    pgrep -af "run.py $CONFIG_FILE"
    exit 1
fi

MAX_TRAIN_SECONDS=$(( MAX_TRAIN_HOURS * 3600 ))
GRACE_SECONDS=$(( GRACE_HOURS * 3600 ))
# Absolute ceiling for the watchdog: everything the run could legitimately
# need, plus an hour.
WATCHDOG_SECONDS=$(( MAX_TRAIN_SECONDS + GRACE_SECONDS + 3600 ))

# -----------------------------------------------------------------------------
# One shutdown implementation, used by both the wrapper and the watchdog.
#
# The real work is in terminate_pod.py (uploaded by the client). It is Python
# rather than bash because it has to verify the pod is genuinely gone, and
# RunPod reports application errors inside HTTP 200 bodies - too fiddly to do
# safely with curl inside a nested heredoc.
#
# Both the pod id and the API key are baked in here because, verified live,
# the pod has neither RUNPOD_POD_ID nor RUNPOD_API_KEY in its environment.
# -----------------------------------------------------------------------------
cat > "$TERMINATOR" <<TERM_EOF
#!/usr/bin/env bash
exec python3 "$TERMINATOR_PY" "\${RUNPOD_POD_ID:-$POD_ID}" "$RUNPOD_API_KEY"
TERM_EOF
chmod +x "$TERMINATOR"
# Contains the account API key.
chmod 600 "$TERMINATOR" 2>/dev/null || true

# Load-bearing, and deliberately fatal. This used to be a warning that said
# "the client will still terminate it" - true only while the client's finally
# terminated unconditionally. Now that the client stands down once this pod
# reports itself armed (SELF_TERMINATE_ARMED below), a missing terminator
# would mean NOTHING can shut the pod down. Refusing to train is the safe
# failure: the user loses a few minutes, not an open-ended GPU bill.
if [ ! -f "$TERMINATOR_PY" ] && [ "$SELF_TERMINATE" = "1" ]; then
    print_error "The program that shuts this computer down is missing, so"
    print_error "training will not start. You have not been charged for any"
    print_error "training. Please try again."
    exit 1
fi

cat > "$WRAPPER" <<WRAPPER_EOF
#!/usr/bin/env bash
# Wrapper around the training run. Owns pod shutdown so that a laptop
# closing, crashing, or losing wifi cannot leave a GPU billing forever.
set -uo pipefail

WORK_DIR="$WORK_DIR"
REPO_DIR="$REPO_DIR"
CONFIG_FILE="$CONFIG_FILE"
PREVIEW_CONFIG="$PREVIEW_CONFIG"

cd "\$REPO_DIR" || exit 1

# 'timeout' guarantees a hung run still reaches the shutdown code below.
timeout --signal=TERM --kill-after=120 $MAX_TRAIN_SECONDS \\
    env HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" DISABLE_TELEMETRY=YES \\
        HF_HUB_ENABLE_HF_TRANSFER=0 \\
    .venv/bin/python run.py "\$CONFIG_FILE"
RC=\$?

# Previews run BEFORE the client is told training finished. monitor() breaks
# out of its loop the moment it sees TRAINING_EXIT_CODE= in the log, so
# anything written after that line risks not being collected.
#
# Every failure here is swallowed on purpose. The LoRA is the deliverable and
# it is already on disk; a preview image is a nicety and must never be able to
# affect the exit code. This is the same trade train_local.py makes in
# generate_previews().
if [ \$RC -eq 0 ] && [ -f "\$PREVIEW_CONFIG" ]; then
    echo "Making preview pictures from the finished LoRA (about 10 minutes)."
    echo "Your LoRA is already saved and safe."
    timeout --signal=TERM --kill-after=60 3600 \\
        env HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" DISABLE_TELEMETRY=YES \\
            HF_HUB_ENABLE_HF_TRANSFER=0 \\
        .venv/bin/python run.py "\$PREVIEW_CONFIG" \\
        || echo "PREVIEW_FAILED - no preview pictures, but the LoRA is fine."
fi

echo ""
echo "TRAINING_EXIT_CODE=\$RC"
if [ \$RC -eq 124 ] || [ \$RC -eq 137 ]; then
    echo "TRAINING_TIMED_OUT after $MAX_TRAIN_HOURS hours"
fi
touch "\$WORK_DIR/TRAINING_DONE"
# A marker named DONE written on a crash reads as success to anything that
# only tests for it. Record the failure separately so the client, and anyone
# reading the pod by hand, can tell the two apart.
if [ \$RC -ne 0 ]; then
    touch "\$WORK_DIR/TRAINING_FAILED"
fi
echo "TRAINING_DONE_MARKER_WRITTEN"

if [ "$SELF_TERMINATE" != "1" ]; then
    echo "Self-terminate disabled; leaving pod running."
    exit \$RC
fi

# Give the client time to download results. It writes CLIENT_DONE the moment
# it has everything, so the normal path shuts down within seconds.
# A failed run gets a longer window than a successful one. On success the
# client collects within seconds and writes CLIENT_DONE. On failure there may
# be a salvageable checkpoint on a disk that is about to be destroyed, and the
# user has to notice, read the message, and start the client again - which the
# default two hours does not reliably allow for. The watchdog ceiling
# (MAX_TRAIN + GRACE + 1h) is unchanged and still hard-kills the pod, so this
# cannot extend the worst-case bill beyond what was already budgeted.
GRACE_LEFT=$GRACE_SECONDS
if [ \$RC -ne 0 ]; then
    GRACE_LEFT=\$(( $GRACE_SECONDS * 3 ))
fi
echo "Waiting up to \$(( GRACE_LEFT / 3600 )) hour(s) for results to be collected..."
WAITED=0
while [ \$WAITED -lt \$GRACE_LEFT ]; do
    if [ -f "\$WORK_DIR/CLIENT_DONE" ]; then
        echo "Client collected the results."
        break
    fi
    sleep 15
    WAITED=\$(( WAITED + 15 ))
done

echo "SHUTTING DOWN POD NOW"
bash "$TERMINATOR"
exit \$RC
WRAPPER_EOF

chmod +x "$WRAPPER"
rm -f "$WORK_DIR/TRAINING_DONE" "$WORK_DIR/CLIENT_DONE"

# -----------------------------------------------------------------------------
# Watchdog: an independent dead-man's switch.
#
# Measured live (Aug 2026): a pod with terminateAfter set to now+10min was
# still RUNNING 12 minutes past that deadline. RunPod's server-side deadline
# cannot be treated as a reliable billing backstop, so this process provides
# one that does not depend on RunPod, on the training wrapper surviving, or
# on the user's laptop staying awake. setsid detaches it so nothing that
# happens to the training process group can take it down with it.
# -----------------------------------------------------------------------------
print_status "Arming shutdown watchdog ($(( WATCHDOG_SECONDS / 3600 )) hour hard limit)..."
setsid nohup bash -c \
    "sleep $WATCHDOG_SECONDS; echo \"WATCHDOG: hard limit reached\"; bash '$TERMINATOR'" \
    > "$WORK_DIR/watchdog.log" 2>&1 < /dev/null &
WATCHDOG_PID=$!
# Checked after a pause rather than immediately: $! is set the instant bash
# forks, so testing it straight away only proves the fork happened. Refusing
# to train is the safe failure here. The client tears the pod down when setup
# exits non-zero, so a friend loses a few minutes, not money - and the one
# thing we must never do is start billing without the hard limit in place.
sleep 2
if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    print_success "Watchdog armed (PID $WATCHDOG_PID)"
    # Handshake for the client's finally. Printed only once the terminator
    # exists AND the watchdog is confirmed alive - i.e. only once this pod can
    # provably shut itself down. Until the client sees this line it stays the
    # killer of last resort. Do not move it earlier.
    echo "SELF_TERMINATE_ARMED=1"
else
    print_error "The safety timer that shuts this computer down could not"
    print_error "be started, so training will not begin. You have not been"
    print_error "charged for any training. Please try again."
    exit 1
fi

# setsid and </dev/null are both load-bearing, not decoration. Launched over
# SSH, a background process inherits the channel's file descriptors, and the
# client's exec_command() then blocks until they close - which for a training
# run is hours. Verified live: without these the client hangs indefinitely
# instead of returning to the monitor loop.
print_status "Starting training (log: $TRAIN_LOG)..."
setsid nohup bash "$WRAPPER" > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
sleep 5

if kill -0 "$TRAIN_PID" 2>/dev/null; then
    print_success "Training started (PID $TRAIN_PID)"
    echo ""
    echo "  Log:    $TRAIN_LOG"
    echo "  Output: $OUTPUT_DIR"
    echo ""
    print_status "The first 30-45 minutes are spent downloading the 35GB model."
    sleep 5
    head -20 "$TRAIN_LOG" 2>/dev/null || true
else
    print_error "Training died immediately. Log:"
    cat "$TRAIN_LOG" 2>/dev/null || true
    exit 1
fi
