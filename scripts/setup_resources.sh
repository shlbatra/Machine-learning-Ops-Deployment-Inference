#!/bin/bash

# Parallel setup driver for the three "always-on" POC resources:
#   1. Artifact Registry repository   (~1 min)
#   2. Vertex AI Feature Store        (~2-5 min)
#   3. Cloud Composer 2 environment   (~25 min — the long pole)
#
# Mirrors destroy_resources.sh. The three are independent of each other, so they
# run concurrently and the wall clock is Composer's ~25 min rather than the sum.
# Each script's output goes to its own log file — interleaving three streams of
# gcloud output on one terminal is unreadable — and a status table is printed at
# the end.
#
# Exit code is 0 only if every script that actually ran succeeded.
#
# Usage:
#   ./scripts/setup_resources.sh                       # run all three in parallel
#   ./scripts/setup_resources.sh --skip-composer       # leave Composer out
#   ./scripts/setup_resources.sh --skip-feature-store
#   ./scripts/setup_resources.sh --skip-artifact-registry
#   ./scripts/setup_resources.sh --feature-store-args="fraud fraud_user"
#   ./scripts/setup_resources.sh --serial              # one at a time (easier debugging)
#   ./scripts/setup_resources.sh --timeout=3600        # per-script watchdog, 0 = off
#   ./scripts/setup_resources.sh --heartbeat=60        # progress line interval
#   ./scripts/setup_resources.sh --log-dir=/tmp/setup  # where to write logs
#   ./scripts/setup_resources.sh --skip-preflight      # bypass the pre-flight checks
#   ./scripts/setup_resources.sh --dry-run             # print what would run

# NOTE: deliberately NOT `set -e`. A failing child script must be recorded and
# reported alongside the others, not abort the driver and hide the rest.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/scripts"

PROJECT_ID="deeplearning-sahil"

# --- defaults ---------------------------------------------------------------
SKIP_ARTIFACT_REGISTRY=false
SKIP_FEATURE_STORE=false
SKIP_COMPOSER=false
FEATURE_STORE_ARGS=""
SERIAL=false
DRY_RUN=false
SKIP_PREFLIGHT=false
TIMEOUT=0                    # seconds; 0 disables the watchdog
HEARTBEAT=60                 # seconds between progress lines
LOG_DIR=""

# --- arg parsing ------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --skip-artifact-registry) SKIP_ARTIFACT_REGISTRY=true ;;
        --skip-feature-store)     SKIP_FEATURE_STORE=true ;;
        --skip-composer)          SKIP_COMPOSER=true ;;
        --feature-store-args=*)   FEATURE_STORE_ARGS="${arg#*=}" ;;
        --serial)                 SERIAL=true ;;
        --dry-run)                DRY_RUN=true ;;
        --skip-preflight)         SKIP_PREFLIGHT=true ;;
        --timeout=*)              TIMEOUT="${arg#*=}" ;;
        --heartbeat=*)            HEARTBEAT="${arg#*=}" ;;
        --log-dir=*)              LOG_DIR="${arg#*=}" ;;
        -h|--help)
            sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# bash 3.2 on macOS has no associative arrays — parallel indexed arrays instead.
NAMES=()
SCRIPTS=()
ARGS=()
PIDS=()
CODES=()
STARTS=()
ELAPSED=()
LOGS=()

add_task() {
    NAMES+=("$1")
    SCRIPTS+=("$2")
    ARGS+=("$3")
    PIDS+=("")
    CODES+=("")
    STARTS+=("0")
    ELAPSED+=("0")
    LOGS+=("")
}

# Composer first: it is by far the longest, so starting it first shaves its
# startup latency off the critical path when the others contend for API quota.
[ "$SKIP_COMPOSER" = false ]          && add_task "composer"          "setup_composer.sh"          ""
[ "$SKIP_FEATURE_STORE" = false ]     && add_task "feature-store"     "setup_feature_store.sh"     "$FEATURE_STORE_ARGS"
[ "$SKIP_ARTIFACT_REGISTRY" = false ] && add_task "artifact-registry" "setup_artifact_registry.sh" ""

TASK_COUNT=${#NAMES[@]}
if [ "$TASK_COUNT" -eq 0 ]; then
    echo "Nothing to do — all three targets were skipped."
    exit 0
fi

# --- colors (only when attached to a terminal) ------------------------------
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GREEN=$'\033[0;32m'; C_YELLOW=$'\033[0;33m'
    C_BOLD=$'\033[1m';   C_OFF=$'\033[0m'
else
    C_RED=""; C_GREEN=""; C_YELLOW=""; C_BOLD=""; C_OFF=""
fi

fmt_duration() {
    local s=$1
    printf "%dm%02ds" $((s / 60)) $((s % 60))
}

# --- pre-flight -------------------------------------------------------------
# These run before anything is launched. Without them a bad credential state
# fails all three ~30s in, and a missing venv fails feature-store one second in
# while Composer keeps building for 25 minutes.
preflight() {
    local failed=false

    if ! command -v gcloud >/dev/null 2>&1; then
        echo "  ${C_RED}FAIL${C_OFF}  gcloud not found on PATH"
        failed=true
    else
        local account
        account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
        if [ -z "$account" ]; then
            echo "  ${C_RED}FAIL${C_OFF}  no active gcloud account — run: gcloud auth login"
            failed=true
        else
            echo "  ok    gcloud account: $account"
        fi
    fi

    for i in $(seq 0 $((TASK_COUNT - 1))); do
        local path="$SCRIPT_DIR/${SCRIPTS[$i]}"
        if [ ! -f "$path" ]; then
            echo "  ${C_RED}FAIL${C_OFF}  missing script: $path"
            failed=true
        elif [ ! -x "$path" ]; then
            echo "  ${C_RED}FAIL${C_OFF}  not executable: $path (chmod +x it)"
            failed=true
        fi
    done

    # setup_feature_store.sh is `python -m feature_store.setup`, so it needs the
    # project's Python environment active. Catch that here rather than 25 minutes
    # into a Composer build.
    if [ "$SKIP_FEATURE_STORE" = false ]; then
        if ! python -c "import feature_store" >/dev/null 2>&1; then
            echo "  ${C_RED}FAIL${C_OFF}  cannot import feature_store — activate the venv (source .venv/bin/activate)"
            failed=true
        else
            echo "  ok    feature_store importable"
        fi
    fi

    if [ "$failed" = true ]; then
        echo ""
        echo "${C_RED}Pre-flight failed.${C_OFF} Fix the above, or re-run with --skip-preflight."
        exit 2
    fi
}

echo "${C_BOLD}=== Resource Setup ===${C_OFF}"
echo "Project: $PROJECT_ID"
echo "Targets: ${NAMES[*]}"
echo "Mode:    $([ "$SERIAL" = true ] && echo "serial" || echo "parallel")"
echo ""

if [ "$SKIP_PREFLIGHT" = false ] && [ "$DRY_RUN" = false ]; then
    echo "Pre-flight checks..."
    preflight
    echo ""
fi

if [ "$DRY_RUN" = true ]; then
    echo "Dry run — would execute:"
    for i in $(seq 0 $((TASK_COUNT - 1))); do
        echo "  $SCRIPT_DIR/${SCRIPTS[$i]} ${ARGS[$i]}"
    done
    exit 0
fi

# --- log directory ----------------------------------------------------------
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="$REPO_ROOT/.setup-logs/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$LOG_DIR" || { echo "Cannot create log dir: $LOG_DIR" >&2; exit 2; }
echo "Logs: $LOG_DIR"
echo ""

# --- launch -----------------------------------------------------------------
for i in $(seq 0 $((TASK_COUNT - 1))); do
    LOGS[$i]="$LOG_DIR/${NAMES[$i]}.log"
    STARTS[$i]=$(date +%s)

    if [ "$SERIAL" = true ]; then
        echo "Running ${NAMES[$i]}..."
        if [ -n "${ARGS[$i]}" ]; then
            # shellcheck disable=SC2086
            "$SCRIPT_DIR/${SCRIPTS[$i]}" ${ARGS[$i]} >"${LOGS[$i]}" 2>&1
        else
            "$SCRIPT_DIR/${SCRIPTS[$i]}" >"${LOGS[$i]}" 2>&1
        fi
        CODES[$i]=$?
        ELAPSED[$i]=$(( $(date +%s) - ${STARTS[$i]} ))
        if [ "${CODES[$i]}" -eq 0 ]; then
            echo "  ${C_GREEN}done${C_OFF} in $(fmt_duration ${ELAPSED[$i]})"
        else
            echo "  ${C_RED}failed${C_OFF} (exit ${CODES[$i]}) after $(fmt_duration ${ELAPSED[$i]})"
        fi
    else
        if [ -n "${ARGS[$i]}" ]; then
            # shellcheck disable=SC2086
            "$SCRIPT_DIR/${SCRIPTS[$i]}" ${ARGS[$i]} >"${LOGS[$i]}" 2>&1 &
        else
            "$SCRIPT_DIR/${SCRIPTS[$i]}" >"${LOGS[$i]}" 2>&1 &
        fi
        PIDS[$i]=$!
        echo "Started ${NAMES[$i]} (pid ${PIDS[$i]}) → ${LOGS[$i]}"
    fi
done

# --- wait, with progress and an optional watchdog ---------------------------
if [ "$SERIAL" = false ]; then
    echo ""
    OVERALL_START=$(date +%s)
    LAST_PRINT=$OVERALL_START
    POLL=2                       # poll often so durations are accurate...
                                 # ...but only print every HEARTBEAT seconds

    DONE=()
    for i in $(seq 0 $((TASK_COUNT - 1))); do DONE+=("false"); done

    while true; do
        running=0
        status_line=""
        now=$(date +%s)

        for i in $(seq 0 $((TASK_COUNT - 1))); do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                running=$((running + 1))
                task_elapsed=$((now - ${STARTS[$i]}))

                if [ "$TIMEOUT" -gt 0 ] && [ "$task_elapsed" -gt "$TIMEOUT" ]; then
                    echo "${C_YELLOW}Timeout${C_OFF} — killing ${NAMES[$i]} after $(fmt_duration $task_elapsed)"
                    # Kill the whole process group: gcloud spawns children that
                    # would otherwise outlive the parent and keep running.
                    kill -TERM "-${PIDS[$i]}" 2>/dev/null || kill -TERM "${PIDS[$i]}" 2>/dev/null
                    continue
                fi
                status_line="$status_line ${NAMES[$i]}=$(fmt_duration $task_elapsed)"
            elif [ "${DONE[$i]}" = false ]; then
                # Stamp the finish time here, not after the whole loop exits —
                # otherwise every task inherits the slowest one's duration.
                DONE[$i]=true
                ELAPSED[$i]=$((now - ${STARTS[$i]}))
            fi
        done

        [ "$running" -eq 0 ] && break

        if [ $((now - LAST_PRINT)) -ge "$HEARTBEAT" ]; then
            echo "  [$(fmt_duration $((now - OVERALL_START)))] $running running:$status_line"
            LAST_PRINT=$now
        fi
        sleep "$POLL"
    done

    for i in $(seq 0 $((TASK_COUNT - 1))); do
        wait "${PIDS[$i]}"
        CODES[$i]=$?
    done
fi

# --- summary ----------------------------------------------------------------
echo ""
echo "${C_BOLD}=== Setup Summary ===${C_OFF}"
printf "%-20s %-10s %-10s %s\n" "RESOURCE" "STATUS" "DURATION" "LOG"
printf "%-20s %-10s %-10s %s\n" "--------" "------" "--------" "---"

FAILED_COUNT=0
for i in $(seq 0 $((TASK_COUNT - 1))); do
    if [ "${CODES[$i]}" -eq 0 ]; then
        status="${C_GREEN}SUCCESS${C_OFF}"
    else
        status="${C_RED}FAILED${C_OFF}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    # printf pads before the color codes are counted, so pad the plain word and
    # colorize separately to keep columns aligned.
    printf "%-20s " "${NAMES[$i]}"
    if [ "${CODES[$i]}" -eq 0 ]; then
        printf "%s%-10s%s" "$C_GREEN" "SUCCESS" "$C_OFF"
    else
        printf "%s%-10s%s" "$C_RED" "FAILED($((CODES[$i])))" "$C_OFF"
    fi
    printf " %-10s %s\n" "$(fmt_duration ${ELAPSED[$i]})" "${LOGS[$i]}"
done

# Anything skipped is reported too — a summary that silently omits a target
# reads as "all three succeeded" when only one ran.
for pair in "artifact-registry:$SKIP_ARTIFACT_REGISTRY" \
            "feature-store:$SKIP_FEATURE_STORE" \
            "composer:$SKIP_COMPOSER"; do
    name="${pair%%:*}"
    skipped="${pair##*:}"
    if [ "$skipped" = true ]; then
        printf "%-20s " "$name"
        printf "%s%-10s%s" "$C_YELLOW" "SKIPPED" "$C_OFF"
        printf " %-10s %s\n" "-" "-"
    fi
done

echo ""
if [ "$FAILED_COUNT" -eq 0 ]; then
    echo "${C_GREEN}All $TASK_COUNT target(s) completed successfully.${C_OFF}"
    if [ "$SKIP_COMPOSER" = false ]; then
        echo ""
        echo "Next: ./scripts/sync_dags.sh   # upload DAGs to the Composer bucket"
    fi
    exit 0
else
    echo "${C_RED}$FAILED_COUNT of $TASK_COUNT target(s) failed.${C_OFF}"
    echo ""
    echo "Last 20 lines of each failed log:"
    for i in $(seq 0 $((TASK_COUNT - 1))); do
        if [ "${CODES[$i]}" -ne 0 ]; then
            echo ""
            echo "--- ${NAMES[$i]} (${LOGS[$i]}) ---"
            tail -20 "${LOGS[$i]}" 2>/dev/null || echo "(no log output)"
        fi
    done
    echo ""
    echo "Re-run just the failures with the matching --skip-* flags for the others."
    exit 1
fi
