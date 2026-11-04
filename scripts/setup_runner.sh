#!/usr/bin/env bash
# Register one more self-hosted GitHub Actions runner on the UK server, beside the live horse trader's runner.
#
#   bash scripts/setup_runner.sh football        # football-trade.yml  (runs-on: vars.FOOTBALL_RUNNER || 'football')
#   bash scripts/setup_runner.sh paper-models    # paper-models.yml, greyhound-prices.yml (vars.PAPER_MODELS_RUNNER || 'paper-models')
#   bash scripts/setup_runner.sh data            # other-data.yml      (runs-on: vars.DATA_RUNNER || 'data')
#   bash scripts/setup_runner.sh morning         # morning-models.yml  (runs-on: vars.MORNING_RUNNER || 'morning')
#
# Run it on the server as the user the live runner runs as (not root). It asks for a registration token: GitHub, the
# repository, Settings > Actions > Runners > New self-hosted runner, the token in the "Configure" step (valid one hour).
# The token is read from the prompt, never from the command line, so it stays out of the shell history.
#
# What it does:
#   - one directory a runner: ~/actions-runner-<label>; refuses if it already exists (no runner is overwritten);
#   - the runner pinned to v2.338.2 (released 26 Oct 2026), the download checked against GitHub's published SHA-256;
#   - registered with ONLY its own label (--no-default-labels): no 'self-hosted' label, so it can never pick up a job
#     meant for the live horse trader's runner, whatever vars.BETFAIR_RUNNER names;
#   - installed and started as its own systemd service, so it survives reboots and runs beside the live runner.
# It changes nothing of the live runner, its directory or its service.
set -euo pipefail

LABEL="${1:-}"
case "$LABEL" in
  football|paper-models|data|morning) ;;
  *) echo "usage: bash scripts/setup_runner.sh football|paper-models|data|morning" >&2; exit 2 ;;
esac

REPO_URL="https://github.com/tomwilson41986/ashcroft"
VERSION="2.338.2"
case "$(uname -m)" in
  x86_64)  ARCH=x64;   SHA="baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40" ;;
  aarch64) ARCH=arm64; SHA="" ;;   # not checked here: take the SHA-256 from GitHub's release page
  *) echo "unsupported machine $(uname -m)" >&2; exit 2 ;;
esac
DIR="$HOME/actions-runner-$LABEL"
TARBALL="actions-runner-linux-$ARCH-$VERSION.tar.gz"

if [ "$(id -u)" = 0 ]; then
  echo "run this as the live runner's user, not root (the service is installed with sudo below)" >&2; exit 2
fi
if [ -e "$DIR" ]; then
  echo "$DIR already exists: a runner is set up there. To redo it: cd $DIR && sudo ./svc.sh stop &&" \
       "sudo ./svc.sh uninstall && ./config.sh remove, then delete the folder" >&2
  exit 2
fi

echo "== Prerequisites"
for c in curl tar python3 sudo; do
  command -v "$c" >/dev/null || { echo "missing: $c" >&2; exit 2; }
done
python3 -c 'import venv' 2>/dev/null || { echo "missing: python3-venv (sudo apt-get install -y python3-venv)" >&2; exit 2; }
echo "Free memory and disk:"; free -h | sed -n 1,2p; df -h "$HOME" | tail -1

echo "== Runner $VERSION ($ARCH) into $DIR"
mkdir -p "$DIR"
cd "$DIR"
curl -fsSL -o "$TARBALL" "https://github.com/actions/runner/releases/download/v$VERSION/$TARBALL"
if [ -n "$SHA" ]; then
  echo "$SHA  $TARBALL" | sha256sum -c -
else
  echo "SHA-256 of the download (compare with the release page before going on):"; sha256sum "$TARBALL"
  read -r -p "Matches? [y/N] " ok; [ "$ok" = y ] || { echo "stopped"; exit 1; }
fi
tar xzf "$TARBALL"
rm -f "$TARBALL"
if ldd ./bin/Runner.Listener 2>/dev/null | grep -q "not found"; then
  echo "Installing the runner's system libraries"; sudo ./bin/installdependencies.sh
fi

echo "== Register (label: $LABEL only)"
read -r -s -p "Registration token (from GitHub's New self-hosted runner page): " TOKEN; echo
[ -n "$TOKEN" ] || { echo "no token" >&2; exit 2; }
./config.sh --unattended --url "$REPO_URL" --token "$TOKEN" \
  --name "$(hostname -s)-$LABEL" --labels "$LABEL" --no-default-labels --work _work
unset TOKEN

echo "== Service"
sudo ./svc.sh install "$(id -un)"
sudo ./svc.sh start
sudo ./svc.sh status | sed -n 1,12p

echo
echo "Done: runner '$(hostname -s)-$LABEL' with label '$LABEL'. GitHub > Settings > Actions > Runners should show it Idle."
