#!/usr/bin/env bash
# Prepare a new machine for the all-sports stream recorder (sports-stream.yml) and the nightly data job
# (other-data.yml), then register both runners on it (the owner, 26 Nov 2026; docs/RUNNERS.md).
#
#   curl -fsSLO https://raw.githubusercontent.com/tomwilson41986/ashcroft/claude/scrape-horse-racing-results-vaHKc/scripts/setup_stream_machine.sh
#   bash setup_stream_machine.sh
#
# Run it as the machine's ordinary user with sudo (on AWS Ubuntu: 'ubuntu'), not root. It asks for a GitHub runner
# registration token once per runner (GitHub > the repository > Settings > Actions > Runners > New self-hosted runner;
# one token, valid an hour, registers both).
#
# What it does, each step safe to run again:
#   - the packages the runners need (curl, git, python3-venv);
#   - the clock on UTC and kept in sync (chrony);
#   - a 2 GB swap file if the machine has no swap (a small machine survives a pip install);
#   - a check that Betfair's login and stream hosts answer from here;
#   - scripts/setup_runner.sh for 'streams' and for 'data', each its own service with only its own label.
# It installs nothing that bets and stores no secret: the Betfair and AWS credentials stay in GitHub's secrets.
set -euo pipefail

BRANCH="${BRANCH:-claude/scrape-horse-racing-results-vaHKc}"
RAW="https://raw.githubusercontent.com/tomwilson41986/ashcroft/$BRANCH/scripts/setup_runner.sh"
LABELS="${LABELS:-streams data}"

if [ "$(id -u)" = 0 ]; then
  echo "run this as the machine's ordinary user (e.g. ubuntu), not root" >&2; exit 2
fi
[ "$(uname -m)" = x86_64 ] || { echo "x86_64 only (setup_runner.sh checks the x64 download)" >&2; exit 2; }

echo "== Packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl git python3-venv chrony >/dev/null

echo "== Clock: UTC, synced"
sudo timedatectl set-timezone UTC
sudo systemctl enable --now chrony >/dev/null 2>&1 || true
date -u

echo "== Swap"
if [ "$(swapon --show --noheadings | wc -l)" = 0 ]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
free -h | sed -n 1,3p

echo "== Betfair reachable from here (no login, just the hosts)"
for url in https://identitysso.betfair.com https://api.betfair.com/exchange/betting/json-rpc/v1; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url" || echo 000)
  echo "$url -> HTTP $code"
  [ "$code" != 000 ] || { echo "cannot reach $url from this machine" >&2; exit 1; }
done
if timeout 10 bash -c 'exec 3<>/dev/tcp/stream-api.betfair.com/443' 2>/dev/null; then
  echo "stream-api.betfair.com:443 -> open"
else
  echo "cannot open stream-api.betfair.com:443 from this machine" >&2; exit 1
fi

echo "== Runners: $LABELS"
curl -fsSL -o "$HOME/setup_runner.sh" "$RAW"
for label in $LABELS; do
  if [ -d "$HOME/actions-runner-$label" ]; then
    echo "runner '$label' already set up in ~/actions-runner-$label: left as it is"
  else
    bash "$HOME/setup_runner.sh" "$label"
  fi
done

echo
echo "Done. In GitHub > Settings > Actions > Runners both should show Idle, labelled: $LABELS."
echo "Then: Actions > \"Every sport's main markets on the Exchange Stream\" > Run workflow (hours 0.25 for a first test)."
