"""Find and test free proxies that can access promo.betfair.com."""

import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

TEST_URL = "https://promo.betfair.com/betfairsp/prices/dwbfpricesukwin01012024.csv"


def fetch_free_proxies():
    """Fetch free proxy lists from public APIs."""
    proxies = []

    # Source 1: free-proxy-list via proxy-scrape
    sources = [
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=GB&ssl=all&anonymity=all",
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=IE&ssl=all&anonymity=all",
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=all&ssl=all&anonymity=elite",
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=socks5&timeout=5000&country=GB&ssl=all&anonymity=all",
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=socks5&timeout=5000&country=all&ssl=all&anonymity=elite",
    ]

    for url in sources:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                for line in resp.text.strip().split("\n"):
                    line = line.strip()
                    if line and ":" in line:
                        proxies.append(line)
        except Exception as e:
            print(f"  Source failed: {e}")

    # Source 2: geonode
    try:
        resp = requests.get(
            "https://proxylist.geonode.com/api/proxy-list?limit=50&page=1&sort_by=lastChecked&sort_type=desc&country=GB&protocols=http%2Chttps%2Csocks5",
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            for p in data.get("data", []):
                proxies.append(f"{p['ip']}:{p['port']}")
    except Exception as e:
        print(f"  Geonode failed: {e}")

    return list(set(proxies))


def test_proxy(proxy_addr, protocol="http"):
    """Test if a proxy can reach promo.betfair.com without Cloudflare block."""
    proxy_url = f"{protocol}://{proxy_addr}"
    try:
        session = requests.Session()
        session.proxies = {"http": proxy_url, "https": proxy_url}
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        resp = session.get(TEST_URL, timeout=15)

        if resp.status_code == 200 and not resp.text.strip().startswith("<!"):
            return proxy_url, True, f"OK ({len(resp.text)} bytes)"
        elif resp.status_code == 403:
            return proxy_url, False, "403 Cloudflare block"
        else:
            return proxy_url, False, f"HTTP {resp.status_code}"
    except Exception as e:
        return proxy_url, False, str(e)[:60]


def main():
    print("Fetching free proxy lists...")
    raw_proxies = fetch_free_proxies()
    print(f"Found {len(raw_proxies)} proxy candidates")

    if not raw_proxies:
        print("No proxies found. Try providing your own with --proxy or --proxy-file.")
        sys.exit(1)

    # Test first batch (limit to avoid taking too long)
    test_batch = raw_proxies[:100]
    print(f"\nTesting {len(test_batch)} proxies against promo.betfair.com...")

    working = []
    tested = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {}
        for addr in test_batch:
            # Try http for most
            futures[executor.submit(test_proxy, addr, "http")] = addr

        for future in as_completed(futures):
            proxy_url, success, msg = future.result()
            tested += 1
            if success:
                working.append(proxy_url)
                print(f"  ✓ WORKING: {proxy_url} - {msg}")
            if tested % 20 == 0:
                print(f"  Tested {tested}/{len(test_batch)}... ({len(working)} working)")

    print(f"\nResults: {len(working)}/{tested} proxies work")

    if working:
        proxy_file = "proxies.txt"
        with open(proxy_file, "w") as f:
            for p in working:
                f.write(p + "\n")
        print(f"\nWorking proxies saved to {proxy_file}")
        print(f"Run: python -m betfair_prices.download --source promo --proxy-file {proxy_file}")
    else:
        print("\nNo working proxies found.")
        print("Free proxies rarely work against Cloudflare-protected sites.")
        print("Options:")
        print("  1. Run download locally: python -m betfair_prices.download --source promo")
        print("  2. Use a paid residential proxy service")


if __name__ == "__main__":
    main()
