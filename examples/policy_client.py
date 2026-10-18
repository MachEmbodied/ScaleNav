"""Send a single observation to a local AWM policy; does not control hardware."""
import argparse
import base64
import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


def post(base_url, endpoint, payload):
    request = Request(base_url.rstrip('/') + endpoint,
                      data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with build_opener(ProxyHandler({})).open(request, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8620')
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--chunk', action='store_true', help='Request an action chunk instead of one action')
    args = parser.parse_args()
    print(json.dumps(post(args.url, '/reset', {'instruction': args.prompt}), indent=2))
    rgb = base64.b64encode(args.image.read_bytes()).decode('ascii')
    print(json.dumps(post(args.url, '/act_chunk' if args.chunk else '/act', {'rgb_b64': rgb}), indent=2))


if __name__ == '__main__':
    main()
