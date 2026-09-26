"""Verify one released archived model without training or remote operations."""
import argparse
import json
from pathlib import Path

import yaml

from pdeobs.archived_verification import require_hash, verify_archived_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'models-manifest', 'model-root', 'data-root',
                 'data-manifest', 'campaign', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--release-map', type=Path,
                        default=Path(__file__).resolve().parents[1]/'results/public_deposits/release_map.json',
                        help='map from the archived digests to the published deposits (this repository)')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; never overwrite verification.')
    row = json.loads(args.index.read_text())['records'][args.identity]
    candidates = [r for r in json.loads(args.models_manifest.read_text())['rows']
                  if r['identity'] == args.identity]
    if len(candidates) != 1:
        raise ValueError('Exactly one matching released model is required.')
    summary_expected = row['dataset_summary_sha256']
    release_map = json.loads(args.release_map.read_text()) if args.release_map.exists() else {}
    # The published corpus is de-identified: its summary.json carries the same content with the
    # provenance fields replaced.  The map from the archived digest to the published one lives in this
    # repository; the public scrub manifest lists published digests only.
    summary_map = release_map.get('dataset', {}).get('summary.json', {})
    if summary_map.get('original_sha256') == summary_expected:
        summary_expected = summary_map['sha256']
    else:
        scrub = args.data_root/'scrub-manifest.json'
        if scrub.exists():  # first-generation scrub manifests carried the mapping themselves
            mapping = json.loads(scrub.read_text()).get('summary_files', {}).get('summary.json', {})
            if mapping.get('original_sha256') == summary_expected:
                summary_expected = mapping['scrubbed_sha256']
    require_hash(args.data_root/'summary.json', summary_expected)
    campaign = yaml.safe_load(args.campaign.read_text())
    problem = campaign['problem_settings'][row['pde']]
    prefix = '/'.join((row['pde'], problem['boundary'], problem['setting']))+'/'
    shards = []
    for entry in json.loads(args.data_manifest.read_text())['files']:
        if entry['path'].startswith(prefix) and entry['path'].endswith('.h5'):
            path = args.data_root/entry['path']
            require_hash(path, entry['sha256'])
            stat = path.stat()
            shards.append({'path':entry['path'], 'sha256':entry['sha256'],
                           'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns})
    if len(shards) != 12 or len({s['path'] for s in shards}) != 12:
        raise ValueError('Expected 12 distinct shards for the archived macrodomain.')
    result = verify_archived_row(row=row, released=candidates[0],
        model_root=args.model_root, data_root=args.data_root,
        data_manifest={'shards':shards, 'original_summary_sha256':row['dataset_summary_sha256']},
        campaign=campaign, output=args.output, device=args.device,
        release_map=release_map.get('models', {}).get(args.identity))
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
