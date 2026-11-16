"""Audit and prepare chronological research datasets without fitting any model."""
import argparse
import json
from pathlib import Path
from src.research.dataset import prepare_dataset

ROOT = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config/dataset_splits.json')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    report=prepare_dataset(args.config,ROOT)
    print(json.dumps({'status':report['status'],'calendar':report.get('calendar', {}),'blockers':report['blockers']},indent=2,ensure_ascii=False))
    return 0 if report['status']=='ready' else 2


if __name__=='__main__':
    raise SystemExit(main())
