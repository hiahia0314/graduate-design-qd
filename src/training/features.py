"""Same-day industry residuals, with strict historical classification coverage."""
import numpy as np
import pandas as pd
from src.factors.computation import compute_factor, dated


def read_industries(path):
    frame = pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path, dtype={'symbol': str})
    frame = dated(frame, ['date', 'symbol'])
    if 'industry' not in frame or frame.industry.isna().any() or frame.industry.astype(str).str.strip().eq('').any():
        raise ValueError('Historical industry data requires nonempty date,symbol,industry')
    return frame[['date', 'symbol', 'industry']]


def attach_industries(frame, industries):
    result = dated(frame, ['date', 'symbol']).merge(
        dated(industries, ['date', 'symbol']), on=['date', 'symbol'], how='left', validate='one_to_one')
    if result.industry.isna().any() or result.industry.astype(str).str.strip().eq('').any():
        raise ValueError('Missing historical industry for date/symbol; no current-industry backfill allowed')
    return result


def neutralize(frame, columns):
    """Industry dummy regression residual == subtract within-industry mean.

    Preserve NaN; group statistics only use today's observed members. No labels
    or future dates enter the transform. Single-member groups have zero residual.
    """
    result = frame.copy()
    values = result[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    result[columns] = values - values.groupby([result.date, result.industry]).transform('mean')
    return result


def prepare_industries(dataset, output):
    """Extract dated level-one classifications from hash-verified source panels.

    Join by original security code and date to the frozen historical membership;
    never substitute the security-master snapshot or fill missing classifications.
    Only train and validation sources are opened.
    """
    import json
    from pathlib import Path
    from src.research.dataset import PreparedSplit, digest
    root = Path(dataset)
    manifest = json.loads((root / 'manifest.json').read_text())
    members = pd.concat([pd.read_parquet(PreparedSplit(root, s).path / 'membership.parquet')
                         for s in ('train', 'validation')], ignore_index=True)
    members['date'] = pd.to_datetime(members.date)
    members['code'] = members.code.astype(str).str.zfill(6)
    frames, hashes = [], {}
    for year in sorted(members.date.dt.year.unique()):
        matches = [Path(p) for p in manifest['source_hashes'] if Path(p).name == f'{year}.parquet']
        if len(matches) != 1:
            raise ValueError(f'Need one recorded annual source for {year}')
        path = matches[0]
        checksum = digest(path)
        if checksum != manifest['source_hashes'][str(path)]:
            raise ValueError(f'Annual source hash mismatch: {path}')
        raw = pd.read_parquet(path, columns=['date', 'code', 'industry_level1_code'])
        raw['date'] = pd.to_datetime(raw.date.astype(str), format='%Y%m%d')
        raw['code'] = raw.code.astype(str).str.zfill(6)
        frames.append(raw.rename(columns={'industry_level1_code': 'industry'}))
        hashes[str(path)] = checksum
    raw = pd.concat(frames, ignore_index=True)
    result = members[['date', 'code', 'symbol']].merge(raw, on=['date', 'code'], how='left', validate='one_to_one')
    if result.industry.isna().any() or result.industry.astype(str).str.strip().eq('').any():
        raise ValueError(f'Missing dated industry for {int(result.industry.isna().sum())} historical member rows')
    result = dated(result[['date', 'symbol', 'industry']], ['date', 'symbol'])
    output = Path(output)
    result.to_parquet(output, index=False)
    audit = {'source_column': 'industry_level1_code', 'source_hashes': hashes,
             'rows': len(result), 'missing': 0, 'splits': ['train', 'validation'],
             'max_daily_industries': int(result.groupby('date').industry.nunique().max()),
             'policy': 'Use provider dated classifications as supplied; no snapshot substitution or fill',
             'output_sha256': digest(output)}
    output.with_suffix('.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    return output
