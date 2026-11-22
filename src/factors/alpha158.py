"""Default Alpha158 formula catalog matching microsoft/qlib Alpha158DL.

Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
No Qlib runtime or market data is required to construct candidates.
"""


def alpha158_formulas():
    result = dict(zip(
        ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'],
        ['($close-$open)/$open', '($high-$low)/$open', '($close-$open)/($high-$low+1e-12)',
         '($high-Greater($open,$close))/$open', '($high-Greater($open,$close))/($high-$low+1e-12)',
         '(Less($open,$close)-$low)/$open', '(Less($open,$close)-$low)/($high-$low+1e-12)',
         '(2*$close-$high-$low)/$open', '(2*$close-$high-$low)/($high-$low+1e-12)']))
    result.update({f'{f.upper()}0': f'${f}/$close' for f in ['open', 'high', 'low', 'vwap']})
    templates = {
        'ROC': 'Ref($close,{d})/$close', 'MA': 'Mean($close,{d})/$close',
        'STD': 'Std($close,{d})/$close', 'BETA': 'Slope($close,{d})/$close',
        'RSQR': 'Rsquare($close,{d})', 'RESI': 'Resi($close,{d})/$close',
        'MAX': 'Max($high,{d})/$close', 'MIN': 'Min($low,{d})/$close',
        'QTLU': 'Quantile($close,{d},0.8)/$close', 'QTLD': 'Quantile($close,{d},0.2)/$close',
        'RANK': 'Rank($close,{d})',
        'RSV': '($close-Min($low,{d}))/(Max($high,{d})-Min($low,{d})+1e-12)',
        'IMAX': 'IdxMax($high,{d})/{d}', 'IMIN': 'IdxMin($low,{d})/{d}',
        'IMXD': '(IdxMax($high,{d})-IdxMin($low,{d}))/{d}',
        'CORR': 'Corr($close,Log($volume+1),{d})',
        'CORD': 'Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),{d})',
        'CNTP': 'Mean($close>Ref($close,1),{d})', 'CNTN': 'Mean($close<Ref($close,1),{d})',
        'CNTD': 'Mean($close>Ref($close,1),{d})-Mean($close<Ref($close,1),{d})',
        'VMA': 'Mean($volume,{d})/($volume+1e-12)', 'VSTD': 'Std($volume,{d})/($volume+1e-12)',
        'WVMA': 'Std(Abs($close/Ref($close,1)-1)*$volume,{d})/(Mean(Abs($close/Ref($close,1)-1)*$volume,{d})+1e-12)',
    }
    for prefix, field in [('', 'close'), ('V', 'volume')]:
        up = f'Sum(Greater(${field}-Ref(${field},1),0),{{d}})'
        down = f'Sum(Greater(Ref(${field},1)-${field},0),{{d}})'
        denominator = f'(Sum(Abs(${field}-Ref(${field},1)),{{d}})+1e-12)'
        templates[prefix+'SUMP'] = up+'/'+denominator
        templates[prefix+'SUMN'] = down+'/'+denominator
        templates[prefix+'SUMD'] = '('+up+'-'+down+')/'+denominator
    for name, template in templates.items():
        for window in [5, 10, 20, 30, 60]:
            result[f'{name}{window}'] = template.format(d=window)
    assert len(result) == 158
    return result
