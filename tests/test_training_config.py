import json
import pytest
from src.training.ranking import RankConfig
from src.training.pipeline import validation_windows


def test_windows_follow_dataset_fixed_and_rolling():
    plan={'train':{'start':'2017-01-01','end':'2021-12-31'},'validation':{'start':'2022-01-01','end':'2023-12-31'}}
    fixed=validation_windows(plan,RankConfig())
    assert len(fixed)==1 and fixed[0]['train_start']=='2017-01-01' and fixed[0]['train_end']=='2021-12-31'
    assert fixed[0]['evaluation_end']=='2023-12-31'
    rolling=validation_windows(plan,RankConfig(validation_mode='rolling'))
    assert [w['train_end'] for w in rolling]==['2021-12-31','2022-12-31']
    assert [w['train_start'] for w in rolling]==['2019-01-01','2020-01-01']


@pytest.mark.parametrize('kwargs',[{'validation_mode':'random'},{'factor_scope':'bad'},{'input_form':'raw'},
                                   {'market_state':'false'},{'early_stopping':1},{'max_estimators':0}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):RankConfig(**kwargs)


def test_cli_overrides_json_and_has_only_one_pipeline(tmp_path,monkeypatch):
    import train_selection_model as cli
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'validation_mode':'rolling','market_state':True,'max_estimators':123,'factor_scope':'selected'}))
    captured={}
    def run(**kwargs):
        captured.update(kwargs)
        return {'status':'completed','metrics':{}}
    monkeypatch.setattr(cli,'run_training',run)
    assert cli.main(['--config',str(config),'--output','config_test','--rounds','500','--validation-mode','fixed','--no-market-state'])==0
    c=captured['config']
    assert c.validation_mode=='fixed' and c.max_estimators==500 and not c.market_state
    assert c.factor_scope=='selected'
    config.write_text('{"typo":true}')
    with pytest.raises(SystemExit):cli.main(['--config',str(config),'--output','config_test'])
