# 项目数据

## 当前真实行情数据：沪深300

活动目录为 `data/raw/akshare/`。下载时间、成分股名单日期、行数和限制以该目录的
`manifest.json` 为准，质量检查结果见 `validation.json`。

| 文件 | 含义 |
| --- | --- |
| `csi300_constituents_current.parquet` | 中证指数发布的当前沪深300成分股快照，代码保留前导零 |
| `csi300_stock_daily_raw.parquet` | 当前成分股2016—2025年不复权日线 |
| `csi300_stock_daily_qfq.parquet` | 同期前复权日线，复权基准为下载时点 |
| `csi300_index_daily_raw.parquet` | 沪深300基准指数，代码000300.SH |
| `stocks/raw/`、`stocks/qfq/` | 按股票拆分的可续传数据 |
| `000300cons.xls` | 官方原始成分股文件 |
| `download_results.csv`、`download_failures.csv` | 下载结果及失败清单 |

股票日线的 `volume` 为股，`amount` 为元，`outstanding_share` 为流通股数，
`turnover` 为比率。腾讯指数接口的 `amount` 实际表示以手计的成交量，因此本项目将其
命名为 `volume_lots`；它不是成交金额，不与股票 `amount` 混用。

**当前快照不等于历史股票池。** 本数据包可用于开发和数据接口验证，不能把300只股票
固定回填为2016—2025年的逐日沪深300成分股；正式研究仍需单独补齐历史有效成员及
调入调出日期。前复权行情也不能直接替代真实成交价格和公司行为记账。

`sample_bars.csv` 是6只虚构股票的单元测试/演示数据，继续保留。两个演示入口仍默认使用
此文件；本次替换原始数据包没有把“当前成分股历史行情”伪装成完整历史回测数据源。

## 重新下载与检查

```bash
.venv/bin/python scripts/download_csi300.py --output data/raw/csi300_staging
.venv/bin/python scripts/verify_csi300.py data/raw/csi300_staging
```

下载器保留已完成的单股文件，失败可以重复运行；默认使用4个下载线程及请求超时。
脚本不会自动覆盖活动目录，需先检查下载状态、质量报告，再归档旧目录并切换。
从全新日期更新成分股快照时应使用新的空输出目录，避免混用不同快照的缓存。
依赖可通过 `pip install -e '.[market-data]'` 安装。

来源：中证指数官方成分文件、AKShare 新浪个股日线、腾讯指数日线。

## 本次下载核查（2026-11-10）

官方成员快照为2026-11-09，共300只；实际行情范围为2016-01-04至2025-12-31。
不复权641,528条，前复权641,527条，基准指数2,430条。两种股票价格数据均覆盖300只股票，
下载失败为0。前复权源缺少 `000002` 在2016-01-14的一条记录；保留这一差异，不静默补齐，
具体日期与股票代码保存在 `validation.json`，原始对应行见 `raw_without_qfq.csv`。

