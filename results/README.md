# Results Directory

Benchmark and analysis outputs are grouped by benchmark, then by output type.

```text
results/
  browsecomp/
    fixtures/        Raw BrowseComp fixture/smoke benchmark JSON outputs
    regret/          Oracle/regret reports derived from BrowseComp outputs
  finrate/
    fixtures/        Raw Fin-RATE fixture/smoke benchmark JSON outputs
    regret/          Oracle/regret reports derived from Fin-RATE outputs
  workbench/
    fixtures/        Small WorkBench smoke outputs
    sweeps/          Larger WorkBench benchmark sweeps
    regret/          Oracle/regret reports derived from WorkBench outputs
    router_analysis/ Router mismatch reports
  pcab_legacy/
    regret/          Legacy PCAB/pilot regret reports
```

Future benchmark runners should write timestamped outputs into these folders.
