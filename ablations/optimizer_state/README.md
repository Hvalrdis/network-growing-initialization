# Optimizer-state ablation

```bash
python3 -m unified_experiments.ablations.optimizer_state.main --show-plan
python3 -m unified_experiments.ablations.optimizer_state.main \
  --optimizer sgd --initialization-mode a --target-width 128-256-512
```

Defaults reproduce the full Appendix-B grid. Results are grouped by optimizer,
initialization mode, and target width. Every case writes per-epoch TSV files plus
`summary.csv` and `summary.json`; final accuracy is averaged over the last five
epochs of each run before mean/std aggregation.

The `Rebuild Optimizer & Restart Scheduler` condition is available through
`--include-scheduler-restart` and is not part of the default manuscript table.
