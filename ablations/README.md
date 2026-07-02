# Ablations

- `vit_growth_axis`: Appendix A / Table 4, comparing fixed-head Grow-d with
  fixed-per-head-dimension Grow-H.
- `optimizer_state`: Appendix B / Table 5, comparing post-growth optimizer-state
  handling while keeping the learning-rate schedule continuous.

Each task has its own module entry point and writes beneath
`outputs_compare/ablations/<task>/` by default.

