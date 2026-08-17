# Repository seed checklist

Before using this package as the `main` branch of a new GitHub repository:

- [ ] `./scripts/verify_preproduct.sh` passes.
- [ ] independent change review has no P0/P1 blockers.
- [ ] bootstrap ZIP integrity is verified.
- [ ] Git bundle verifies and contains the expected branch/history.
- [ ] no CUDA speed claim is present unless CUDA was executed.
- [ ] repository visibility/licensing choice is confirmed by the owner.
- [ ] GitHub Actions CPU workflow is enabled.
- [ ] default branch is set intentionally.
- [ ] next development starts from a feature branch.
