# Continuous integration

`github-actions-tests.yml` runs the test suite on every push. It lives here instead of in
`.github/workflows/` because the GitHub CLI login used to push this repository lacks the
`workflow` permission. To turn it on:

```bash
gh auth refresh -s workflow
mkdir -p .github/workflows
git mv ci/github-actions-tests.yml .github/workflows/tests.yml
git commit -m "Enable GitHub Actions tests" && git push
```
