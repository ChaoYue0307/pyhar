# Releasing

pyhar publishes to PyPI as **`pyhar-agents`** (import name `pyhar`). Two ways to
cut a release: **automated** (recommended — tag a commit, CI publishes) or
**manual** (`build` + `twine`).

---

## 0. Pre-flight (both paths)

Bump the version in **two** places (keep them identical):

- `pyproject.toml` → `version = "X.Y.Z"`
- `src/pyhar/__init__.py` → `__version__ = "X.Y.Z"`

Then make sure the tree is green and the changelog is updated:

```bash
pip install -e ".[dev]"
ruff check src tests
mypy src
pytest -q
# add a "## [X.Y.Z] — <date>" section to CHANGELOG.md
git commit -am "Release X.Y.Z"
git push origin main
```

---

## 1. Automated release (recommended)

Uses PyPI **Trusted Publishing** (OpenID Connect) — no API tokens to store. The
workflow at `.github/workflows/release.yml` builds and publishes whenever you
push a `vX.Y.Z` tag.

**One-time setup on PyPI** (per project):

1. Sign in at <https://pypi.org>. Because the project doesn't exist yet, add a
   **pending publisher**: your account → *Publishing* → *Add a pending publisher*.
2. Fill in:
   - PyPI Project Name: `pyhar-agents`
   - Owner: `ChaoYue0307`
   - Repository name: `pyhar`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
The GitHub side is **already configured**: a `pypi` environment exists with a
required-reviewer gate (owner `ChaoYue0307`), so every publish waits for a
one-click approval before it runs. Adjust reviewers under repo *Settings →
Environments → pypi* if you want.

**Every release after that:**

```bash
git tag v0.3.0
git push origin v0.3.0
```

CI builds the sdist + wheel, then pauses for approval. **Approve the deployment**
under the repo's **Actions** tab (the run for the tag) and it publishes. Then
create a GitHub Release for the tag (optional but nice):

```bash
gh release create v0.3.0 --title "v0.3.0" --notes-from-tag
```

---

## 2. Manual release (`build` + `twine`)

If you'd rather publish from your machine:

```bash
pip install -e ".[dev]"          # includes build + twine
rm -rf dist/                     # start clean
python -m build                  # -> dist/pyhar_agents-X.Y.Z{.tar.gz, -py3-none-any.whl}
twine check dist/*               # validate metadata/README rendering
```

Dry-run on **TestPyPI** first (recommended for a first publish):

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ pyhar-agents
python -c "import pyhar; print(pyhar.__version__)"
```

Then the real thing:

```bash
twine upload dist/*
```

`twine` will prompt for credentials — use a [PyPI API token](https://pypi.org/help/#apitoken)
(username `__token__`, password `pypi-…`). Store it in `~/.pypirc` or the
`TWINE_USERNAME` / `TWINE_PASSWORD` env vars.

Finally tag the release:

```bash
git tag v0.3.0 && git push origin v0.3.0
```

---

## Verify the published package

```bash
pip install pyhar-agents==X.Y.Z
python -c "import pyhar; print(pyhar.__version__)"
```

## Notes

- **Versions are immutable on PyPI.** You can't re-upload the same version — bump
  the patch number if you need to fix a bad upload (and `yank` the bad one).
- `dist/`, `build/`, and `*.egg-info/` are git-ignored; never commit build output.
- The distribution name (`pyhar-agents`) and import name (`pyhar`) differ on
  purpose — don't "fix" one to match the other.
