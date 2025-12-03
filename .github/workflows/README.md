# GitHub Actions Workflows

## Publishing to PyPI

The `publish.yml` workflow automatically publishes new releases to PyPI when you create a GitHub release.

### Setup Instructions

#### 1. Configure PyPI Trusted Publishing

1. Go to https://pypi.org/manage/account/publishing/
2. Scroll to "Pending publishers" or "Add a new pending publisher"
3. Fill in the form:
   - **PyPI Project Name**: `hearken`
   - **Owner**: `HipsterBrown` (your GitHub username)
   - **Repository name**: `hearken`
   - **Workflow name**: `publish.yml`
   - **Environment name**: (leave empty)
4. Click "Add"

**Note**: You can set this up before the package exists on PyPI. The first release will create the package automatically.

#### 2. Create a Release

To publish a new version:

1. Update version in `pyproject.toml` and `hearken/__init__.py`
2. Commit and push:
   ```bash
   git add pyproject.toml hearken/__init__.py
   git commit -m "chore: bump version to vX.Y.Z"
   git push
   ```
3. Create and push a git tag:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z: Brief description"
   git push --tags
   ```
4. Go to https://github.com/HipsterBrown/hearken/releases/new
5. Select your tag
6. Add release notes
7. Click "Publish release"

The workflow will automatically:
- Run tests on Python 3.11, 3.12, and 3.13
- Run type checking with mypy
- Run linting with ruff
- Build the package
- Publish to PyPI (if all tests pass)

### Workflow Triggers

- **On Release Published**: Runs when you publish a GitHub release
- Tests must pass before publishing
- Uses PyPI trusted publishing (no API tokens needed)

### Manual Build (for testing)

To test the build locally:

```bash
# Install build tools
uv build

# Check the distribution
ls dist/
```

This creates:
- `dist/hearken-X.Y.Z.tar.gz` (source distribution)
- `dist/hearken-X.Y.Z-py3-none-any.whl` (wheel)

### Troubleshooting

**"Invalid publisher" error:**
- Verify the trusted publisher is configured correctly on PyPI
- Ensure the workflow name exactly matches: `publish.yml`
- Check that the repository owner and name are correct

**Tests failing:**
- The workflow requires all tests to pass before publishing
- Check the Actions tab for detailed error logs
- Fix issues and create a new release

**Version already exists:**
- PyPI does not allow re-uploading the same version
- Increment the version number in `pyproject.toml`
- Create a new tag and release
