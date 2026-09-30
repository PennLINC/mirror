# MIRROR reproducibility guide

This directory is a self-contained [Jupyter Book](https://jupyterbook.org/)
covering MIRROR data curation, derivatives creation, and manuscript analyses.
Lab replication begins at registration and warping, after source metric
processing is complete.

## Build locally

From the repository root:

```bash
python -m venv /tmp/mirror-guide-env
source /tmp/mirror-guide-env/bin/activate
python -m pip install -r reproducibility_guide/requirements.txt
jupyter-book build reproducibility_guide
```

Open `reproducibility_guide/_build/html/0_basic_info.html` in a browser. The generated
`_build/` directory is ignored by `reproducibility_guide/.gitignore`.

## Publish

The workflow at
`.github/workflows/deploy_reproducibility_guide.yml` builds and publishes the
book when guide files are pushed to `main`. It can also be run manually from
the repository's **Actions** tab.

Before the first deployment, a repository administrator must open
**Settings → Pages** and select **GitHub Actions** as the source under **Build
and deployment**. Subsequent pushes that change `reproducibility_guide/` will
update the published site automatically.
