"""
Constants for the GCR image of KFP container components.
"""

import os
from pathlib import Path


def _image_tag() -> str:
    """
    Returns the image tag for the KFP component.

    If a revision file exists, it reads and returns the content of the file.
    The revision file is created by the CICD when publishing the package.

    Otherwise, it calls the :func:`_local_git_revision` function to get the local git revision.

    :returns: The image tag for the component.
    :rtype: str
    """
    revision_filepath = Path(__file__).parent.parent.joinpath("REVISION")
    if revision_filepath.exists():
        with revision_filepath.open() as f:
            return f.read().strip()
    else:
        return _local_git_revision()


def _local_git_revision() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )  # noqa: S603
    except subprocess.CalledProcessError as e:
        print(e.output)
        return os.environ.get("BUILDKITE_COMMIT", "main")


GCR_IMAGE_TAG = _image_tag()
BASE_IMAGE = (
    f"gcr.io/shopify-docker-images/apps/ci/shop-promise-modeling:{GCR_IMAGE_TAG}"
)
