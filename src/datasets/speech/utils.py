import os
import subprocess
import tarfile
import zipfile
from typing import Optional


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    remove_finished: bool = False,
):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    print('Extracting {} to {}'.format(archive, extract_root))
    archive = os.path.join(download_root, filename)
    p = subprocess.Popen(['wget', url, '-O', archive])
    p.wait()

    try:
        if filename.endswith('.tgz') or filename.endswith('.tar.gz'):
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall(path=extract_root)
        if filename.endswith('.zip'):
            with zipfile.ZipFile(archive, 'r') as z:
                z.extractall(extract_root)
        if remove_finished:
            os.remove(archive)
    except:  # noqa E722
        return
