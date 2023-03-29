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

    archive = os.path.join(download_root, filename)
    p = subprocess.Popen(['wget', url, '-O', archive])
    p.wait()

    try:
        if filename.endswith('.tgz') or filename.endswith('.tar.gz'):
            with tarfile.open(archive, 'r:gz') as tar:

                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tar, path=extract_root)
        if filename.endswith('.zip'):
            with zipfile.ZipFile(archive, 'r') as z:
                z.extractall(extract_root)
        if remove_finished:
            os.remove(archive)
    except:  # noqa E722
        return
