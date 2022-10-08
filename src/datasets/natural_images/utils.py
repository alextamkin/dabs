import os
import tarfile
import zipfile
from typing import Optional

import torchvision.datasets.utils as utils


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    '''Updated method for handling *.tgz file types

    Modified from Pytorch torchvision dataset utility script:
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py
    '''

    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    utils.download_url(url, download_root, filename)

    archive = os.path.join(download_root, filename)
    print('Extracting {} to {}'.format(archive, extract_root))

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
