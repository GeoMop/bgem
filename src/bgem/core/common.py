import os.path
from typing import *
import shutil
from pathlib import Path
import numpy as np
import logging

from .memoize import File

class workdir:
    """
    Context manager for creation and usage of a workspace dir.

    name: the workspace directory
    inputs: list of files and directories to copy into the workspaceand
        TODO: fine a sort of robust ad portable reference
    clean: if true the workspace would be deleted at the end of the context manager.
    TODO: clean_before / clean_after
    TODO: File constructor taking current workdir environment, openning virtually copied files.
    TODO: Workdir would not perform change of working dir, but provides system interface for: subprocess, file openning
          only perform CD just before executing a subprocess also interacts with concept of an executable.
    portable reference and with lazy evaluation. Optional true copy possible.
    """
    CopyArgs = Union[str, Tuple[str, str]]
    def __init__(self, name:str="sandbox", inputs:List[CopyArgs] = None, clean=False):

        if inputs is None:
            inputs = []
        self._inputs = inputs
        self.work_dir = os.path.abspath(name)
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self._clean = clean
        self._orig_dir = os.getcwd()

    def copy(self, src, dest=None):
        """
        :param src: Realtive or absolute path.
        :param dest: Relative path with respect to work dir.
                    Default is the same as the relative source path,
                    for abs path it is the just the last name in the path.
        """
        if isinstance(src, File):
            src = src.path
        if isinstance(dest, File):
            dest = dest.path
        #if dest == ".":
        #    if os.path.isabs(src):
        #        dest = os.path.basename(src)
        #    else:
        #        dest = src
        if dest is None:
            dest = ""
        dest = os.path.join(self.work_dir, dest, os.path.basename(src))
        dest_dir, _ = os.path.split(dest)
        if not os.path.isdir(dest_dir):
            #print(f"MAKE DIR: {dest_dir}")
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
        abs_src = os.path.abspath(src)

        # ensure that we always update the target
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        elif os.path.isfile(dest):
            os.remove(dest)

        # TODO: perform copy, link or redirectio to src during extraction of the File object from dictionary
        # assumes custom tag for file, file_link, file_copy etc.
        if os.path.isdir(src):
            #print(f"COPY DIR: {abs_src} TO DESTINATION: {dest}")
            shutil.copytree(abs_src, dest, dirs_exist_ok=True)
        else:
            try:
                shutil.copy2(abs_src, dest)
            except FileNotFoundError:
                FileNotFoundError(f"COPY FILE: {abs_src} TO DESTINATION: {dest}")

    def __enter__(self):
        for item in self._inputs:
            #print(f"treat workspace item: {item}")
            if isinstance(item, Tuple):
                self.copy(*item)
            else:
                self.copy(item)
        os.chdir(self.work_dir)

        return self.work_dir

    def __exit__(self, type, value, traceback):
        os.chdir(self._orig_dir)
        if self._clean:
            shutil.rmtree(self.work_dir)


def substitute_placeholders(file_in: str, file_out: str, params: Dict[str, Any]):
    """
    In the template `file_in` substitute the placeholders in format '<name>'
    according to the dict `params`. Write the result to `file_out`.
    TODO: set Files into params, in order to compute hash from them.
    TODO: raise for missing value in dictionary
    """
    used_params = []
    files = []
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        if isinstance(value, File):
            files.append(value)
            value = value.path
        placeholder = '<%s>' % name
        n_repl = text.count(placeholder)
        if n_repl > 0:
            used_params.append(name)
            text = text.replace(placeholder, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)

    return File(file_out, files), used_params


# Directory for all flow123d main input templates.
# These are considered part of the software.

# TODO: running with stdout/ stderr capture, test for errors, log but only pass to the main in the case of
# true error


def sample_from_population(n_samples:int, frequency:Union[np.array, int]):
    if type(frequency) is int:
        frequency = np.full(len(frequency), 1, dtype=int)
    else:
        frequency = np.array(frequency, dtype=int)

    cumul_freq = np.cumsum(frequency)
    total_samples = np.sum(frequency)
    samples = np.random.randint(0, total_samples, size=n_samples + 1)
    samples[-1] = total_samples # stopper
    sample_seq = np.sort(samples)
    # put samples into bins given by cumul_freq
    bin_samples = np.empty_like(samples)
    i_sample = 0
    for ifreq, c_freq in enumerate(cumul_freq):

        while sample_seq[i_sample] < c_freq:
            bin_samples[i_sample] = ifreq
            i_sample += 1

    return bin_samples[:-1]
