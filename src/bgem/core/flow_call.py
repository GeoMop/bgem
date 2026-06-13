from typing import *
import logging
import os
import attrs
from . import dotdict, memoize, File, report, substitute_placeholders, workdir
import subprocess
from pathlib import Path
import yaml

def search_file(basename, extensions):
    """
    Return first found file or None.
    """
    if type(extensions) is str:
        extensions = (extensions,)
    for ext in extensions:
        if os.path.isfile(basename + ext):
            return File(basename + ext)
    return None

class EquationOutput:
    def __init__(self, eq_name, balance_name):
        self.eq_name: str = eq_name
        self.spatial_file: File = search_file(eq_name+"_fields", (".msh", ".pvd"))
        self.balance_file: File = search_file(balance_name+"_balance", ".txt"),
        self.observe_file: File = search_file(eq_name+"_observe", ".yaml")

    def _load_yaml_output(self, file, basename):
        if file is None:
            raise FileNotFoundError(f"Not found Flow123d output file: {self.eq_name}_{basename}.yaml.")
        with open(file.path, "r") as f:
            loaded_yaml = yaml.load(f, yaml.CSafeLoader)
        return dotdict.create(loaded_yaml)

    def observe_dict(self):
        return self._load_yaml_output(self.observe_file, 'observe')

    def balance_dict(self):
        return self._load_yaml_output(self.balance_file, 'balance')

    def balance_df(self):
        """
        create a dataframe for the Balance file
        rows for times, columns are tuple (region, value),
        values =[ flux,  flux_in,  flux_out,  mass,  source,  source_in,  source_out,  flux_increment,  source_increment,  flux_cumulative,  source_cumulative,  error ]
        :return:
        TODO: ...
        """
        dict = self.balance_dict()
        pass



class FlowOutput:

    def __init__(self, process: subprocess.CompletedProcess, stdout: File, stderr: File, output_dir="output"):
        self.process = process
        self.stdout = stdout
        self.stderr = stderr
        #with workdir(output_dir):
        self.log = File("flow123.0.log")
        # TODO: flow ver 4.0 unify output file names
        self.hydro = EquationOutput("flow", "water")
        self.solute = EquationOutput("solute", "mass")
        self.mechanic = EquationOutput("mechanics", "mechanics")

    @property
    def success(self):
        return self.process.returncode == 0

    def check_conv_reasons(self):
        """
        Check correct convergence of the solver.
        Reports the divergence reason and returns false in case of divergence.
        """
        with open(self.log.path, "r") as f:
            for line in f:
                tokens = line.split(" ")
                try:
                    i = tokens.index('convergence')
                    if tokens[i + 1] == 'reason':
                        value = tokens[i + 2].rstrip(",")
                        conv_reason = int(value)
                        if conv_reason < 0:
                            print("Failed to converge: ", conv_reason)
                            return False
                except ValueError:
                    continue
        return True


#@memoize
def _prepare_inputs(file_in, params):
    import pathlib
    in_dir, template = os.path.split(file_in)

    root = pathlib.Path(template).with_suffix(".yaml")
    try:
        root = pathlib.Path(root).with_suffix("_tmpl")
    except:
        pass

    root = str(root)
    #root = template.removesuffix(".yaml").removesuffix("_tmpl")
    template_path = Path(file_in).rename(Path(in_dir) / (root + "_tmpl.yaml"))
    #suffix = "_tmpl.yaml"
    #assert template[-len(suffix):] == suffix
    #filebase = template[:-len(suffix)]
    main_input = Path(in_dir) / (root + ".yaml")
    main_input, used_params =  substitute_placeholders(str(template_path), str(main_input), params)
    return main_input


#@memoize
def _flow_subprocess(arguments, main_input):
    filebase, ext = os.path.splitext(os.path.basename(main_input.path))
    arguments.append(main_input.path)
    logging.info("Running Flow123d: " + " ".join(arguments))

    stdout_path = filebase + "_stdout"
    stderr_path = filebase + "_stderr"
    with open(stdout_path, "w") as stdout:
        with open(stderr_path, "w") as stderr:
            print("Call: ", ' '.join(arguments))
            completed = subprocess.run(arguments, stdout=stdout, stderr=stderr)
    return File(stdout_path), File(stderr_path), completed

#@report
#@memoize
def call_flow(cfg:'dotdict', file_in:File, params: Dict[str,str]) -> FlowOutput:
    """
    Run Flow123d in actual work dir with main input given be given template and dictionary of parameters.

    1. prepare the main input file from filebase_in + "_tmpl.yamlL"
    2. run Flow123d

    TODO: pass only flow configuration
    """
    main_input = _prepare_inputs(file_in, params)
    stdout, stderr, completed = _flow_subprocess(cfg.flow_executable.copy(), main_input)
    logging.info(f"Exit status: {completed.returncode}")
    if completed.returncode != 0:
        with open(stderr.path, "r") as stderr:
            print(stderr.read())
        raise Exception("Flow123d ended with error")

    fo = FlowOutput(completed, stdout.path, stderr.path)
    conv_check = fo.check_conv_reasons()
    logging.info(f"converged: {conv_check}")
    return fo

# TODO:
# - call_flow variant with creating dir, copy,


