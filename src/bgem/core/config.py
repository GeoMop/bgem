from dataclasses import dataclass
from typing import *

import os
import yaml
import re
from socket import gethostname
from glob import iglob

from yamlinclude import YamlIncludeConstructor
from yamlinclude.constructor import WILDCARDS_REGEX, get_reader_class_by_name


class YamlLimitedSafeLoader(type):
    """Meta YAML loader that skips the resolution of the specified YAML tags."""
    def __new__(cls, name, bases, namespace, do_not_resolve: List[str]) -> Type[yaml.SafeLoader]:
        do_not_resolve = set(do_not_resolve)
        implicit_resolvers = {
            key: [(tag, regex) for tag, regex in mappings if tag not in do_not_resolve]
            for key, mappings in yaml.SafeLoader.yaml_implicit_resolvers.items()
        }
        return super().__new__(
            cls,
            name,
            (yaml.SafeLoader, *bases),
            {**namespace, "yaml_implicit_resolvers": implicit_resolvers},
        )

class YamlNoTimestampSafeLoader(
    metaclass=YamlLimitedSafeLoader, do_not_resolve={"tag:yaml.org,2002:timestamp"}
):
    """A safe YAML loader that leaves timestamps as strings."""
    pass

class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    TODO: keep somehow reference to the original YAML in order to report better
    KeyError origin.
    """
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            return self.__getattribute__(item)

    @classmethod
    def create(cls, cfg : Any):
        """
        - recursively replace all dicts by the dotdict.
        """
        if isinstance(cfg, dict):
            items = ( (k, cls.create(v)) for k,v in cfg.items())
            return dotdict(items)
        elif isinstance(cfg, list):
            return [cls.create(i) for i in cfg]
        elif isinstance(cfg, tuple):
            return tuple([cls.create(i) for i in cfg])
        else:
            return cfg

    @staticmethod
    def serialize(cfg):
        if isinstance(cfg, (dict, dotdict)):
            return { k:dotdict.serialize(v) for k,v in cfg.items()}
        elif isinstance(cfg, list):
            return [dotdict.serialize(i) for i in cfg]
        elif isinstance(cfg, tuple):
            return tuple([dotdict.serialize(i) for i in cfg])
        else:
            return cfg

Key = Union[str, int]
Path = Tuple[Key]
VariantPatch = Dict[str, dotdict]

@dataclass
class PathIter:
    path: Path
    # full address path
    i: int = 0
    # actual level of the path; initial -1 is before first call to `idx` or `key`.

    def is_leaf(self):
        return self.i == len(self.path)

    def idx(self):
        try:
            return int(self.path[self.i]), PathIter(self.path, self.i + 1)
        except ValueError:
            raise IndexError(f"Variant substitution: IndexError at address: '{self.address()}'.")

    def key(self):
        key = self.path[self.i]
        if len(key) > 0 and not key[0].isdigit():
            return key, PathIter(self.path, self.i + 1)
        else:
            raise KeyError(f"Variant substitution: KeyError at address: '{self.address()}'.")

    def address(self):
        sub_path = self.path[:self.i + 1]
        return '/'.join([str(v) for v in sub_path])


def _item_update(key:Key, val:dotdict, sub_path:Key, sub:dotdict):
    sub_key, path = sub_path
    if key == sub_key:
        if path.empty():
            # Recursion termination
            return sub
        else:
            return deep_update(val, path, sub)
    else:
        return val

def deep_update(cfg: dotdict, iter:PathIter, substitute:dotdict):
    if iter.is_leaf():
        return substitute
    new_cfg = dotdict(cfg)
    if isinstance(cfg, list):
        key, sub_path = iter.idx()
    elif isinstance(cfg, (dict, dotdict)):
        key, sub_path = iter.key()
    else:
        raise TypeError(f"Variant substitution: Unknown type {type(cfg)}")
    new_cfg[key] = deep_update(cfg[key], sub_path, substitute)
    return new_cfg



def apply_variant(cfg:dotdict, variant:VariantPatch) -> dotdict:
    """
    In the `variant` dict the keys are interpreted as the address
    in the YAML file. The address is a list of strings and ints separated by '/'
    and representing an item of the YAML file.
    For every `(address, value)` item of the `variant` dict the referenced item
    in `cfg` is replaced by `value`.

    Implemented by recursion with copy of changed collections.
    May be slow for too many variant items and substitution of the large collection.
    :param cfg:
    :param variant: dictionary path -> dotdict
    :return:
    """
    new_cfg = cfg
    for path_str, val in variant.items():
        path = tuple(path_str.split('/'))
        assert path
        new_cfg = deep_update(new_cfg, PathIter(path), val)
    return new_cfg

class YamlInclude(YamlIncludeConstructor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.included_files = []

    def load(
            self,
            loader,
            pathname: str,
            recursive: bool = False,
            encoding: str = '',
            reader: str = ''
    ):  # pylint:disable=too-many-arguments
        if not encoding:
            encoding = self._encoding or self.DEFAULT_ENCODING
        if self._base_dir:
            pathname = os.path.join(self._base_dir, pathname)
        reader_clz = None
        if reader:
            reader_clz = get_reader_class_by_name(reader)
        if re.match(WILDCARDS_REGEX, pathname):
            result = []
            iterable = iglob(pathname, recursive=recursive)
            for path in filter(os.path.isfile, iterable):
                self.included_files.append(path)
                if reader_clz:
                    result.append(reader_clz(path, encoding=encoding, loader_class=type(loader))())
                else:
                    result.append(self._read_file(path, loader, encoding))
            return result
        self.included_files.append(pathname)
        if reader_clz:
            return reader_clz(pathname, encoding=encoding, loader_class=type(loader))()
        return self._read_file(pathname, loader, encoding)

def resolve_machine_configuration(cfg:dotdict, hostname) -> dotdict:
    # resolve machine configuration
    if 'machine_config' not in cfg:
        return cfg
    if hostname is None:
        hostname = gethostname()
    machine_cfg = cfg.machine_config.get(hostname, None)
    if machine_cfg is None:
        machine_cfg = cfg.machine_config.get('__default__', None)
    if machine_cfg is None:    
        raise KeyError(f"Missing hostname: {hostname} in 'cfg.machine_config'.")
    cfg.machine_config = machine_cfg
    return cfg

def load_config(path, collect_files=False, hostname=None):
    """
    Load configuration from given file replace, dictionaries by dotdict
    uses pyyaml-tags namely for:
    include tag:
        geometry: <% include(path="config_geometry.yaml")>
    """
    instance = YamlInclude.add_to_loader_class(loader_class=YamlNoTimestampSafeLoader, base_dir=os.path.dirname(path))
    cfg_dir = os.path.dirname(path)
    with open(path) as f:
        cfg = yaml.load(f, Loader=YamlNoTimestampSafeLoader)
    cfg['_config_root_dir'] = os.path.abspath(cfg_dir)
    dd = dotdict.create(cfg)
    dd = resolve_machine_configuration(dd, hostname)
    if collect_files:
        referenced = instance.included_files
        referenced.append(path)
        referenced.extend(collect_referenced_files(dd, ['.', cfg_dir]))
        dd['_file_refs'] = referenced
    return dd

def dump_config(config):
    with open("__config_resolved.yaml", "w") as f:
        yaml.dump(config, f)

def path_search(filename, path):
    if not isinstance(filename, str):
        return []
    # Abs paths intentionally not included
    # if os.path.isabs(filename):
    #     if os.path.isabs(filename) and os.path.isfile(filename):
    #         return [os.path.abspath(filename)]
    #     else:
    #         return []
    for dir in path:
        full_name = os.path.join(dir, filename)
        if os.path.isfile(full_name):
            return [os.path.abspath(full_name)]
    return []

FilePath = NewType('FilePath', str)
def collect_referenced_files(cfg:dotdict, search_path:List[str]) -> List[FilePath]:
    if isinstance(cfg, (dict, dotdict)):
        referenced = [collect_referenced_files(v, search_path) for v in cfg.values()]
    elif isinstance(cfg, (list, tuple)):
        referenced = [collect_referenced_files(v, search_path) for v in cfg]
    else:
        return path_search(cfg, search_path)
    # flatten
    return [i for l in referenced for i in l]




