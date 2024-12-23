from typing import Dict, Tuple, List
import attrs
import bih
import numpy as np
import bisect

from bgem.gmsh.gmsh_io import GmshIO
from bgem.gmsh import heal_mesh

from bgem.core import File, memoize, report


# @njit
def element_vertices(all_nodes: np.array, node_indices: np.array):
    return all_nodes[node_indices[:], :]


# @njit
def element_loc_mat(all_nodes: np.array, node_indices: List[int]):
    n = element_vertices(all_nodes, node_indices)
    return (n[1:, :] - n[0]).T


# @njit
def element_compute_volume(all_nodes: np.array, node_indices: List[int]):
    return np.linalg.det(element_loc_mat(all_nodes, node_indices)) / 6


@attrs.define
class Element:
    mesh: 'Mesh'
    type: int
    tags: Tuple[int, int]
    node_indices: List[int]

    def vertices(self):
        return element_vertices(self.mesh.nodes, np.array(self.node_indices, dtype=int))

    def loc_mat(self):
        return element_loc_mat(self.mesh.nodes, self.node_indices)

    def volume(self):
        return element_compute_volume(self.mesh.nodes, self.node_indices)

    def barycenter(self):
        return np.mean(self.vertices(), axis=0)

    def gmsh_tuple(self, node_map):
        node_ids = [node_map[inode] for inode in self.node_indices]
        return (self.type, self.tags, node_ids)


# @memoize
def _load_mesh(mesh_file: 'File', heal_tol=None):
    # mesh_file = mesh_file.path
    if heal_tol is None:
        gmsh_io = GmshIO(str(mesh_file))
        return Mesh(gmsh_io, file=mesh_file)
    else:
        hm = heal_mesh.HealMesh.read_mesh(str(mesh_file), node_tol=heal_tol * 0.8)
        report(hm.heal_mesh)(gamma_tol=heal_tol)
        # hm.move_all(geom_dict["shift_vec"])
        # elm_to_orig_reg = hm.map_regions(new_reg_map)
        report(hm.stats_to_yaml)(mesh_file.with_suffix(".heal_stats.yaml"))
        # assert hm.healed_mesh_name == mesh_healed
        hm.write()
        return Mesh.load_mesh(hm.healed_mesh_name, None)

    # !! can not memoize static and class methods (have no name)


# @report
# @njit
def mesh_compute_el_volumes(nodes: np.array, node_indices: np.array) -> np.array:
    return np.array([element_compute_volume(nodes, ni) for ni in node_indices])


class Mesh:

    @staticmethod
    def load_mesh(mesh_file: 'File', heal_tol=None) -> 'Mesh':
        return _load_mesh(mesh_file, heal_tol)

    @staticmethod
    def empty(mesh_path) -> 'Mesh':
        return Mesh(GmshIO(), mesh_path)

    def __init__(self, gmsh_io: GmshIO, file):

        self.gmsh_io: GmshIO = gmsh_io
        # TODO: remove relation to file
        # rather use a sort of generic wrapper around loadable objects
        # in order to relay on the underlying files for the caching
        self.file: 'File' = file
        self.reinit()

    def reinit(self):
        # bounding interval hierarchy for the mesh elements
        # numbers elements from 0 as they are added
        self._update_nodes()
        self._update_elements()
        self._update_regions()

        # _boxes: List[bih.AABB]
        self._bih: bih.BIH = None

        self._el_volumes: np.array = None
        self._el_barycenters: np.array = None

    def _update_nodes(self):
        self.node_ids = []
        self.node_indices = {}
        self.nodes = np.empty((len(self.gmsh_io.nodes), 3))
        for i, (nid, node) in enumerate(self.gmsh_io.nodes.items()):
            self.node_indices[nid] = i
            self.node_ids.append(nid)
            self.nodes[i, :] = node

    def _update_elements(self):
        self.el_ids = []
        self.el_indices = {}
        self.elements = []
        for i, (eid, el) in enumerate(self.gmsh_io.elements.items()):
            type, tags, node_ids = el
            element = Element(self, type, tags, [self.node_indices[nid] for nid in node_ids])
            self.el_indices[eid] = i
            self.el_ids.append(eid)
            self.elements.append(element)

    def _update_regions(self):
        self.regions = {dim_tag: name for name, dim_tag in self.gmsh_io.physical.items()}

    def __getstate__(self):
        return (self.gmsh_io, self.file)

    def __setstate__(self, args):
        self.gmsh_io, self.file = args
        self.reinit()

    @property
    def bih(self):
        if self._bih is None:
            self._bih = self._build_bih()
        return self._bih

    def _build_bih(self):
        el_boxes = []
        for el in self.elements:
            node_coords = el.vertices()
            box = bih.AABB(node_coords)
            el_boxes.append(box)
        _bih = bih.BIH()
        _bih.add_boxes(el_boxes)
        _bih.construct()
        return _bih

    def candidate_indices(self, box):
        list_box = box.tolist()
        return self.bih.find_box(bih.AABB(list_box))

    # def el_volume(self, id):
    #     return self.elements[self.el_indices[id]].volume()

    @property
    # @report
    def el_volumes(self):
        if self._el_volumes is None:
            node_indices = np.array([e.node_indices for e in self.elements], dtype=int)
            print(f"Compute el volumes: {self.nodes.shape}, {node_indices.shape}")
            self._el_volumes = mesh_compute_el_volumes(self.nodes, node_indices)
        return self._el_volumes

    def el_barycenters(self):
        if self._el_barycenters is None:
            self._el_barycenters = np.array([e.barycenter() for e in self.elements])
        return self._el_barycenters

    def fr_map(self, dfn: 'FractureSet', reg_id_to_fr: Dict[str, int]):
        """
        Get int field with fracture idx for fracture elements and len(dfn) for other
        :param dfn:
        :param reg_id_to_fr:
        :return:
        TODO: better way to map elements to fracture numbers
        Limiting factors:
        - we have no control over actual region IDs produced by meshing so we must use
          physical names to encode fracture numbers
        - alternative could be based on functional approach that would allow mapping shape IDs back to shapes
          and then shapes (including fractures), may be associated to various attributes
        - we currently we also depend on gmsh_io.regions when reading the mesh
        """

        # TODO better association mechanism
        # fr_reg_to_idx = {fr.region.id - 100000 - 2: idx for idx, fr in enumerate(fractures)}
        def fr_id_from_name(name: str):
            try:
                _, fam, fr_id = name.split('_')
                fr_id = int(fr_id)
            except ValueError:
                fr_id = len(dfn)
            return fr_id

        el_type = [31, 1, 2, 4]  # gmsh element types
        reg_id_to_fr = {(el_type[dim], reg_id): fr_id_from_name(name) for (reg_id, dim), name in self.regions.items()}
        fr_map = [reg_id_to_fr.get((e.type, e.tags[0]), len(dfn)) for e in self.elements]
        return np.array(fr_map)

    # def el_loc_mat(self, id):
    #     return self.elements[self.el_indices[id]].loc_mat()

    # def el_barycenter(self, id):
    #     return self.elements[self.el_indices[id]].barycenter()

    # def el_nodes(self, id):
    #     return self.elements[self.el_indices[id]].vertices()

    def submesh(self, elements, file_path):
        gmesh = GmshIO()
        active_nodes = np.full((len(self.nodes),), False)
        for iel in elements:
            el = self.elements[iel]
            active_nodes[el.node_indices] = True
        sub_nodes = self.nodes[active_nodes]
        new_for_old_nodes = np.zeros((len(self.nodes),), dtype=int)
        new_for_old_nodes[active_nodes] = np.arange(1, len(sub_nodes) + 1, dtype=int)
        gmesh.nodes = {(nidx + 1): node for nidx, node in enumerate(sub_nodes)}
        gmesh.elements = {(eidx + 100): self.elements[iel].gmsh_tuple(node_map=new_for_old_nodes) for eidx, iel in
                          enumerate(elements)}
        # print(gmesh.elements)
        gmesh.physical = self.gmsh_io.physical
        # gmesh.write(file_path)
        gmesh.normalize()
        return Mesh(gmesh, "")

    # Returns field P0 values of field.
    # Selects the closest time step lower than 'time'.
    # TODO: we might do time interpolation
    def get_p0_values(self, field_name: str, time):
        field_dict = self.gmsh_io.element_data[field_name]

        # determine maximal index of time step, where times[idx] <= time
        times = [v.time for v in list(field_dict.values())]
        last_time_idx = bisect.bisect_right(times, time) - 1

        values = field_dict[last_time_idx].values
        value_ids = field_dict[last_time_idx].tags
        value_to_el_idx = [self.el_indices[iv] for iv in value_ids]
        values_mesh = np.empty_like(values)
        values_mesh[value_to_el_idx[:]] = values
        return values_mesh

    def get_static_p0_values(self, field_name: str):
        field_dict = self.gmsh_io.element_data[field_name]
        assert len(field_dict) == 1
        values = field_dict[0].values
        value_ids = field_dict[0].tags
        value_to_el_idx = [self.el_indices[iv] for iv in value_ids]
        values_mesh = np.empty_like(values)
        values_mesh[value_to_el_idx[:]] = values
        return values_mesh

    def get_static_p1_values(self, field_name: str):
        field_dict = self.gmsh_io.node_data[field_name]
        assert len(field_dict) == 1
        values = field_dict[0].values
        value_ids = field_dict[0].tags
        value_to_node_idx = [self.node_indices[iv] for iv in value_ids]
        values_mesh = np.empty_like(values)
        values_mesh[value_to_node_idx[:]] = values
        return values_mesh

    def write_fields(self, file_name: str, fields: Dict[str, np.array] = None) -> 'File':
        self.gmsh_io.write(file_name, format="msh2")
        if fields is not None:
            self.gmsh_io.write_fields(file_name, self.el_ids, fields)
        return file_name  # File(file_name)

    def map_regions(self, new_reg_map):
        """
        Replace all (reg_id, dim) regions by the new regions.
        new_reg_map: (reg_id, dim) -> new (reg_id, dim, reg_name)
        return: el_id -> old_reg_id
        """
        # print(self.mesh.physical)
        # print(new_reg_map)

        new_els = {}
        el_to_old_reg = {}

        for el_id, el in self.gmsh_io.elements.items():
            type, tags, nodes = el
            tags = list(tags)
            old_reg_id = tags[0]
            dim = len(nodes) - 1
            old_id_dim = (old_reg_id, dim)
            if old_id_dim in new_reg_map:
                el_to_old_reg[self.el_indices[el_id]] = old_id_dim
                reg_id, reg_dim, reg_name = new_reg_map[old_id_dim]
                if reg_dim != dim:
                    Exception(f"Assigning region of wrong dimension: ele dim: {dim} region dim: {reg_dim}")
                self.gmsh_io.physical[reg_name] = (reg_id, reg_dim)
                tags[0] = reg_id
            self.gmsh_io.elements[el_id] = (type, tags, nodes)
        # remove old regions
        id_to_reg = {id_dim: k for k, id_dim in self.gmsh_io.physical.items()}
        for old_id_dim in new_reg_map.keys():
            if old_id_dim in id_to_reg:
                del self.gmsh_io.physical[id_to_reg[old_id_dim]]

        el_ids = self.el_ids
        self._update_elements()
        assert_idx = np.random.randint(0, len(el_ids), 50)
        assert all((el_ids[i] == self.el_ids[i] for i in assert_idx))
        return el_to_old_reg

    def el_dim_slice(self, dim):
        i_begin = len(self.elements)
        i_end = i_begin
        el_type = [21, 1, 2, 4][dim]
        for iel, el in enumerate(self.elements):
            if el.type == el_type:
                i_begin = iel
                break
        for iel, el in enumerate(self.elements[i_begin:], start=i_begin):
            if el.type != el_type:
                i_end = iel
                break
        for iel, el in enumerate(self.elements[i_end:], start=i_end):
            if el.type == el_type:
                raise IndexError(f"Elements of dimension {dim} does not form a slice.")
        return slice(i_begin, i_end, 1)
