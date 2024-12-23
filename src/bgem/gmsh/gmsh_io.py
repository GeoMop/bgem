"""Module containing an expanded python gmsh class"""
from __future__ import print_function
import threading
import os.path
import numpy as np
import gmsh


def gmsh_finalize():
    # Prevent error when setting signal out of the mian thread
    gmsh.clear()
    if threading.current_thread() is not threading.main_thread():
        gmsh.oldsig = None
    gmsh.finalize()


# class ElementType(enum.IntEnum):
#     simplex_1d = 1
#     simplex_2d = 2
#     simplex_3d = 4
#
# element_sizes = {
#     1: 1,
#     2: 2,
#     4: 3
# }
#


class ModelDataItem:
    def __init__(self, time, tags, values):
        """
        :param time:
        :param tags: list or ndarray of tags
        :param values: list of ndarrays (1D) or ndarray (2D)
        """
        self.time = time
        self.tags = tags
        self.values = values


class GmshIO:
    """This is a class for storing nodes and elements. Based on Gmsh.py

    Members:
    nodes -- A dict of the form { nodeID: [ xcoord, ycoord, zcoord] }
    elements -- A dict of the form { elemID: (type, [tags], [nodeIDs]) }
                Usual tags: (physical_tag, entity_tag)
    physical -- A dict of the form { name: (id, dim) }

    Methods:
    read([file]) -- Parse a Gmsh version 1.0 or 2.0 mesh file
    write([file]) -- Output a Gmsh version 2.0 mesh file
    """

    # el_type: num of nodes per element
    tdict = {1: 2, 2: 3, 3: 4, 4: 4, 5: 5, 6: 6, 7: 5, 8: 3, 9: 6, 10: 9, 11: 10, 15: 1}

    def __init__(self, filename=None):
        """Initialise Gmsh data structure"""
        self.reset()
        self.filename = filename
        if self.filename is not None:
            self._read()

    def reset(self):
        """Reinitialise Gmsh data structure"""
        self.nodes = {}
        self.elements = {}
        self.physical = {}
        self.node_data = {}
        self.element_data = {}
        self.element_node_data = {}

    def normalize(self):
        """
        Due to GMSH bug the API did not preserve node and element IDs
        in MSH2 format. As workaround, we write the mash and read it from the GMSH model
        to normalize the IDs.
        """
        gmsh.initialize(argv=[])
        model_name = "model"
        gmsh.model.add(model_name)
        self._write_nodes()
        physical_dict = self._write_elements()
        self._write_physical(physical_dict)
        fname = "__auxiliary_normalize_mesh.msh2"
        gmsh.write(fname)
        gmsh_finalize()

        self.reset()
        gmsh.initialize()
        gmsh.open(fname)
        self._read_nodes()
        self._read_elements()
        self._read_physical()
        # write views
        # for view_tag in gmsh.view.getTags():
        #     gmsh.view.write(view_tag, filename)

        gmsh_finalize()

    def _read_nodes(self):
        # nodes
        nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodes()
        for i, node_tag in enumerate(nodeTags):
            offset = i * 3
            self.nodes[int(node_tag)] = [float(coord[offset]),
                                         float(coord[offset + 1]),
                                         float(coord[offset + 2])]

    def _read_elements(self):
        # elements

        for entity_dim, entity_tag in gmsh.model.getEntities():
            physicalTags = gmsh.model.getPhysicalGroupsForEntity(entity_dim, entity_tag)
            if len(physicalTags):
                physical_tag = int(physicalTags[0])
            else:
                physical_tag = -1

            elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(entity_dim, entity_tag)
            for i, type in enumerate(elementTypes):
                for j, element_tag in enumerate(elementTags[i]):
                    nodes_num = self.tdict[type]
                    offset = j * nodes_num
                    self.elements[int(element_tag)] = (int(type),
                                                       [physical_tag, entity_tag],
                                                       [int(nodeTags[i][offset + k]) for k in range(nodes_num)])

    def _read_physical(self):
        # physical
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            self.physical[name] = (tag, dim)

    def _read_all_data(self):
        # data
        for view_tag in gmsh.view.getTags():
            steps_num = int(gmsh.view.option.getNumber(view_tag, "NbTimeStep"))
            name = gmsh.view.option.getString(view_tag, "Name")
            for step in range(steps_num):
                data_type, tags, data, time, num_components = gmsh.view.getModelData(view_tag, step)
                # can not print before failure, but only read 10 values, from 13, have to check contents
                # possibly run in valgirnd
                print(f"{step}: {data_type}, {tags}, {time}, {num_components}")
                if data_type == "NodeData":
                    data_dict = self.node_data
                elif data_type == "ElementData":
                    data_dict = self.element_data
                elif data_type == "ElementNodeData":
                    data_dict = self.element_node_data
                else:
                    continue
                if name not in data_dict:
                    data_dict[name] = {}
                data_dict[name][step] = ModelDataItem(time, tags, data)

    def _read(self):
        gmsh.initialize()
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename)
        gmsh.open(self.filename)

        self._read_nodes()
        self._read_elements()
        self._read_physical()
        self._read_all_data()

        gmsh_finalize()

    def get_reg_ids_by_physical_names(self, reg_names, check_dim=-1):
        """
        Returns ids of regions given by names.
        :param reg_names: names of the regions
        :param check_dim: possibly check, that the regions have the chosen dimension
        :return: list of regions ids
        """
        assert len(self.physical) > 0
        reg_ids = []
        for fr in reg_names:
            rid, dim = self.physical[fr]
            if check_dim >= 0:
                assert dim == check_dim
            reg_ids.append(rid)
        return reg_ids

    def get_elements_of_regions(self, reg_ids):
        """
        Supposes one region per element, on the first position in element tags.
        :param reg_ids: region indices
        :return:  of elements of the specified region indices
        """
        ele_ids_list = []
        for eid, elem in self.elements.items():
            type, tags, node_ids = elem
            # suppose only one region per element
            if tags[0] in reg_ids:
                ele_ids_list.append(eid)
        return np.array(ele_ids_list)

    def write_ascii(self, filename=None):
        """Dump the mesh out to a Gmsh 2.0 msh file."""

        if not filename:
            filename = self.filename

        self.write(filename)

    def write_binary(self, filename=None):
        """Dump the mesh out to a Gmsh 2.0 msh file."""

        if not filename:
            filename = self.filename

        self.write(filename, binary=True)

    def set_preserve_ids(self):
        gmsh.option.setNumber("PreserveNumberingMsh2", 1)

    def _write_nodes(self):
        # nodes
        max_entity_tag = 0
        for type, tags, node_tags in self.elements.values():
            entity_tag = tags[1]
            dim = self.tdict[type] - 1
            if dim == 0 and entity_tag > max_entity_tag:
                max_entity_tag = entity_tag
        node_entity_tag = max_entity_tag + 1
        gmsh.model.addDiscreteEntity(0, node_entity_tag)
        node_tags = []
        coords = []
        for node_tag, coord in self.nodes.items():
            node_tags.append(node_tag)
            coords.extend(coord)
        gmsh.model.mesh.addNodes(0, node_entity_tag, node_tags, coords)

    def _write_elements(self):
        # elements
        # created_nodes = set()
        created_entities = set()
        physical_dict = {}
        for element_tag, (type, tags, node_tags) in self.elements.items():
            physical_tag = tags[0]
            entity_tag = tags[1]
            dim = self.tdict[type] - 1

            # create entity
            if (dim, entity_tag) not in created_entities:
                gmsh.model.addDiscreteEntity(dim, entity_tag)
                created_entities.add((dim, entity_tag))
                phy_dim_tag = (dim, physical_tag)
                if phy_dim_tag not in physical_dict:
                    physical_dict[phy_dim_tag] = []
                physical_dict[phy_dim_tag].append(entity_tag)

            # create nodes
            # for node_tag in node_tags:
            #     if node_tag not in created_nodes:
            #         gmsh.model.mesh.addNodes(dim, entity_tag, [node_tag], self.nodes[node_tag])

            gmsh.model.mesh.addElementsByType(entity_tag, type, [element_tag], node_tags)
        return physical_dict

    def _write_physical(self, physical_dict):
        # physical
        for (dim, physical_tag), entity_tags in physical_dict.items():
            gmsh.model.addPhysicalGroup(dim, entity_tags, physical_tag)
        for name, v in self.physical.items():
            gmsh.model.setPhysicalName(v[1], v[0], name)

    def _write_all_data(self, f_handle):
        for data_dict, data_type in [(self.node_data, "NodeData"), (self.element_data, "ElementData"),
                                     (self.element_node_data, "ElementNodeData")]:
            for name, steps_dict in data_dict.items():
                for step, data_item in steps_dict.items():
                    first_el_value_shape = np.atleast_1d(data_item.values[0]).shape[0]

                    if data_type == "ElementNodeData":
                        first_el_n_nodes = len(self.elements[data_item.tags[0]][2])
                        n_comp = first_el_value_shape // first_el_n_nodes
                    else:
                        n_comp = first_el_value_shape
                    self._write_model_data(f_handle, data_type, data_item.tags, name, data_item.values, data_item.time,
                                           step, n_comp)

    def write(self, filename, binary=False, format='auto'):
        """
        :param filename: output path
        :param binary:
        :param format: auto, msh1, msh2, msh22, msh3, msh4, msh40, msh41, msh,
            unv, vtk, wrl, mail, stl, p3d, mesh, bdf, cgns, med, diff, ir3, inp,
            ply2, celum, su2, x3d, dat, neu, m, key, off
        TOOD: find a way to distinguish particular GMSH file format.
        Seems that only working is usage of particular extension.
        :return:
        """
        argv = []
        if binary:
            argv.append("-bin")
        argv.extend(["-format", format])
        # gmsh.setMesh.Format
        gmsh.initialize(argv=argv)
        # if format == 'msh2':
        #    self.set_preserve_ids()

        model_name = "model"
        gmsh.model.add(model_name)

        self._write_nodes()
        physical_dict = self._write_elements()
        self._write_physical(physical_dict)

        # data
        # for data_dict, data_type in [(self.node_data, "NodeData"), (self.element_data, "ElementData"),
        #                              (self.element_node_data, "ElementNodeData")]:
        #     for name, steps_dict in data_dict.items():
        #         view_tag = gmsh.view.add("")
        #         gmsh.view.option.setNumber(view_tag, "NbTimeStep", len(steps_dict))
        #         gmsh.view.option.setString(view_tag, "Name", name)
        #         for step, (time, values_dict) in steps_dict.items():
        #             tags = []
        #             data = []
        #             for tag, d in values_dict.items():
        #                 tags.append(tag)
        #                 data.append(d)
        #             gmsh.view.addModelData(view_tag, step, model_name, data_type, tags, data, time)

        gmsh.write(filename)

        # write views
        # for view_tag in gmsh.view.getTags():
        #     gmsh.view.write(view_tag, filename)
        gmsh_finalize()
        # data
        assert not binary
        with open(filename, "a") as f:
            self._write_all_data(f)

    def _write_model_data(self, f, ele_ids, name, values, time, time_idx, n_comp, data_type="ElementData"):
        """
        Write given element data to the MSH file. Write only a single '$ElementData' section.
        :param f: Output file handle.
        :param ele_ids: Iterable giving element ids of N value rows given in 'values'
        :param name: Field name.
        :param values: np.array (N, L); N number of elements, L values per element (components)
        :param time:
        :param time_idx:
        :param n_comp: numer of components per value
        :param data_type: one of "ElementData", "NodeData", "ElementNodeData"
        :return:

        """
        n_els = len(values)  # works as shape[0] for arrays

        header = (f'1\n'
                  f'"{str(name)}"\n'
                  f"1\n"
                  f"{time}\n"
                  f"3\n"
                  f"{time_idx}\n"
                  f"{n_comp}\n"
                  f"{n_els}\n")

        f.write('${}\n'.format(data_type))
        f.write(header)
        if data_type == "ElementNodeData":
            for ele_id, value_row in zip(ele_ids, values):
                n_values = len(value_row) // n_comp
                value_line = " ".join([str(val) for val in value_row])
                f.write(f"{int(ele_id):d} {n_values} {value_line}\n")
        else:
            if len(values.shape) == 1:
                values = values[:, None]
            for ele_id, value_row in zip(ele_ids, values):
                value_line = " ".join([str(val) for val in value_row.ravel()])
                f.write(f"{int(ele_id):d} {value_line}\n")
        f.write('$End{}\n'.format(data_type))

    def write_element_data(self, f, ele_ids, name, values, time=0, time_idx=0):
        """
        Write given element data to the MSH file. Write only a single '$ElementData' section.
        :param f: Output file stream.
        :param ele_ids: Iterable giving element ids of N value rows given in 'values'
        :param name: Field name.
        :param values: np.array (N, L); N number of elements, L values per element (components)
        :return:

        TODO: Generalize to time dependent fields.
        """
        n_els = values.shape[0]
        # ElementData has all values of the same size.
        n_comp = np.atleast_1d(values[0]).shape[0]
        self._write_model_data(f, ele_ids, name, values, time, time_idx, n_comp, data_type="ElementData")

    def write_node_data(self, f, ele_ids, name, values, time=0, time_idx=0):
        """
        Write given element data to the MSH file. Write only a single '$ElementData' section.
        :param f: Output file stream.
        :param ele_ids: Iterable giving element ids of N value rows given in 'values'
        :param name: Field name.
        :param values: np.array (N, L); N number of elements, L values per element (components)
        :return:

        TODO: Generalize to time dependent fields.
        """
        n_els = values.shape[0]
        # ElementData has all values of the same size.
        n_comp = np.atleast_1d(values[0]).shape[0]
        self._write_model_data(f, ele_ids, name, values, time, time_idx, n_comp, data_type="NodeData")

    def write_fields(self, file_name, ele_ids, fields):
        """
        Append the (element) field data to the `file_name` file.
        :param file_name: Target file (or None for current mesh file)
        :param ele_ids: Element IDs in computational mesh corresponding to order of
        field values in element's barycenter.
        :param fields: {'field_name' : values_array, ..}
        """
        if not file_name:
            file_name = self.filename
        with open(file_name, "a") as fout:
            # fout.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
            for name, values in fields.items():
                self.write_element_data(fout, ele_ids, name, values)

    # def read_element_data(self):
    #     """
    #     Write given element data to the MSH file. Write only a single '$ElementData' section.
    #     :param f: Output file stream.
    #     :param ele_ids: Iterable giving element ids of N value rows given in 'values'
    #     :param name: Field name.
    #     :param values: np.array (N, L); N number of elements, L values per element (components)
    #     :return:
    #
    #     TODO: Generalize to time dependent fields.
    #     """
    #
    #     n_els = values.shape[0]
    #     n_comp = np.atleast_1d(values[0]).shape[0]
    #     np.reshape(values, (n_els, n_comp))
    #     header_dict = dict(
    #         field=str(name),
    #         time=0,
    #         time_idx=0,
    #         n_components=n_comp,
    #         n_els=n_els
    #     )
    #
    #     header = "1\n" \
    #              "\"{field}\"\n" \
    #              "1\n" \
    #              "{time}\n" \
    #              "3\n" \
    #              "{time_idx}\n" \
    #              "{n_components}\n" \
    #              "{n_els}\n".format(**header_dict)
    #
    #     f.write('$ElementData\n')
    #     f.write(header)
    #     assert len(values.shape) == 2
    #     for ele_id, value_row in zip(ele_ids, values):
    #         value_line = " ".join([str(val) for val in value_row])
    #         f.write("{:d} {}\n".format(int(ele_id), value_line))
    #     f.write('$EndElementData\n')
