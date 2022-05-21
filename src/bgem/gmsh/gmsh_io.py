"""Module containing an expanded python gmsh class"""
from __future__ import print_function

import struct
import numpy as np
import enum
import gmsh


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

class GmshIO:
    """This is a class for storing nodes and elements. Based on Gmsh.py

    Members:
    nodes -- A dict of the form { nodeID: [ xcoord, ycoord, zcoord] }
    elements -- A dict of the form { elemID: (type, [tags], [nodeIDs]) }
    physical -- A dict of the form { name: (id, dim) }

    Methods:
    read([file]) -- Parse a Gmsh version 1.0 or 2.0 mesh file
    write([file]) -- Output a Gmsh version 2.0 mesh file
    """

    def __init__(self, filename=None):
        """Initialise Gmsh data structure"""
        self.reset()
        self.filename = filename
        if self.filename:
            self.read()

    def reset(self):
        """Reinitialise Gmsh data structure"""
        self.nodes = {}
        self.elements = {}
        self.physical = {}
        self.node_data = {}
        self.element_data = {}
        self.element_node_data = {}

    def read_physical_names(self, mshfile=None):
        """Read physical names from a Gmsh .msh file.

        Reads Gmsh format 1.0 and 2.0 mesh files,
        reads only '$PhysicalNames' section.
        """

        if not mshfile:
            mshfile = open(self.filename, 'r')

        readmode = 0
        print('Reading %s' % mshfile.name)
        line = 'a'
        while line:
            line = mshfile.readline()
            line = line.strip()

            if line.startswith('$'):
                if line == '$PhysicalNames':
                    readmode = 5
                else:
                    readmode = 0
            elif readmode == 5:
                columns = line.split()
                if len(columns) == 3:
                    self.physical[str(columns[2]).strip('\"')] = (int(columns[1]), int(columns[0]))
        mshfile.close()

        return self.physical

    def read(self):
        """Read a Gmsh .msh file.

        Reads Gmsh format 1.0 and 2.0 mesh files, storing the nodes and
        elements in the appropriate dicts.
        """

        gmsh.initialize()
        gmsh.open(self.filename)

        # nodes
        nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodes()
        for i, node_tag in enumerate(nodeTags):
            offset = i * 3
            self.nodes[int(node_tag)] = [float(coord[offset]),
                                       float(coord[offset + 1]),
                                       float(coord[offset + 2])]

        # elements
        # el_type: num of nodes per element
        tdict = {1: 2, 2: 3, 3: 4, 4: 4, 5: 5, 6: 6, 7: 5, 8: 3, 9: 6, 10: 9, 11: 10, 15: 1}

        for entity_dim, entity_tag in gmsh.model.getEntities():
            physicalTags = gmsh.model.getPhysicalGroupsForEntity(entity_dim, entity_tag)
            if len(physicalTags):
                physical_tag = int(physicalTags[0])
            else:
                physical_tag = -1

            elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(entity_dim, entity_tag)
            for i, type in enumerate(elementTypes):
                for j, element_tag in enumerate(elementTags[i]):
                    nodes_num = tdict[type]
                    offset = j * nodes_num
                    self.elements[int(element_tag)] = (int(type),
                                                       [physical_tag, entity_tag],
                                                       [int(nodeTags[i][offset + k]) for k in range(nodes_num)])

        # physical
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            self.physical[name] = (tag, dim)

        # data
        for view_tag in gmsh.view.getTags():
            steps_num = int(gmsh.view.option.getNumber(view_tag, "NbTimeStep"))
            name = gmsh.view.option.getString(view_tag, "Name")
            for step in range(steps_num):
                dataType, tags, data, time, numComponents = gmsh.view.getModelData(view_tag, step)
                if dataType == "NodeData":
                    data_dict = self.node_data
                elif dataType == "ElementData":
                    data_dict = self.element_data
                elif dataType == "ElementNodeData":
                    data_dict = self.element_node_data
                else:
                    continue
                if name not in data_dict:
                    data_dict[name] = {}
                values_dict = {}
                for i, tag in enumerate(tags):
                    values_dict[int(tag)] = data[i].tolist()
                data_dict[name][step] = (time, values_dict)

        gmsh.clear()
        gmsh.finalize()

        print('  %d Nodes' % len(self.nodes))
        print('  %d Elements' % len(self.elements))

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
        :return: indices of elements of the specified region indices
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

        self.write(filename, bin=True)

    def write(self, filename, binary=False):
        if binary:
            argv = ["", "-bin"]
        else:
            argv = []
        gmsh.initialize(argv=argv)

        model_name = "model"
        gmsh.model.add(model_name)

        # nodes
        # el_type : dim
        tdict = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 1, 9: 2, 10: 2, 11: 3, 15: 0}
        max_entity_tag = 0
        for type, tags, node_tags in self.elements.values():
            entity_tag = tags[1]
            dim = tdict[type]
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

        # elements
        #created_nodes = set()
        created_entities = set()
        physical_dict = {}
        for element_tag, (type, tags, node_tags) in self.elements.items():
            physical_tag = tags[0]
            entity_tag = tags[1]
            dim = tdict[type]

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

        # physical
        for (dim, physical_tag), entity_tags in physical_dict.items():
            gmsh.model.addPhysicalGroup(dim, entity_tags, physical_tag)
        for name, v in self.physical.items():
            gmsh.model.setPhysicalName(v[1], v[0], name)

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

        gmsh.clear()
        gmsh.finalize()

    def write_element_data(self, f, ele_ids, name, values):
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
        n_comp = np.atleast_1d(values[0]).shape[0]
        np.reshape(values, (n_els, n_comp))
        header_dict = dict(
            field=str(name),
            time=0,
            time_idx=0,
            n_components=n_comp,
            n_els=n_els
        )

        header = "1\n" \
                 "\"{field}\"\n" \
                 "1\n" \
                 "{time}\n" \
                 "3\n" \
                 "{time_idx}\n" \
                 "{n_components}\n" \
                 "{n_els}\n".format(**header_dict)

        f.write('$ElementData\n')
        f.write(header)
        assert len(values.shape) == 2
        for ele_id, value_row in zip(ele_ids, values):
            value_line = " ".join([str(val) for val in value_row])
            f.write("{:d} {}\n".format(int(ele_id), value_line))
        f.write('$EndElementData\n')

    def write_fields(self, msh_file, ele_ids, fields):
        """
        Creates input data msh file for Flow model.
        :param msh_file: Target file (or None for current mesh file)
        :param ele_ids: Element IDs in computational mesh corrsponding to order of
        field values in element's barycenter.
        :param fields: {'field_name' : values_array, ..}
        """
        if not msh_file:
            msh_file = open(self.filename, 'w')
        with open(msh_file, "w") as fout:
            fout.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
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
