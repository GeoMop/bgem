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

    def read_element_data_head(self, mshfile):

        columns = mshfile.readline().strip().split()
        n_str_tags = int(columns[0])
        assert (n_str_tags == 1)
        field = mshfile.readline().strip().strip('"')

        columns = mshfile.readline().strip().split()
        n_real_tags = int(columns[0])
        assert (n_real_tags == 1)
        columns = mshfile.readline().strip().split()
        time = float(columns[0])

        columns = mshfile.readline().strip().split()
        n_int_tags = int(columns[0])
        assert (n_int_tags == 3)
        columns = mshfile.readline().strip().split()
        t_idx = int(columns[0])
        columns = mshfile.readline().strip().split()
        n_comp = int(columns[0])
        columns = mshfile.readline().strip().split()
        n_elem = int(columns[0])
        return field, time, t_idx, n_comp, n_elem

    def read_element_data_block(self, mshfile):
        field, time, t_idx, n_comp, n_ele = self.read_element_data_head(mshfile)
        field_time_dict = self.element_data.setdefault(field, {})
        assert t_idx not in field_time_dict
        elem_data = {}
        field_time_dict[t_idx] = (time, elem_data)
        for i in range(n_ele):
            line = mshfile.readline()
            if line.startswith('$'):
                raise Exception("Insufficient number of entries in the $ElementData block: {} time={}".format(field, time))
            columns = line.split()
            iel = int(columns[0])
            values = [float(v) for v in columns[1:]]
            assert len(values) == n_comp
            elem_data[iel] = values

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
        for i, el_tag in enumerate(nodeTags):
            offset = i * 3
            self.nodes[int(el_tag)] = [float(coord[offset]),
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

    def write_ascii(self, mshfile=None):
        """Dump the mesh out to a Gmsh 2.0 msh file."""

        if not mshfile:
            mshfile = open(self.filename, 'w')

        print('$MeshFormat\n2.2 0 8\n$EndMeshFormat', file=mshfile)
        print('$PhysicalNames\n%d' % len(self.physical), file=mshfile)
        for name in sorted(self.physical.keys()):
            value = self.physical[name]
            region_id, dim = value
            print('%d %d "%s"' % (dim, region_id, name), file=mshfile)
        print('$EndPhysicalNames', file=mshfile)
        print('$Nodes\n%d' % len(self.nodes), file=mshfile)
        for node_id in sorted(self.nodes.keys()):
            coord = self.nodes[node_id]
            print(node_id, ' ', ' '.join([str(c) for c in coord]), sep="",
                  file=mshfile)
        print('$EndNodes', file=mshfile)
        print('$Elements\n%d' % len(self.elements), file=mshfile)
        for ele_id in sorted(self.elements.keys()):
            elem = self.elements[ele_id]
            (ele_type, tags, nodes) = elem
            print(ele_id, ' ', ele_type, ' ', len(tags), ' ',
                  ' '.join([str(c) for c in tags]), ' ',
                  ' '.join([str(c) for c in nodes]), sep="", file=mshfile)
        print('$EndElements', file=mshfile)

    def write_binary(self, filename=None):
        """Dump the mesh out to a Gmsh 2.0 msh file."""

        if not filename:
            filename = self.filename

        mshfile = open(filename, 'wr')

        mshfile.write("$MeshFormat\n2.2 1 8\n")
        mshfile.write(struct.pack('@i', 1))
        mshfile.write("\n$EndMeshFormat\n")
        mshfile.write("$Nodes\n%d\n" % (len(self.nodes)))
        for node_id, coord in self.nodes.items():
            mshfile.write(struct.pack('@i', node_id))
            mshfile.write(struct.pack('@3d', *coord))
        mshfile.write("\n$EndNodes\n")
        mshfile.write("$Elements\n%d\n" % (len(self.elements)))
        for ele_id, elem in self.elements.items():
            (ele_type, tags, nodes) = elem
            mshfile.write(struct.pack('@i', ele_type))
            mshfile.write(struct.pack('@i', 1))
            mshfile.write(struct.pack('@i', len(tags)))
            mshfile.write(struct.pack('@i', ele_id))
            for c in tags:
                mshfile.write(struct.pack('@i', c))
            for c in nodes:
                mshfile.write(struct.pack('@i', c))
        mshfile.write("\n$EndElements\n")

        mshfile.close()

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
