import pytest

from bgem.gmsh.gmsh import ObjectSet, Region


def make_region(dim, tag):
    return Region(dim=dim, id=tag, name="region_{}".format(tag))


def make_object_set(factory, dim_tags):
    regions = [make_region(dim, tag) for dim, tag in dim_tags]
    obj = ObjectSet(factory=factory, dim_tags=list(dim_tags), regions=regions)
    obj.mesh_step_size = [10 + i for i in range(len(dim_tags))]
    return obj


def assert_object_set_data(obj, dim_tags, regions, mesh_step_size):
    assert obj.dim_tags == dim_tags
    assert obj.regions == regions
    assert obj.mesh_step_size == mesh_step_size


def test_dim_tag_set_compares_unsorted_dimtags():
    factory = object()
    obj = make_object_set(factory, [(2, 20), (1, 10), (3, 30)])
    same_dimtags = make_object_set(factory, [(3, 30), (2, 20), (1, 10)])
    different_dimtags = make_object_set(factory, [(3, 30), (2, 20), (0, 1)])

    assert obj.dim_tags_set == same_dimtags.dim_tags_set
    assert obj.dim_tags_set != different_dimtags.dim_tags_set


def test_dt_copy_preserves_data_without_sharing_lists():
    factory = object()
    obj = make_object_set(factory, [(2, 20), (1, 10), (3, 30)])

    copied = obj.dt_copy()

    assert copied is not obj
    assert_object_set_data(copied, obj.dim_tags, obj.regions, obj.mesh_step_size)

    copied.dim_tags.pop()
    copied.regions.pop()
    copied.mesh_step_size.pop()

    assert obj.dim_tags == [(2, 20), (1, 10), (3, 30)]
    assert len(obj.regions) == 3
    assert obj.mesh_step_size == [10, 11, 12]


def test_dt_intersection_filters_self_by_other_dimtags_preserving_self_data():
    factory = object()
    obj = make_object_set(factory, [(2, 20), (1, 10), (3, 30), (2, 21)])
    other_a = make_object_set(factory, [(3, 30), (0, 5)])
    other_b = make_object_set(factory, [(2, 20), (3, 30)])

    result = obj.dt_intersection(other_a, other_b)

    assert result.factory is factory
    assert_object_set_data(
        result,
        [(2, 20), (3, 30)],
        [obj.regions[0], obj.regions[2]],
        [10, 12],
    )


def test_dt_intersection_rejects_non_object_set_arguments():
    obj = make_object_set(object(), [(2, 20)])

    with pytest.raises(Exception, match="expecting ObjectSet"):
        obj.dt_intersection([(2, 20)])


def test_dt_drop_removes_all_matching_dimtags_in_place():
    factory = object()
    obj = make_object_set(factory, [(2, 20), (1, 10), (3, 30), (2, 21)])
    drop_a = make_object_set(factory, [(1, 10), (0, 5)])
    drop_b = make_object_set(factory, [(2, 21)])

    result = obj.dt_drop(drop_a, drop_b)

    assert result is obj
    assert_object_set_data(
        obj,
        [(2, 20), (3, 30)],
        [make_region(2, 20), make_region(3, 30)],
        [10, 12],
    )


def test_dt_drop_rejects_non_object_set_arguments():
    obj = make_object_set(object(), [(2, 20)])

    with pytest.raises(Exception, match="expecting ObjectSet"):
        obj.dt_drop([(2, 20)])
