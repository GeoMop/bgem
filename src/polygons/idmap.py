"""
A dictionary that automaticaly assign unique IDs to appended objects
- IDs are set to the objects
- Objects can be compared by IDs.
TODO: Make auxiliary class for producing IDs and allow
several IdMaps to source from common ID source
"""
class IdObject:

    def __init__(self):
        self.attr = None

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class IdSource:
    pass

class IdMap(dict):
    def __init__(self):
        self._next_id = -1
        # Source of new object IDs. The last used ID.
        self.hint_id = None
        # Te ID used for the very next new object unless it is None
        # sed in implementation of UNDO/REDO.
        super().__init__()


    def get_new_id(self):
        if self.hint_id is not None:
            assert self.hint_id not in self
            new_id = self.hint_id
            self.hint_id = None
            return new_id
        self._next_id += 1
        while self._next_id in self:
            self._next_id += 1
        return self._next_id

    def append(self, obj):
        id = self.get_new_id()
        obj.id = id
        self[obj.id] = obj
        return obj

