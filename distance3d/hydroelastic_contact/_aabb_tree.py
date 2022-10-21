import numpy as np
import numba
from ._aabb import _aabb_overlap, _merge_aabb, _aabb_volume, _sort_aabbs

"""
https://github.com/JamesRandall/SimpleVoxelEngine/blob/master/voxelEngine/src/AABBTree.cpp
"""


INDEX_NONE = -1
PARENT_INDEX = 0
LEFT_INDEX = 1
RIGHT_INDEX = 2
TYPE_INDEX = 3
TYPE_NONE = -1
TYPE_LEAF = 1
TYPE_BRANCH = 2


class AabbTree:
    def __init__(self, aabbs, pre_insertion_methode="none"):
        """Creation of the aabb tree

        Parameters
        ----------
        aabbs : array, shape (n, 3, 2)
            An array containing a list of aabbs of the tree.

        pre_insertion_methode : str, optional (default: "none")
            The operation that is performed on the aabbs before tree creation.
            Use "sort" for a cleaner tree with slightly longer creation times.
            Use "shuffle" for a faster creation but with some non-optimal placement in the tree.

        Returns
        -------
        aabb_tree : AabbTree
        """

        if len(aabbs) == 0:
            return

        root = INDEX_NONE
        filled_len = len(aabbs)

        # TODO Don't just double the size. RAM is wasted
        # TODO Documentation
        # TODO Proper Unit Tests

        nodes = np.full([filled_len * 2, 4], INDEX_NONE)
        z = np.zeros((len(nodes) - filled_len, 3, 2), dtype=aabbs.dtype)
        aabbs = np.append(aabbs, z, axis=0)

        insert_order = np.array(range(filled_len))
        if pre_insertion_methode == "sort":
            insert_order = _sort_aabbs(aabbs[:len(nodes) - filled_len])
        elif pre_insertion_methode == "shuffle":
            np.random.shuffle(insert_order)

        root, nodes, aabbs = insert_aabbs(root, nodes, aabbs, filled_len, insert_order)

        self.root = root
        self.nodes = nodes
        self.aabbs = aabbs

    def __str__(self):
        lines, *_ = print_aabb_tree_recursive(self.root, self.nodes)
        return '\r\n'+'\r\n'.join(lines)

    def overlaps_aabb_tree(self, other):
        """ Check overlapping of another tree.

        Parameters
        ----------
        other : AabbTree
            The other Tree for overlap testing.

        Returns
        -------
        is_overlapping : bool
            True if there is an overlap in the two trees.

        overlap_tetrahedron1 : array, shape (n)
            The indexes of the overlapping tetrahedron in this tree.

        overlap_tetrahedron2 : array, shape (n)
            The indexes of the overlapping tetrahedron in the other tree.

        overlap_pairs : array, shape (n, 2)
            An array of all overlapping pairs.
        """
        overlap_tetrahedron1, overlap_tetrahedron2, overlap_pairs = query_overlap_of_other_tree(
            self.root, self.nodes, self.aabbs, other.root, other.nodes, other.aabbs)

        is_overlapping = len(overlap_pairs) > 0
        return is_overlapping, np.unique(overlap_tetrahedron1), np.unique(overlap_tetrahedron2), overlap_pairs

    def overlaps_aabb(self, aabb):
        """ Check overlapping of an aabb.

        Parameters
        ----------
        aabb : array, shape (3, 2)
            The aabb that is checked.

        Returns
        -------
        is_overlapping : bool
            True if there is an overlap in the two trees.

        overlap_pairs : array, shape (n, 2)
            An array of all overlapping pairs.
        """
        overlap_pairs = query_overlap(aabb, self.root, self.nodes, self.aabbs)
        return len(overlap_pairs) > 0, overlap_pairs


@numba.njit(cache=True)
def insert_aabbs(root, nodes, aabbs, filled_len, insert_order):
    """Inserts aabbs into the tree defined in root and nodes.

    Parameters
    ----------
    root : int
        The index of the tree root node.

    nodes : array, shape (n, 4)
        The index links of the tree structure.
        The indices of the 1-axis correspond to the aabb indices.
        The first entry in of the 2-axis is the index of the parent.
        The second entry in of the 2-axis is the index of the left child.
        The third entry in of the 2-axis is the index of the right child.
        The forth entry in of the 2-axis is 1 if leaf and 2 if branch node.

    aabbs : array, shape (3, 2)
        The aabbs that are inserted.

    filled_len : int
        The length to which the nodes and aabbs are filled.

    insert_order : array, shape (filled_len)
        A list of indexes describing the insert order into the tree.

    Returns
    -------
    root : int
        The index of the tree root node.

    nodes : array, shape (n, 4)
        The index links of the tree structure.

    aabbs : array, shape (3, 2)
        The aabbs of the tree.
    """
    for i in insert_order:
        root, nodes, aabbs, filled_len = insert_leaf(root, i, nodes, aabbs, filled_len)

    return root, nodes, aabbs


@numba.njit(cache=True)
def insert_leaf(root_node_index, leaf_node_index, nodes, aabbs, filled_len):
    """
    Inserts a new leaf into the tree.
    """
    # The Node that's going to be added.
    # A node in this system consists of three variables. An int as index.
    # The node array with containing [parent_index, left_child_index, right_child_index, type]

    nodes[leaf_node_index, TYPE_INDEX] = TYPE_LEAF

    # If there is no root make new leaf root.
    if root_node_index == INDEX_NONE:
        return leaf_node_index, nodes, aabbs, filled_len

    # Traverse the tree down till you find a leaf.
    tree_node_index = root_node_index
    while nodes[tree_node_index, TYPE_INDEX] == TYPE_BRANCH:

        # Getting nodes from arrays
        left_node_index = nodes[tree_node_index, LEFT_INDEX]
        right_node_index = nodes[tree_node_index, RIGHT_INDEX]

        # Whether the left or right child is traverse down depends on the cost of increasing the size of the aabbs.
        cost_new_parent = _aabb_volume(
            _merge_aabb(aabbs[leaf_node_index], aabbs[tree_node_index]))

        cost_left = _aabb_volume(
            _merge_aabb(aabbs[leaf_node_index], aabbs[left_node_index]))

        cost_right = _aabb_volume(
            _merge_aabb(aabbs[leaf_node_index], aabbs[right_node_index]))

        if cost_left > cost_new_parent and cost_right > cost_new_parent:
            break

        # otherwise go down the cheaper child.
        if cost_left < cost_right:
            tree_node_index = left_node_index
        else:
            tree_node_index = right_node_index

    # Setting up sibling node
    sibling_index = tree_node_index

    # Inserting new leaf into tree
    old_parent_index = nodes[sibling_index, PARENT_INDEX]

    # Adding Parent
    new_parent_index = filled_len
    filled_len += 1

    nodes[new_parent_index, PARENT_INDEX] = old_parent_index
    nodes[new_parent_index, LEFT_INDEX] = sibling_index
    nodes[new_parent_index, RIGHT_INDEX] = leaf_node_index
    nodes[new_parent_index, TYPE_INDEX] = TYPE_BRANCH
    aabbs[new_parent_index] = _merge_aabb(aabbs[leaf_node_index], aabbs[sibling_index])

    # Setting Parent in children
    nodes[leaf_node_index, PARENT_INDEX] = new_parent_index
    nodes[sibling_index, PARENT_INDEX] = new_parent_index

    if old_parent_index == INDEX_NONE:
        root_node_index = new_parent_index
    else:
        if nodes[old_parent_index, LEFT_INDEX] == sibling_index:
            nodes[old_parent_index, LEFT_INDEX] = new_parent_index
        else:
            nodes[old_parent_index, RIGHT_INDEX] = new_parent_index

    # Set the index to the parent for upward traversal.
    tree_node_index = nodes[leaf_node_index, PARENT_INDEX]
    aabbs = fix_upward_tree(tree_node_index, nodes, aabbs)

    return root_node_index, nodes, aabbs, filled_len


@numba.njit(cache=True)
def fix_upward_tree(tree_node_index, nodes, aabbs):
    """
    Fixes the aabbs of the parent branches by setting them to the merge of the children aabbs.
    """

    # Go the tree back up while fixing the aabbs.
    while tree_node_index != INDEX_NONE:
        tree_node = nodes[tree_node_index]

        # Every branch in the traversal should not have any None fields.
        assert tree_node[LEFT_INDEX] != INDEX_NONE and tree_node[RIGHT_INDEX] != INDEX_NONE

        aabbs[tree_node_index] = _merge_aabb(aabbs[tree_node[LEFT_INDEX]], aabbs[tree_node[RIGHT_INDEX]])

        # Moving on parent up
        tree_node_index = tree_node[PARENT_INDEX]

    return aabbs


@numba.njit(cache=True)
def query_overlap_of_other_tree(root1, nodes1, aabbs1, root2, nodes2, aabbs2):
    """
    Queries the overlapping aabbs by traversing the trees.
    """

    broad_tetrahedra1 = []
    broad_tetrahedra2 = []
    stack = [root2]

    while len(stack) != 0:

        node_index = stack[-1]
        stack = stack[:-1]

        if node_index == INDEX_NONE:
            continue

        node_aabb = aabbs2[node_index]
        if nodes2[node_index, TYPE_INDEX] == TYPE_BRANCH and \
                len(query_overlap(node_aabb, root1, nodes1, aabbs1, break_at_first_leaf=True)) >= 1:
            stack.extend([nodes2[node_index, 1], nodes2[node_index, 2]])

        elif nodes2[node_index, TYPE_INDEX] == TYPE_LEAF:
            overlaps = query_overlap(node_aabb, root1, nodes1, aabbs1)
            broad_tetrahedra1.extend(overlaps)
            broad_tetrahedra2.extend([node_index] * len(overlaps))

    broad_pairs = list(zip(broad_tetrahedra1, broad_tetrahedra2))

    return np.array(broad_tetrahedra1), np.array(broad_tetrahedra2), broad_pairs


@numba.njit(cache=True)
def query_overlap(test_aabb, root_node_index, nodes, aabbs, break_at_first_leaf=False):
    """
    Queries the overlapping aabbs by traversing the tree.
    """
    overlaps = []
    stack = [root_node_index]

    while len(stack) != 0:

        node_index = stack[-1]
        stack = stack[:-1]

        if node_index == INDEX_NONE:
            continue

        node_aabb = aabbs[node_index]
        if _aabb_overlap(node_aabb, test_aabb):

            if nodes[node_index, TYPE_INDEX] == TYPE_LEAF:
                overlaps.extend([node_index])

                if break_at_first_leaf:
                    break
            else:
                stack.extend([nodes[node_index, 1], nodes[node_index, 2]])

    return np.array(overlaps)


def print_aabb_tree_recursive(node_index, nodes):
    """Returns list of strings, width, height, and horizontal coordinate of the root."""
    # From https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python

    # No child.
    line = '%s' % node_index
    width = len(line)

    if nodes[node_index, LEFT_INDEX] == INDEX_NONE and nodes[node_index, RIGHT_INDEX] == INDEX_NONE:
        line = '%s' % node_index
        height = 1
        middle = width // 2
        return [line], width, height, middle

    # Only left child.
    if nodes[node_index, RIGHT_INDEX] == INDEX_NONE:
        lines, n, p, x = print_aabb_tree_recursive(nodes[node_index, LEFT_INDEX], nodes)

        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + line
        second_line = x * ' ' + '/' + (n - x - 1 + width) * ' '
        shifted_lines = [line + width * ' ' for line in lines]
        return [first_line, second_line] + shifted_lines, n + width, p + 2, n + width // 2

    # Only right child.
    if nodes[node_index, LEFT_INDEX] == INDEX_NONE:
        lines, n, p, x = print_aabb_tree_recursive(nodes[node_index, RIGHT_INDEX], nodes)

        first_line = line + x * '_' + (n - x) * ' '
        second_line = (width + x) * ' ' + '\\' + (n - x - 1) * ' '
        shifted_lines = [width * ' ' + line for line in lines]
        return [first_line, second_line] + shifted_lines, n + width, p + 2, width // 2

    # Two children.
    left, n, p, x = print_aabb_tree_recursive(nodes[node_index, LEFT_INDEX], nodes)
    right, m, q, y = print_aabb_tree_recursive(nodes[node_index, RIGHT_INDEX], nodes)

    first_line = (x + 1) * ' ' + (n - x - 1) * '_' + line + y * '_' + (m - y) * ' '
    second_line = x * ' ' + '/' + (n - x - 1 + width + y) * ' ' + '\\' + (m - y - 1) * ' '
    if p < q:
        left += [n * ' '] * (q - p)
    elif q < p:
        right += [m * ' '] * (p - q)
    zipped_lines = zip(left, right)
    lines = [first_line, second_line] + [a + width * ' ' + b for a, b in zipped_lines]
    return lines, n + m + width, max(p, q) + 2, n + width // 2

