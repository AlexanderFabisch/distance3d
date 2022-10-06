import numpy as np
import numba

"""
https://github.com/JamesRandall/SimpleVoxelEngine/blob/master/voxelEngine/src/AABBTree.cpp

node = (aabb_index, parent, child0, child1, )
"""

INDEX_NONE = -1
PARENT_INDEX = 0
LEFT_INDEX = 1
RIGHT_INDEX = 2
TYPE_INDEX = 3
TYPE_NONE = -1
TYPE_LEAF = 1
TYPE_BRANCH = 2


def new_tree_from_aabbs(aabbs):
    """ Create root and nodes arrays for new tree.

    Parameters
    ----------
    aabbs : np.array([3, 2])
        Axis-aligned bounding boxes array

    Returns
    -------
    root_node_index : int
        The new index of the root node.

    nodes : np.array(4, dtype=int)
        The new nodes array.

    aabbs : np.array([3, 2])
        Axis-aligned bounding boxes array
    """

    root = INDEX_NONE
    l = len(aabbs)
    nodes = np.full([l * 2, 4], INDEX_NONE)
    z = np.zeros((len(nodes) - l, 3, 2), dtype=aabbs.dtype)
    aabbs = np.append(aabbs, z, axis=0)

    return insert_aabbs(root, nodes, aabbs, l)


@numba.njit(cache=True)
def insert_aabbs(root, nodes, aabbs, l):
    for i in range(l):
        root, nodes, aabbs, l = insert_leaf(root, i, nodes, aabbs, l)

    return root, nodes, aabbs


@numba.njit(cache=True)
def insert_leaf(root_node_index, leaf_node_index, nodes, aabbs, len):
    """ Inserts a new leaf into the tree.

    Parameters
    ----------
    root_node_index : int
        The index of the root node. Set to -1 if it's the first leaf.

    leaf_node_index : int
        The index of the new leaf.

    nodes : np.array(4, dtype=int)
        The nodes array. Completly filled with -1 at the beginning.

    aabbs : np.array([3, 2])
        Axis-aligned bounding boxes array

    Returns
    -------
    root_node_index : int
        The new index of the root node.

    nodes : np.array(4, dtype=int)
        The new nodes array.

    aabbs : np.array([3, 2])
        The new Axis-aligned bounding boxes array

    l : int
        New len
    """

    # The Node that's going to be added.
    # A node in this system consists of three variables. An int as index.
    # The node array with containing [parent_index, left_child_index, right_child_index, type]

    nodes[leaf_node_index, TYPE_INDEX] = TYPE_LEAF

    # If there is no root make new leaf root.
    if root_node_index == INDEX_NONE:
        return leaf_node_index, nodes, aabbs, len

    # Traverse the tree down till you find a leaf.
    tree_node_index = root_node_index
    while nodes[tree_node_index, TYPE_INDEX] == TYPE_BRANCH:

        # Getting nodes from arrays
        left_node_index = nodes[tree_node_index, LEFT_INDEX]
        right_node_index = nodes[tree_node_index, RIGHT_INDEX]

        # Whether the left or right child is traverse down depends on the cost of increasing the size of the aabbs.
        cost_new_parent = aabb_surface(
            merge_aabb(aabbs[leaf_node_index], aabbs[tree_node_index]))

        cost_left = aabb_surface(
            merge_aabb(aabbs[leaf_node_index], aabbs[left_node_index]))

        cost_right = aabb_surface(
            merge_aabb(aabbs[leaf_node_index], aabbs[right_node_index]))

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
    new_parent_index = len
    len += 1

    nodes[new_parent_index, PARENT_INDEX] = old_parent_index
    nodes[new_parent_index, LEFT_INDEX] = sibling_index
    nodes[new_parent_index, RIGHT_INDEX] = leaf_node_index
    nodes[new_parent_index, TYPE_INDEX] = TYPE_BRANCH
    aabbs[new_parent_index] = merge_aabb(aabbs[leaf_node_index], aabbs[sibling_index])

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

    return root_node_index, nodes, aabbs, len


@numba.njit(cache=True)
def fix_upward_tree(tree_node_index, nodes, aabbs):
    # Go the tree back up while fixing the aabbs.
    while tree_node_index != INDEX_NONE:
        tree_node = nodes[tree_node_index]

        # Every branch in the traversal should not have any None fields.
        assert tree_node[LEFT_INDEX] != INDEX_NONE and tree_node[RIGHT_INDEX] != INDEX_NONE

        aabbs[tree_node_index] = merge_aabb(aabbs[tree_node[LEFT_INDEX]], aabbs[tree_node[RIGHT_INDEX]])

        # Moving on parent up
        tree_node_index = tree_node[PARENT_INDEX]

    return aabbs


@numba.njit(cache=True)
def merge_aabb(aabb1, aabb2):
    return np.array(
        [[min(aabb1[0, 0], aabb2[0, 0]), max(aabb1[0, 1], aabb2[0, 1])],
         [min(aabb1[1, 0], aabb2[1, 0]), max(aabb1[1, 1], aabb2[1, 1])],
         [min(aabb1[0, 0], aabb2[2, 0]), max(aabb1[2, 1], aabb2[2, 1])]]
    )


@numba.njit(cache=True)
def aabb_surface(aabb):
    return aabb_x_size(aabb) * aabb_y_size(aabb) * aabb_z_size(aabb)


@numba.njit(cache=True)
def aabb_x_size(aabb):
    return aabb[0, 1] - aabb[0, 0]


@numba.njit(cache=True)
def aabb_y_size(aabb):
    return aabb[1, 1] - aabb[1, 0]


@numba.njit(cache=True)
def aabb_z_size(aabb):
    return aabb[2, 1] - aabb[2, 0]


@numba.njit(cache=True)
def aabb_overlap(aabb1, aabb2):
    return aabb1[0, 0] <= aabb2[0, 1] and aabb1[0, 1] >= aabb2[0, 0] \
           and aabb1[1, 0] <= aabb2[1, 1] and aabb1[1, 1] >= aabb2[1, 0] \
           and aabb1[2, 0] <= aabb2[2, 1] and aabb1[2, 1] >= aabb2[2, 0]


@numba.njit(cache=True)
def aabb_contains(aabb1, aabb2):
    return aabb1[0, 0] <= aabb2[0, 0] and aabb1[0, 1] >= aabb2[0, 1] \
           and aabb1[1, 0] <= aabb2[1, 0] and aabb1[1, 1] >= aabb2[1, 1] \
           and aabb1[2, 0] <= aabb2[2, 0] and aabb1[2, 1] >= aabb2[2, 1]


@numba.njit(cache=True)
def query_overlap(test_aabb, root_node_index, nodes, aabbs, break_at_first_leaf=False):
    overlaps = []
    stack = [root_node_index]

    while len(stack) != 0:

        node_index = stack[-1]
        stack = stack[:-1]

        if node_index == INDEX_NONE:
            continue

        node_aabb = aabbs[node_index]
        if aabb_overlap(node_aabb, test_aabb):

            if nodes[node_index, TYPE_INDEX] == TYPE_LEAF:
                overlaps.extend([node_index])

                if break_at_first_leaf:
                    break
            else:
                stack.extend([nodes[node_index, 1], nodes[node_index, 2]])

    return np.array(overlaps)


@numba.njit(cache=True)
def query_overlap_of_other_tree(root1, nodes1, aabbs1, root2, nodes2, aabbs2):
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


def print_aabb_tree(root_node_index, nodes):
    lines, *_ = print_aabb_tree_recursive(root_node_index, nodes)
    for line in lines:
        print(line)


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
