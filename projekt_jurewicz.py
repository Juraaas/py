from typing import Any, Callable, List


def visit(node: 'BinaryNode') -> None:
    print(node)


class BinaryNode:
    value: Any
    left_child: 'BinaryNode'
    right_child: 'BinaryNode'

    def __init__(self, value: Any, left_child: 'BinaryNode', right_child: 'BinaryNode') -> None:
        self.value = value
        self.left_child = None
        self.right_child = None

    def is_leaf(self) -> bool:
        if self.left_child.len() and self.right_child is None:
            return True
        else:
            return False

    def add_left_child(self, value: Any):
        self.left_child = BinaryNode(value, None, None)

    def add_right_child(self, value: Any):
        self.right_child = BinaryNode(value, None, None)

    def traverse_in_order(self, visit: Callable[[Any], None]):
        if self.left_child is not None:
            self.left_child.traverse_in_order(visit)
        visit(self.value)
        if self.right_child is not None:
            self.right_child.traverse_in_order(visit)

    def traverse_post_order(self, visit: Callable[[Any], None]):
        if self.left_child is not None:
            self.left_child.traverse_post_order(visit)
        if self.right_child is not None:
            self.right_child.traverse_post_order(visit)
        visit(self.value)

    def traverse_pre_order(self, visit: Callable[[Any], None]):
        visit(self.value)
        if self.left_child is not None:
            self.left_child.traverse_pre_order(visit)
        if self.right_child is not None:
            self.right_child.traverse_pre_order(visit)

    def __str__(self) -> str:
        return str(self.value)


class BinaryTree:
    root: BinaryNode

    def __init__(self, root):
        self.root = root

    def traverse_in_order(self, visit: Callable[[Any], None]):
        if self.root is not None:
            self.root.traverse_in_order(visit)

    def traverse_post_order(self, visit: Callable[[Any], None]):
        if self.root is not None:
            self.root.traverse_post_order(visit)

    def traverse_pre_order(self, visit: Callable[[Any], None]):
        if self.root is not None:
            self.root.traverse_pre_order(visit)

    def horizontal_sum(self, tree: BinaryNode) -> List[Any]:
        result: List[int] = []

        #dodawanie elementu do listy i sumowanie jezeli poziom sie powtorzyl
        def level_assign(node: 'TreeNode', level: int = 0) -> None:
            if len(result) <= level:
                result.append(node.value)
            else:
                result[level] += node.value
        #przypisanie poziomu
            for i in node.left_child, node.right_child:
                if i is not None:
                    level_assign(i, level + 1)
        level_assign(tree.root)
        return result

    def show(self, tree: BinaryNode, space: int, height: int):
        if tree is None:
            return
        #ilosc spacji zgodnie ze wpisanym parametrem
        space += height
        self.show(tree.right_child, space, height)
        print()
        for i in range(height, space):
            print(' ', end='')
        print(tree.value, end='')
        print()
        self.show(tree.left_child, space, height)


tree_root = BinaryNode(10, None, None)

tree = BinaryTree(tree_root)

tree_root.add_left_child(9)
tree.root.add_right_child(2)
tree.root.left_child.add_left_child(1)
tree.root.left_child.add_right_child(3)
tree.root.right_child.add_left_child(4)
tree.root.right_child.add_right_child(6)
tree.root.right_child.right_child.add_right_child(5)

# tree.traverse_in_order(visit)
# print("\n")
# tree.traverse_post_order(visit)
# print("\n")
# tree.traverse_pre_order(visit)
# print("\n")
print(tree.horizontal_sum(tree))
tree.show(tree_root, 0, 5)
