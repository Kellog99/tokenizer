class TreeNode:
    def __init__(self, value: str):
        self.value = value
        self.children: dict[str, TreeNode] = {}

    def add_child(self, value: str) -> None:
        if value not in self.children:
            self.children[value] = TreeNode(value)

    def get_child(self, value: str):
        return self.children.get(value, None)

    def is_child(self, value: str) -> bool:
        return value in self.children


def add_branch(tree: TreeNode, branch: list[str]):
    if len(branch) > 0:
        if not tree.is_child(branch[0]):
            tree.add_child(branch[0])
        add_branch(tree.get_child(branch[0]), branch=branch[1:])


def remove_branch(tree: TreeNode, branch: list[str]):
    if len(branch) > 0:
        if tree.is_child(branch[1]):
            remove_branch(tree, branch[1:])
        else:
            pass