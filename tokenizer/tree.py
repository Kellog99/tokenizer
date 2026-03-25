from collections import deque


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


def dfs(node: TreeNode, word: str) -> list[str]:
    if len(node.children.keys()) == 0:
        return [word]

    out = []
    for key, node in node.children.items():
        out.extend(dfs(node, word + key))
    return out
