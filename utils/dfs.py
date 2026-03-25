from utils.tree import TreeNode


def dfs(node: TreeNode, word: str) -> list[str]:
    if len(node.children.keys()) == 0:
        return [word]

    out = []
    for key, node in node.children.items():
        out.extend(dfs(node, word + key))
    return out
