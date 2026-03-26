from utils.tree import TreeNode


def dfs(node: TreeNode, word: str) -> list[str]:
    if len(node.children.keys()) == 0:
        # for sure this contains the end of a phrase token
        return [word]

    out = []
    for key, node in node.children.items():
        if node.is_word_leaf:
            out.append(word + key)
        out.extend(dfs(node, word + key))
    return out
