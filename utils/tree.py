from typing import Optional, Self


class TreeNode:
    def __init__(
            self,
            key: str,
            is_word_leaf: bool = True,
            children: Optional[dict] = None
    ):
        self.key = key
        self.is_word_leaf = is_word_leaf
        self.children: dict[str, TreeNode] = children if children else {}

    def add_child(self, child):
        if isinstance(child, TreeNode) and child:
            if child.key not in self.children:
                self.children[child.key] = child
        else:
            raise ValueError("The child node has to be a TreeNode")

    def get_child(self, key: str) -> Optional[Self]:
        """
        This function returns a specific node's child
         :param key: 's key
         :return: TreeNode if it exists
        """
        return self.children.get(key, None)

    def is_child(self, key: str) -> bool:
        return key in self.children.keys()


def is_branch(tree: TreeNode, branch: str) -> bool:
    root = tree
    while (len(branch) > 0):
        if branch[0] in root.children:
            root = root.children[branch[0]]
            branch = branch[1:]
        else:
            return False
    return True


def add_branch(tree: TreeNode, branch: str):
    if len(branch) > 0:
        if not tree.is_child(branch[0]):
            tree.add_child(
                TreeNode(
                    key=branch[0],
                    is_word_leaf=len(branch) == 1
                )
            )
        add_branch(tree.get_child(branch[0]), branch=branch[1:])


def remove_branch(tree: TreeNode, branch: str) -> bool:
    """
    This function aims to remove an existing branch from the tree

    :param tree: tree where the branch has to be removed from
    :param branch: branch to remove
    """
    if len(branch) == 0:
        if len(tree.children) > 0:
            # This case means that there is a tokenization of a word that includes it
            # ex token1 = a, token2= ab
            return False
        return True

    if not tree.is_child(branch[0]):
        raise ValueError("The next path does not exists.")

    remove = remove_branch(
        tree=tree.get_child(branch[0]),
        branch=branch[1:]
    )

    # A child is remove if it has to
    if remove:
        del tree.children[branch[0]]
    return len(tree.children.keys()) == 0
