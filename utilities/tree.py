def vertical_lines(last_node_flags):
    vertical_lines = []
    vertical_line = '\u2502'
    for last_node_flag in last_node_flags[0:-1]:
        if last_node_flag == False:
            vertical_lines.append(vertical_line + ' ' * 3)
        else:
            vertical_lines.append(' ' * 4)
    return ''.join(vertical_lines)

def horizontal_line(last_node_flags):
    horizontal_line = '\u251c\u2500\u2500 '
    horizontal_line_end = '\u2514\u2500\u2500 '
    if last_node_flags[-1]:
        return horizontal_line_end
    else:
        return horizontal_line

class Tree:

    def __init__(self):
        self.nodes = {}
        self.root = None

    def is_expandable(self, node):
        if node.terminal:
            return False
        if len(node.untried_actions) > 0:
            return True
        return False

    def iter(self, state, depth, last_node_flags):
        if state is None:
            node = self.root
        else:
            node = self.nodes[state]

        if depth == 0:
            yield "", node
        else:
            yield vertical_lines(last_node_flags) + horizontal_line(last_node_flags), node

        children = [self.nodes[state] for state in node.children_identifiers]
        last_index = len(children) - 1

        depth += 1
        for index, child in enumerate(children):
            last_node_flags.append(index == last_index)
            for edge, node in self.iter(child.state, depth, last_node_flags):
                yield edge, node
            last_node_flags.pop()

    def add_node(self, node, parent=None):
        if isinstance(node.state, dict):
            state = next(iter(node.state.values()))
            node.state = state
        
        self.nodes.update({node.state: node})

        if parent is None:
            self.root = node
            self.nodes[node.state].parent = None
        else:
            if isinstance(parent.state, dict):
                parent_state = next(iter(parent.state.values()))
                parent.state = parent_state
            
            self.nodes[parent.state].children_identifiers.append(node.state)
            self.nodes[node.state].parent_identifier = parent.state

    def children(self, node):
        children = []
        for state in self.nodes[node.state].children_identifiers:
            children.append(self.nodes[state])
        return children

    def parent(self, node):
        parent_state = self.nodes[node.state].parent_identifier
        if parent_state is None:
            return None
        else:
            return self.nodes[parent_state]

    def show(self):
        lines = ""
        for edge, node in self.iter(state=None, depth=0, last_node_flags=[]):
            lines += "{}{}\n".format(edge, node)
        print(lines)