class Search(object):
    pass

class SearchTree(object):
    def __init__(self):
        self.root = None
        self.adj_list = {}

    def add_child(self, u, v):
        if u == -1:
            self.root = v
            self.adj_list[v] = []
            return
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
        if v not in self.adj_list:
            self.adj_list[v] = []
    
    def get_dict(self, u=None):
        if u is None:
            return self.get_dict(root)
        children = []
        for v in self.adj_list[u]:
            children.append(self.get_dict(v))
        ret = {
            'name': u,
            'children': children
        }
        return ret
        