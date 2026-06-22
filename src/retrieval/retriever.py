# from __future__ import annotations

# def find_nodes_by_ids(nodes, ids):
#     found = []
#     for n in nodes:
#         if n["node_id"] in ids:
#             found.append(n)
#         if n["nodes"]:
#             found.extend(find_nodes(n["nodes"], ids))
#     return found