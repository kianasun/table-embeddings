import pickle
from collections import defaultdict
from pathlib import Path

import rdflib
from rdflib.term import URIRef


def read_tables_from_nquads(quad_file):
    predicate_to_columns = defaultdict(list)

    with quad_file.open() as reader:
        idx = 0
        for line in reader.readlines():
            idx += 1
            try:
                g = rdflib.ConjunctiveGraph()
                g.parse(data=line.strip(), format="nquads")

                for s, p, o, u in g.quads((None, None, None)):
                    # print(o, type(o))
                    if isinstance(o, URIRef):
                        continue
                    predicate_to_columns[str(p)].append(str(o))
            except:
                continue

            if idx % 1000000 == 0:
                print(idx)
                pickle.dump(predicate_to_columns, (Path("output") / f"dict_{idx // 1000000}.pkl").open("wb"))
                predicate_to_columns.clear()

    return predicate_to_columns
