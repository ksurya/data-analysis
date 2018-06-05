from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy import create_engine
from collections import deque

import numpy as np
import simplejson
import string
import cPickle
import os


Base = declarative_base()


class Author(Base):

    __tablename__ = "authors"

    id = Column(Integer, primary_key=True)
    email = Column(String)
    institution = Column(Text)
    last_name = Column(String)
    first_name = Column(String)
    middle_name = Column(String)
    avatar = Column(String)
    address = Column(Text)
    vitae = Column(Text)


class Document(Base):

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    title = Column(Text)
    abstract = Column(Text)
    publication_date = Column(DateTime)
    submission_date = Column(DateTime)
    online_date = Column(DateTime)
    revision_date = Column(DateTime)
    accepted_date = Column(DateTime)
    cover_url = Column(String)
    full_url = Column(String)
    first_page = Column(Integer)
    last_page = Column(Integer)
    pages = Column(Integer)
    document_type = Column(String)
    type = Column(String)
    article_id = Column(Integer)
    context_key = Column(Integer)
    label = Column(Integer)
    submission_path = Column(String)
    source_id = Column(Integer)


class Keyword(Base):

    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True)
    keyword = Column(Text)


class DocumentAuthorLink(Base):

    __tablename__ = "documents_authors"

    documents_id = Column(Integer, ForeignKey("documents.id"), primary_key=True)
    authors_id = Column(Integer, ForeignKey("authors.id"), primary_key=True)


class DocumentKeywordLink(Base):

    __tablename__ = "documents_keywords"

    documents_id = Column(Integer, ForeignKey("documents.id"), primary_key=True)
    keywords_id = Column(Integer, ForeignKey("keywords.id"), primary_key=True)


class Node(int):
    """Represents node in a graph."""

    def __init__(self, id):
        self.id = id  # `id` in database
        self.last_name = None
        self.email = None
        self.institution = None
        self.cluster = None  # cluster name
        super(int, self).__init__()


class Network(list):
    """Represents adjacency list data structure."""

    def __init__(self, size):
        """Creates a list of `size` elements."""
        super(list, self).__init__()
        self.extend([None] * size)

    def add_node(self, node):
        """Add a note into the graph.

        Args
            node: Node instance
        Returns
            None
        """
        self[node - 1] = (node, [])

    def add_edge(self, node1, node2):
        """Add undirected edge between two nodes.

        This method does not verify whether there exists
        an edge between the two nodes.

        Args
            node1: Node instance
            node2: Node instance
        Returns
            None
        """
        self[node1 - 1][1].append(node2)
        self[node2 - 1][1].append(node1)

    def has_edge(self, node1, node2):
        """Check whether there is an edge between two nodes.

        Args
            node1: Node instance
            node2: Node instance
        Returns
            Boolean, True if edge exists
        """
        return node2 in self[node1 - 1][1]

    def get_node(self, id):
        """Get Node object.

        Args
            id: int
        Returns
            instance of Node corresponding to `id`
        """
        return self[id - 1][0]

    def get_children(self, node):
        """Get children of a node.

        Args
            node: Node instance
        Returns
            List Node objects
        """
        return self[node - 1][1]

    def search_clusters(self):
        """Identify/search clusters in the graph.

        Perform breath-first search to identify clusters
        in the graph. Each node is assigned a cluster ID.
        Each cluster ID is a node within the graph.
        """
        for idx in xrange(1, len(self) + 1):
            node = self.get_node(idx)
            if node.cluster is not None:
                continue

            node.cluster = node.id
            queue = deque()
            queue.appendleft(node)

            while len(queue) > 0:
                node = queue.pop()
                for c in self.get_children(node):
                    if c.cluster is None:
                        c.cluster = node.cluster
                        queue.appendleft(c)

    def get_clusters(self):
        """Get a set of cluster roots.

        Args
            None
        Returns
            Set, root nodes of clusters
        """
        clusters = set()
        for idx in xrange(1, len(self) + 1):
            node = self.get_node(idx)
            root = self.get_node(node.cluster)
            clusters.add(root)
        return clusters

    def get_cluster_nodes(self, node):
        """Get all nodes in a cluster.

        Args
            node: Node, root node of a cluster
        Returns
            List of nodes in the cluster
        """
        node_list = []
        traversed = {}
        queue = deque()

        queue.appendleft(node)
        traversed[node] = True

        while len(queue) > 0:
            node = queue.pop()
            node_list.append(node)
            for c in self.get_children(node):
                if traversed.get(c) is not True:
                    queue.appendleft(c)
                    traversed[c] = True

        return node_list

    def has_path(self, start, dst, max_dist):
        """Check whether there exists a path between two nodes.

        Dijkstra's algorithm to find the shortest path
        between `start` and `dst` nodes.

        Args
            start: Node, starting node
            dst: Node, destination node
            max_dist, int, maximum distance allowed
        Returns
            Bool, True if there exists a path between start and dst
            and the minimum distance of path is less than max_list
        """

        traversed = {}
        distance = {start: 0}
        node_set = set()
        node_set.add(start)

        while len(node_set) > 0:
            node = None
            for n in node_set:
                if node is None or distance[n] < distance[node]:
                    node = n
            traversed[node] = True
            node_set.remove(node)
            if node == dst:
                return True

            if distance[node] < max_dist:
                for c in self.get_children(node):
                    if distance[node] + 1 < distance.get(c, np.inf):
                        distance[c] = distance[node] + 1
                    if traversed.get(c) is not True:
                        node_set.add(c)

        return False


def get_session_object(credentials_pth):
    """Get database session object.

    Args
        credentials_pth: String, absolute or relative path to JSON object
            containing database credentials. The JSON object should have
            {
                "host": "<host address>",
                "port": "<port>",
                "database": "<database>",
                "user": "<username>",
                "password": "<password>"
            }
    Returns
        Session instance
    """
    try:
        with open(credentials_pth, "rb") as fp:
            credentials = simplejson.load(fp)
    except IOError:
        print "ERROR: Unable to open database credentials: ", credentials_pth
        print "ERROR: Please ensure the file exists"
        raise
    uri = "mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(uri.format(**credentials))
    session_cls = sessionmaker()
    session_cls.configure(bind=engine)
    return session_cls()


def create_network(session):
    """Create a network from database tables.

    Args
        session: Session instance
    Returns
        Network instance
    """
    size = session.query(Author).count()
    network = Network(size)

    # add nodes. authors
    query = session.query(Author)
    for author in query.yield_per(10000).enable_eagerloads(False):
        node = Node(author.id)
        node.last_name = author.last_name
        node.institution = author.institution
        node.email = author.email
        network.add_node(node)

    # add edges. there exists an edge between any two co-authors
    da1 = aliased(DocumentAuthorLink)
    da2 = aliased(DocumentAuthorLink)
    query = session.query(da1, da2).\
        join((da2, da1.documents_id == da2.documents_id)).\
        filter(da1.authors_id < da2.authors_id).\
        group_by(da1.authors_id, da2.authors_id)
    for da1, da2 in query.yield_per(10000).enable_eagerloads(False):
        node1 = network.get_node(da1.authors_id)
        node2 = network.get_node(da2.authors_id)
        network.add_edge(node1, node2)

    return network


def sanitized(s):
    """Remove all characters other than alphabet + whitespace."""
    words = ("" if s is None else s.lower()).split()
    letters = list(" ".join(words))
    filtered = [i for i in letters if i in string.ascii_lowercase + string.whitespace]
    return "".join(filtered)


def is_duplicate(node, node_list, network):
    """Check whether two nodes are similar.

    Args
        node: Node instance
        node_list: List of Node instances
        network: Network instance representing the nodes
    Returns
        Boolean, True if `node` is similar to at least one node
        in the `node_list` array
    """
    for n in node_list:
        # definitely a duplicate if email addr match
        if node.email and n.email and node.email.lower() == n.email.lower():
            return True

        # sometimes people use multiple sub domains but their
        # user name in an institution remains same
        if node.institution and n.institution and node.email and n.email and \
                sanitized(node.institution) == sanitized(n.institution) and \
                node.email.split("@")[0].lower() == n.email.split("@")[0].lower():
            return True

        # not a duplicate if last names don't match. last name
        # could have two words.. sometimes abbreviated..
        if sanitized(node.last_name) != sanitized(n.last_name):
            continue

        # check the distance between the two nodes
        if network.has_path(node, n, 3):
            return True

    return False


def find_duplicate_authors(network, output_file_pth):
    """Find duplicates in the graph and write to a CSV file.

    Each line in the `output_file_path` file represents a
    list of nodes that are similar (duplicates)

    Args
        network: Network, graph of authors
        output_file_pth: String, absolute or relative path to
            to the file to write duplicates data
    Returns
        None
    """

    # search clusters in the network
    network.search_clusters()

    # get root nodes of clusters
    clusters = network.get_clusters()
    size = len(clusters)

    # write duplicates
    file_obj = open(output_file_pth, "wb")

    # find duplicates within each cluster
    while size > 0:

        # represents cluster root
        root = clusters.pop()
        all_nodes = network.get_cluster_nodes(root)
        print "INFO: Processing cluster {0}, size {1} ".format(root, len(all_nodes))

        # key: last name of author.
        # value: array of duplicate author records (nodes)
        container = {}

        # iterate over each node in the cluster and check
        # whether its a duplicate.
        for node in network.get_cluster_nodes(root):
            key = sanitized(node.last_name)
            duplicates = container.get(key, [])
            if is_duplicate(node, duplicates, network) or len(duplicates) == 0:
                duplicates.append(node)
            container[key] = duplicates

        # write to a file
        for v in container.itervalues():
            if len(v) > 1:
                file_obj.write(",".join([str(i.id) for i in v]) + "\n")

        size -= 1

    file_obj.close()


def main():
    cache_file = "network.pickle"
    if os.path.exists(cache_file):
        print "INFO: Using cached network data"
        network = cPickle.load(open(cache_file, "rb"))
    else:
        print "INFO: Creating network (will be cached)"
        fp = open(cache_file, "wb")
        session = get_session_object("credentials.json")
        network = create_network(session)
        cPickle.dump(network, fp)
        fp.close()

    print "INFO: Finding duplicates"
    find_duplicate_authors(network, "duplicates.csv")


if __name__ == "__main__":
    main()
