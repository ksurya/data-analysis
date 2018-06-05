# A simple approach to author name disambiguation in scientific publication text documents

- Corpus: Fintime50 publication texts stored in MySQL database
- Hypothesis: Nodes that represent a set of duplicate authors must have at least one common co-author. Therefore, a set of nodes representing an author entity would have to belong to the same cluster.
- `output` directory contain visualizations of the graph.
- There are several small clusters containing 2-7 nodes and few big clusters. `output/clusters.csv` contains cluster id and corresponding size.
- `output/duplicates.csv` contain identified duplicate author records. Each line represent a set of duplicates.




Date/Period: Summer 2017

Advisor: Professor Harry Wang, Alfred Lerner's College of Business and Economics

