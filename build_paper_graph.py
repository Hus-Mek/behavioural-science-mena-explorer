import json
import re
from pathlib import Path
from collections import Counter
import networkx as nx

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANALYSES_DIR = DATA_DIR / "analyses"

BEHAVIOURAL_TERMS = [
    "nudge", "behavior", "behaviour", "cognitive", "decision", "choice",
    "heuristic", "bias", "motivation", "incentive", "reward",
    "social norm", "conformity", "compliance", "attitude", "perception",
    "learning", "memory", "emotion", "risk", "trust", "cooperation",
    "habit", "automaticity", "self-control", "attention", "salience",
    "dual process", "prospect", "framing", "priming", "anchoring",
]

MIDDLE_EAST_TERMS = [
    "saudi", "arabia", "uae", "emirati", "dubai", "riyadh",
    "qatar", "kuwait", "bahrain", "oman", "muscat",
    "egypt", "cairo", "jordan", "amman", "lebanon",
    "mena", "middle east", "gulf", "arab", "islamic", "muslim",
]

def load_papers():
    files = sorted(RAW_DIR.glob("papers_*.json"))
    all_papers = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    all_papers.extend(data)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    return all_papers

def build_paper_graph(papers):
    G = nx.Graph()
    G.graph["directed"] = False
    G.graph["multigraph"] = False
    
    # Add paper nodes
    for paper in papers:
        pid = paper.get("id") or paper.get("entry_id") or ""
        if not pid:
            continue
        
        title = paper.get("title", "")
        abstract = (paper.get("summary") or "")[:500]
        authors = paper.get("authors", [])
        year = ""
        pub = paper.get("published", "")
        if pub:
            try:
                year = str(pub)[:4]
            except:
                year = pub[:4]
        
        # Load analysis if available
        analysis_path = ANALYSES_DIR / f"{re.sub(r'[^a-zA-Z0-9._-]', '_', pid)}.json"
        analysis = None
        if analysis_path.exists():
            try:
                analysis = json.load(open(analysis_path, encoding="utf-8"))
            except:
                pass
        
        G.add_node(pid, label=title[:80], title=title, abstract=abstract,
                   authors=authors, year=year, source=paper.get("source", ""),
                   analysis=analysis, domain="paper")
    
    # Build edges based on shared concepts
    for term in BEHAVIOURAL_TERMS + MIDDLE_EAST_TERMS:
        matching = []
        for paper in papers:
            pid = paper.get("id") or paper.get("entry_id") or ""
            if not pid:
                continue
            text = ((paper.get("title") or "") + " " + (paper.get("summary") or "")).lower()
            if term in text:
                matching.append(pid)
        
        # Connect papers sharing concepts
        for i, p1 in enumerate(matching):
            for p2 in matching[i+1:]:
                if G.has_edge(p1, p2):
                    G[p1][p2]["weight"] += 1
                    G[p1][p2]["shared_concepts"].append(term)
                else:
                    G.add_edge(p1, p2, weight=1, shared_concepts=[term])
    
    # Author-based connections
    author_papers = {}
    for paper in papers:
        pid = paper.get("id") or paper.get("entry_id") or ""
        if not pid:
            continue
        for author in paper.get("authors", []):
            if author not in author_papers:
                author_papers[author] = []
            author_papers[author].append(pid)
    
    for author, pids in author_papers.items():
        for i, p1 in enumerate(pids):
            for p2 in pids[i+1:]:
                if G.has_edge(p1, p2):
                    G[p1][p2]["weight"] += 2
                    G[p1][p2].setdefault("shared_authors", []).append(author)
                else:
                    G.add_edge(p1, p2, weight=2, shared_authors=[author], shared_concepts=[])
    
    return G

def cluster_graph(G, min_comm=3):
    """Run Louvain community detection."""
    if G.number_of_nodes() == 0:
        return {}
    
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, resolution=1.0)
        return partition
    except ImportError:
        # Fallback to greedy modularity
        communities = nx.community.greedy_modularity_communities(G, resolution=1.0)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition

def score_cohesion(G, partition):
    """Score communities by internal edge density."""
    scores = {}
    for comm_id in set(partition.values()):
        nodes = [n for n, c in partition.items() if c == comm_id]
        if len(nodes) < 2:
            scores[comm_id] = 0.0
            continue
        subgraph = G.subgraph(nodes)
        possible_edges = len(nodes) * (len(nodes) - 1) / 2
        actual_edges = subgraph.number_of_edges()
        scores[comm_id] = round(actual_edges / possible_edges, 3) if possible_edges > 0 else 0
    return scores

def detect_god_nodes(G):
    """Find nodes with high betweenness centrality (bridge nodes)."""
    if G.number_of_nodes() == 0:
        return []
    centrality = nx.betweenness_centrality(G)
    threshold = sorted(centrality.values(), reverse=True)[:max(5, len(G)//20)]
    if not threshold:
        return []
    avg_top = sum(threshold) / len(threshold)
    gods = [n for n, c in centrality.items() if c > avg_top * 1.5 and c > 0.01]
    return gods

def generate_html(G, partition, output_path):
    """Generate interactive HTML for the graph."""
    nodes_json = [{"id": n, "label": G.nodes[n].get("label", n)[:50],
                   "title": G.nodes[n].get("title", n),
                   "year": G.nodes[n].get("year", ""),
                   "community": partition.get(n, 0)}
                  for n in G.nodes()]
    
    edges_json = [{"from": u, "to": v,
                   "value": d.get("weight", 1),
                   "title": ", ".join(d.get("shared_concepts", []))}
                  for u, v, d in G.edges(data=True)]
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Paper Knowledge Graph - Behavioural Science MENA</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #121212; color: #e0e0e0; font-family: -apple-system, sans-serif; }}
        #graph {{ width: 100vw; height: 100vh; }}
        #info {{ position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; max-width: 300px; display: none; }}
        #legend {{ position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 8px; }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="info"></div>
    <div id="legend">
        <strong>Paper Graph</strong><br>
        Node size = connections<br>
        Color = community<br>
        Hover node for details
    </div>
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes_json)});
        const edges = new vis.DataSet({json.dumps(edges_json)});
        const container = document.getElementById('graph');
        const data = {{ nodes, edges }};
        const options = {{
            physics: {{ stabilization: false }},
            nodes: {{ shape: 'dot', font: {{ size: 14 }} }},
            edges: {{ color: {{ color: '#444', highlight: '#666' }} }},
            layout: {{ improvedLayout: true }}
        }};
        const network = new vis.Network(container, data, options);
        network.on('hoverNode', function(params) {{
            const node = nodes.get(params.node);
            document.getElementById('info').innerHTML = '<strong>' + node.title + '</strong><br>' + node.year;
            document.getElementById('info').style.display = 'block';
        }});
        network.on('blurNode', function() {{
            document.getElementById('info').style.display = 'none';
        }});
    </script>
</body>
</html>'''
    
    output_path.write_text(html, encoding="utf-8")
    return len(nodes_json)

def main():
    print("Loading papers...")
    papers = load_papers()
    print(f"Loaded {len(papers)} papers")
    
    print("Building graph...")
    G = build_paper_graph(papers)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print("Clustering...")
    partition = cluster_graph(G)
    cohesion = score_cohesion(G, partition)
    
    print("Detecting god nodes...")
    gods = detect_god_nodes(G)
    
    # Export graph.json in vis-network format
    graph_out = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": n, **G.nodes[n], "community": partition.get(n, 0)} 
                  for n in G.nodes()],
        "edges": [{"source": u, "target": v, **d} 
                  for u, v, d in G.edges(data=True)]
    }
    
    out_path = ROOT / "graphify-out" / "paper_graph.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph_out, ensure_ascii=False, indent=2), encoding="utf-8")
    
    analysis = {
        "communities": {str(i): v for i, v in partition.items()},
        "cohesion": {str(i): v for i, v in cohesion.items()},
        "gods": gods,
    }
    (ROOT / "graphify-out" / "paper_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("Generating HTML...")
    html_path = ROOT / "graphify-out" / "paper_graph.html"
    node_count = generate_html(G, partition, html_path)
    print(f"HTML generated: {node_count} nodes rendered")
    
    print(f"\nOutputs in {ROOT / 'graphify-out'}")
    print(f"  paper_graph.json - graph data")
    print(f"  paper_graph.html - interactive graph")

if __name__ == "__main__":
    main()