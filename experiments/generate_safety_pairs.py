"""Generate item pairs for pairwise safety comparison study.

Creates pair assignments for Bradley-Terry style pairwise comparisons of
safety items. For smoke/pilot stages, generates full round-robin (all C(N,2)
pairs). For full stage, generates a random regular graph with degree 8 per
item to keep pair count manageable, then verifies connectivity via BFS.

Usage:
    python generate_safety_pairs.py --stage smoke
    python generate_safety_pairs.py --stage pilot
    python generate_safety_pairs.py --stage full
    python generate_safety_pairs.py --stage full --anchors-safe cse_01_opus45 --anchors-unsafe vcr_09_grok41
"""

import argparse
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

from config import SEED

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stage configs for pairwise pairing
PAIR_STAGES = {
    "smoke": {"n_items": 10, "method": "round_robin"},
    "pilot": {"n_items": 30, "method": "round_robin"},
    "full": {"n_items": 141, "method": "regular_graph", "degree": 8},
}


def load_items(n_items):
    """Load and stratified-sample safety items."""
    items_path = PROJECT_ROOT / "data/items_safety.csv"
    items = pd.read_csv(items_path)

    if n_items >= len(items):
        return items

    # Stratified sample by category
    n_per_cat = max(1, n_items // items["category"].nunique())
    rng = random.Random(SEED)
    sampled = []
    for _, grp in items.groupby("category"):
        indices = list(grp.index)
        rng.shuffle(indices)
        sampled.extend(indices[:min(len(indices), n_per_cat)])

    # If stratified sampling didn't reach n_items, fill from remainder
    remaining = [i for i in items.index if i not in set(sampled)]
    rng.shuffle(remaining)
    sampled.extend(remaining[:max(0, n_items - len(sampled))])

    return items.loc[sampled[:n_items]].reset_index(drop=True)


def generate_round_robin(item_ids):
    """Generate all C(N,2) pairs."""
    return list(combinations(sorted(item_ids), 2))


def generate_regular_graph(item_ids, degree, rng):
    """Generate a random regular graph and ensure connectivity.

    Uses networkx to create a random regular graph with the specified degree.
    If the graph is disconnected, adds bridging edges between components.

    Parameters
    ----------
    item_ids : list
        List of item identifiers.
    degree : int
        Target degree per node.
    rng : random.Random
        Random number generator for reproducibility.

    Returns
    -------
    list of tuple
        List of (item_id_a, item_id_b) pairs.
    """
    n = len(item_ids)

    # random_regular_graph requires n*degree to be even
    actual_degree = degree
    if (n * degree) % 2 != 0:
        actual_degree = degree + 1
        print(f"  Adjusted degree from {degree} to {actual_degree} "
              f"(n*d must be even for {n} nodes)")

    # Generate random regular graph using integer node labels
    G = nx.random_regular_graph(actual_degree, n, seed=SEED)

    # Check connectivity and fix if needed
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"  WARNING: Graph has {len(components)} components, adding bridges")
        for i in range(len(components) - 1):
            # Pick one node from each adjacent component and add an edge
            u = rng.choice(list(components[i]))
            v = rng.choice(list(components[i + 1]))
            G.add_edge(u, v)

    # Map integer nodes back to item_ids
    sorted_ids = sorted(item_ids)
    pairs = []
    for u, v in G.edges():
        pairs.append((sorted_ids[u], sorted_ids[v]))

    return pairs


def ensure_anchors_connected(pairs, anchor_ids, all_item_ids, rng):
    """Ensure anchor items are connected to the graph.

    If an anchor has degree 0 (not in any pair), connect it to a random
    non-anchor item.

    Parameters
    ----------
    pairs : list of tuple
        Existing pairs.
    anchor_ids : set
        Set of anchor item IDs.
    all_item_ids : list
        All item IDs in the study.
    rng : random.Random
        Random number generator.

    Returns
    -------
    list of tuple
        Updated pairs with anchors connected.
    """
    # Find items that appear in at least one pair
    paired_items = set()
    for a, b in pairs:
        paired_items.add(a)
        paired_items.add(b)

    non_anchor_ids = [i for i in all_item_ids if i not in anchor_ids]

    added = 0
    for anchor in anchor_ids:
        if anchor not in paired_items:
            # Connect to a random non-anchor item
            partner = rng.choice(non_anchor_ids)
            pairs.append((anchor, partner))
            paired_items.add(anchor)
            added += 1

    if added > 0:
        print(f"  Added {added} edges to connect isolated anchor items")

    return pairs


def randomize_ab_order(pairs, rng):
    """Randomly assign which item is A and which is B for each pair."""
    randomized = []
    for a, b in pairs:
        if rng.random() < 0.5:
            randomized.append((a, b))
        else:
            randomized.append((b, a))
    return randomized


def print_summary(pairs, item_ids, stage):
    """Print summary statistics about the generated pairs."""
    # Build degree distribution
    degree_count = Counter()
    for a, b in pairs:
        degree_count[a] += 1
        degree_count[b] += 1

    degrees = list(degree_count.values())
    paired_items = set(degree_count.keys())
    unpaired = set(item_ids) - paired_items

    # Check connectivity via networkx
    G = nx.Graph()
    G.add_nodes_from(item_ids)
    G.add_edges_from(pairs)
    connected = nx.is_connected(G)
    n_components = nx.number_connected_components(G)

    print(f"\n=== Pair Generation Summary ({stage}) ===")
    print(f"  Items: {len(item_ids)}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Items with at least one pair: {len(paired_items)}")
    if unpaired:
        print(f"  WARNING: {len(unpaired)} items have no pairs: {sorted(unpaired)[:5]}...")
    print(f"  Graph connected: {connected}")
    if not connected:
        print(f"  Connected components: {n_components}")
    print(f"  Degree distribution:")
    print(f"    Min: {min(degrees) if degrees else 0}")
    print(f"    Max: {max(degrees) if degrees else 0}")
    print(f"    Mean: {sum(degrees) / len(degrees):.1f}" if degrees else "    Mean: 0")
    print(f"    Median: {sorted(degrees)[len(degrees) // 2]}" if degrees else "    Median: 0")

    # Degree histogram
    deg_hist = Counter(degrees)
    print(f"    Histogram: {dict(sorted(deg_hist.items()))}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate item pairs for pairwise safety comparison"
    )
    parser.add_argument(
        "--stage", required=True, choices=PAIR_STAGES.keys(),
        help="Stage: smoke (10 items, round-robin), pilot (30), full (141, regular graph)"
    )
    parser.add_argument(
        "--anchors-safe", type=str, default="",
        help="Comma-separated item_ids to use as safe anchors"
    )
    parser.add_argument(
        "--anchors-unsafe", type=str, default="",
        help="Comma-separated item_ids to use as unsafe anchors"
    )
    args = parser.parse_args()

    random.seed(SEED)
    rng = random.Random(SEED)

    stage_cfg = PAIR_STAGES[args.stage]

    # Parse anchor item IDs
    anchor_safe = set(args.anchors_safe.split(",")) if args.anchors_safe else set()
    anchor_unsafe = set(args.anchors_unsafe.split(",")) if args.anchors_unsafe else set()
    anchor_ids = anchor_safe | anchor_unsafe

    # Load items
    items = load_items(stage_cfg["n_items"])
    item_ids = items["item_id"].tolist()

    # Ensure anchors are in the item set
    existing_ids = set(item_ids)
    if anchor_ids:
        # Load full item list to find anchor rows
        all_items = pd.read_csv(PROJECT_ROOT / "data/items_safety.csv")
        missing_anchors = anchor_ids - existing_ids
        if missing_anchors:
            anchor_rows = all_items[all_items["item_id"].isin(missing_anchors)]
            if len(anchor_rows) < len(missing_anchors):
                not_found = missing_anchors - set(anchor_rows["item_id"])
                print(f"WARNING: Anchor item_ids not found in items_safety.csv: {not_found}")
            items = pd.concat([items, anchor_rows], ignore_index=True)
            items = items.drop_duplicates(subset="item_id").reset_index(drop=True)
            item_ids = items["item_id"].tolist()
            print(f"  Added {len(missing_anchors)} anchor items to item set "
                  f"(total: {len(item_ids)})")

    print(f"=== Generating safety pairs: {args.stage} ===")
    print(f"  Items: {len(item_ids)}")
    print(f"  Method: {stage_cfg['method']}")
    if anchor_ids:
        print(f"  Safe anchors: {anchor_safe or 'none'}")
        print(f"  Unsafe anchors: {anchor_unsafe or 'none'}")

    # Generate pairs
    if stage_cfg["method"] == "round_robin":
        pairs = generate_round_robin(item_ids)
    elif stage_cfg["method"] == "regular_graph":
        degree = stage_cfg["degree"]
        print(f"  Target degree: {degree}")
        pairs = generate_regular_graph(item_ids, degree, rng)
    else:
        raise ValueError(f"Unknown method: {stage_cfg['method']}")

    # Ensure anchors are connected
    if anchor_ids:
        pairs = ensure_anchors_connected(pairs, anchor_ids, item_ids, rng)

    # Randomize A/B assignment
    pairs = randomize_ab_order(pairs, rng)

    # Print summary
    print_summary(pairs, item_ids, args.stage)

    # Build output DataFrame
    item_lookup = items.set_index("item_id")["category"].to_dict()
    rows = []
    for pair_id, (id_a, id_b) in enumerate(pairs):
        rows.append({
            "pair_id": pair_id,
            "item_id_a": id_a,
            "item_id_b": id_b,
            "category_a": item_lookup.get(id_a, "unknown"),
            "category_b": item_lookup.get(id_b, "unknown"),
        })

    out_df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / f"data/processed/safety_pairs_{args.stage}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")
    print(f"  Shape: {out_df.shape}")


if __name__ == "__main__":
    main()
