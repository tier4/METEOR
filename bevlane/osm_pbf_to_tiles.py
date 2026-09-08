"""Offline OSM tile builder: japan .osm.pbf -> Overpass-JSON tile cache.

Public Overpass endpoints rate-limit / time out under our 5k-scene
extraction, so build the same tile_{ti}_{tj}.json files extract_sdmap.py
expects from a local Geofabrik extract instead. Overpass bbox semantics:
a way belongs to every tile one of its nodes falls in, and all its nodes
are included (recursed) regardless of tile.
"""
import argparse
import json
import os
import sys

import numpy as np
import osmium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_sdmap import TILE_DEG


def needed_tiles(index_path):
    tiles = set()
    for line in open(index_path):
        line = line.strip()
        if not line:
            continue
        sc, raw = line.split("|")
        ep_p = os.path.join(raw, "annotation/ego_pose.json")
        try:
            ep = json.load(open(ep_p))
        except Exception:
            continue
        geo = np.array([e["geocoordinate"][:2] for e in ep
                        if e.get("geocoordinate")], np.float64)
        if not len(geo):
            continue
        pad = 0.004
        la0, la1 = geo[:, 0].min() - pad, geo[:, 0].max() + pad
        lo0, lo1 = geo[:, 1].min() - pad, geo[:, 1].max() + pad
        for ti in range(int(np.floor(la0 / TILE_DEG)),
                        int(np.floor(la1 / TILE_DEG)) + 1):
            for tj in range(int(np.floor(lo0 / TILE_DEG)),
                            int(np.floor(lo1 / TILE_DEG)) + 1):
                tiles.add((ti, tj))
    return tiles


class TileBuilder(osmium.SimpleHandler):
    def __init__(self, tiles):
        super().__init__()
        self.tiles = tiles
        self.way_els = {t: [] for t in tiles}     # way json per tile
        self.node_ids = {t: set() for t in tiles}  # recursed way nodes
        self.node_els = {t: [] for t in tiles}     # crossing/signal/sign nodes
        self.node_loc = {}                          # id -> (lat, lon) needed
        self.node_tags = {}                         # id -> tags (signs etc.)

    def _tile(self, lat, lon):
        return (int(np.floor(lat / TILE_DEG)), int(np.floor(lon / TILE_DEG)))

    KEEP_NODE_HW = ("crossing", "traffic_signals", "stop", "give_way",
                    "speed_camera")

    def node(self, n):
        hw = n.tags.get("highway")
        if hw in self.KEEP_NODE_HW or "traffic_sign" in n.tags:
            t = self._tile(n.location.lat, n.location.lon)
            if t in self.tiles:
                self.node_els[t].append(
                    {"type": "node", "id": n.id, "lat": n.location.lat,
                     "lon": n.location.lon, "tags": dict(n.tags)})
        if hw or "traffic_sign" in n.tags:
            # remember tags so recursed way-nodes keep them (Overpass does)
            self.node_tags[n.id] = dict(n.tags)

    def way(self, w):
        if "highway" not in w.tags:
            return
        locs = []
        for nd in w.nodes:
            if not nd.location.valid():
                return
            locs.append((nd.ref, nd.location.lat, nd.location.lon))
        hit = {self._tile(la, lo) for _, la, lo in locs} & self.tiles
        if not hit:
            return
        el = {"type": "way", "id": w.id, "tags": dict(w.tags),
              "nodes": [i for i, _, _ in locs]}
        for i, la, lo in locs:
            self.node_loc[i] = (la, lo)
        for t in hit:
            self.way_els[t].append(el)
            self.node_ids[t].update(i for i, _, _ in locs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--index", default="out/sdmap_raw_index.txt")
    ap.add_argument("--cache", default="out/osm_cache")
    args = ap.parse_args()

    tiles = needed_tiles(args.index)
    print(f"needed tiles: {len(tiles)}", flush=True)
    h = TileBuilder(tiles)
    h.apply_file(args.pbf, locations=True, idx="flex_mem")
    print("pbf pass done", flush=True)

    os.makedirs(args.cache, exist_ok=True)
    n = 0
    for t in tiles:
        els = []
        for i in sorted(h.node_ids[t]):
            el = {"type": "node", "id": i,
                  "lat": h.node_loc[i][0], "lon": h.node_loc[i][1]}
            if i in h.node_tags:
                el["tags"] = h.node_tags[i]
            els.append(el)
        els += h.node_els[t]
        els += h.way_els[t]
        cp = os.path.join(args.cache, f"tile_{t[0]}_{t[1]}.json")
        tmp = cp + ".tmp"
        json.dump({"elements": els}, open(tmp, "w"))
        os.replace(tmp, cp)
        n += 1
    print(f"wrote {n} tiles -> {args.cache}", flush=True)


if __name__ == "__main__":
    main()
