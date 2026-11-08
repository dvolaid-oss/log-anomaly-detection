"""Drain log parser: turn free-text log messages into templates (event IDs).

    Receiving block blk_-160899 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
        → mask →  Receiving block <BLK> src: /<IP> dest: /<IP>
        → Drain → E5 = "Receiving block <BLK> src: /<IP> dest: /<IP>"

Two steps:
1. **Masking** replaces values we already know are variables (block IDs, IPs, numbers)
   with placeholders, using regexes from the config.
2. **Drain** (He et al., 2017) groups the masked messages: it walks a fixed-depth tree keyed
   on message length and first tokens, and puts a message into an existing group if enough
   tokens match (`sim_threshold`). Tokens that differ inside a group become `<*>`.

We use the `drain3` library for step 2 and wrap it, so the rest of the project only sees
`fit`, `match`, `templates`, `save` and `load`.
"""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonpickle
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


@dataclass(frozen=True)
class Mask:
    """Replace every match of `pattern` with `<name>`."""

    name: str
    pattern: str

    @property
    def placeholder(self) -> str:
        return f"<{self.name}>"


def event_id(cluster_id: int) -> str:
    return f"E{cluster_id}"


class LogParser:
    """Masking + Drain, with the settings from the config's `parsing` section."""

    def __init__(self, sim_threshold: float = 0.4, depth: int = 4, masks: Iterable[Mask] = ()):
        self.sim_threshold = sim_threshold
        self.depth = depth
        self.masks = list(masks)
        self._compiled = [(re.compile(m.pattern), m.placeholder) for m in self.masks]

        config = TemplateMinerConfig()
        config.drain_sim_th = sim_threshold
        config.drain_depth = depth
        # We mask ourselves (see `mask`), so drain3's own masking stays empty.
        self._miner = TemplateMiner(config=config)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LogParser:
        parsing = config.get("parsing", {})
        masks = [Mask(m["name"], m["pattern"]) for m in parsing.get("masks", [])]
        return cls(parsing.get("sim_threshold", 0.4), parsing.get("depth", 4), masks)

    def mask(self, message: str) -> str:
        """Apply the masks in order. Order matters: block IDs and IPs contain numbers,
        so they must be masked before the generic number mask."""
        for regex, placeholder in self._compiled:
            message = regex.sub(placeholder, message)
        return message

    def fit(self, masked_messages: Iterable[str]) -> list[str]:
        """Learn templates from already-masked messages; return each message's event ID.

        Drain is an *online* algorithm: it reads messages one by one, and a group's
        template can still get more general later. A message's group (its event ID) never
        changes, though, so the IDs returned here stay valid.
        """
        return [event_id(self._miner.add_log_message(m)["cluster_id"]) for m in masked_messages]

    def match(self, message: str) -> str | None:
        """Event ID for a NEW raw message, without learning from it (None if no template fits).

        Used to parse logs the parser has never seen, e.g. uploads in the dashboard.
        """
        cluster = self._miner.match(self.mask(message), full_search_strategy="fallback")
        return event_id(cluster.cluster_id) if cluster else None

    @property
    def templates(self) -> dict[str, str]:
        """{event_id: template}, e.g. {"E5": "Receiving block <BLK> src: /<IP> dest: /<IP>"}."""
        return {
            event_id(c.cluster_id): c.get_template()
            for c in sorted(self._miner.drain.clusters, key=lambda c: c.cluster_id)
        }

    def save(self, path: Path) -> None:
        """Save settings + the learned Drain tree as zlib-compressed JSON."""
        state = {
            "sim_threshold": self.sim_threshold,
            "depth": self.depth,
            "masks": [[m.name, m.pattern] for m in self.masks],
            "drain": jsonpickle.dumps(self._miner.drain, keys=True),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zlib.compress(json.dumps(state).encode("utf-8")))

    @classmethod
    def load(cls, path: Path) -> LogParser:
        """Load a parser saved with `save`.

        jsonpickle can rebuild arbitrary Python objects, so ONLY load files this project
        created itself, never a model file from someone else.
        """
        state = json.loads(zlib.decompress(path.read_bytes()).decode("utf-8"))
        parser = cls(state["sim_threshold"], state["depth"], [Mask(*m) for m in state["masks"]])
        drain = jsonpickle.loads(state["drain"], keys=True)
        parser._miner.drain.id_to_cluster = drain.id_to_cluster
        parser._miner.drain.clusters_counter = drain.clusters_counter
        parser._miner.drain.root_node = drain.root_node
        return parser
