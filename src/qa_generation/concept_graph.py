from dataclasses import dataclass

@dataclass
class ConceptGraph:
    instance_graph: dict[str, list[str]] = None
    part_graph: dict[str, list[str]] = None