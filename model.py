"""
model.py
========
Task 1 - Environment & Base Setup
(+ road network and data collection)

This module defines ``WasteManagementModel``, the environment that hosts
every agent in the simulation.

The model is intentionally small and modular:

* A ``MultiGrid`` is used because several agents may share a cell
  (e.g. a truck standing on a bin, or trucks queuing on the landfill).
* A ``RandomActivation`` scheduler activates every agent once per tick,
  in a fresh random order, which keeps the simulation fair between agents.
* The model owns the shared, non-agent state (statistics, references to
  the bins/trucks/landfill) so that agents never have to search the whole
  grid to find each other.

The world is a small city map: roads run along a regular grid of rows and
columns, and their crossings form intersections. Trucks, bins and the
landfill are placed on the road network, and trucks use a breadth-first
search over road cells to plan their route (see ``next_road_step``).

A ``DataCollector`` records one row of statistics per tick so the front-end
and the saved artifacts can plot the simulation history.
"""

from collections import deque

import mesa
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

# Imported here so the model can create the agents.
# These classes are implemented in Task 2 (agents.py).
from agents import LandfillAgent, SmartBinAgent, TruckAgent


class WasteManagementModel(mesa.Model):
    """The smart-city waste-management environment.

    Parameters
    ----------
    width, height : int
        Size of the grid in cells.
    n_bins : int
        Number of ``SmartBinAgent`` instances to scatter along the roads.
    n_trucks : int
        Number of ``TruckAgent`` instances to place along the roads.
    road_spacing : int
        Distance between parallel roads. Roads exist at every
        ``road_spacing``-th row and column, so the crossings are intersections.
    seed : int | None
        Optional random seed, forwarded to Mesa for reproducible runs.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        n_bins: int = 6,
        n_trucks: int = 3,
        road_spacing: int = 5,
        seed: int | None = None,
    ) -> None:
        # ``mesa.Model.__init__`` creates the seeded RNG used by every agent
        # (``self.random``) and the unique-id generator (``self.next_id()``).
        super().__init__(seed=seed)

        # ------------------------------------------------------------------
        # 1. Environment: grid + scheduler
        # ------------------------------------------------------------------
        self.width = width
        self.height = height

        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)

        # Handy flag for run.py / the visualization.
        self.running = True

        # ------------------------------------------------------------------
        # 2. Road network (the "map")
        # ------------------------------------------------------------------
        if road_spacing < 1:
            raise ValueError("road_spacing must be a positive integer")

        self.road_spacing = road_spacing
        # Columns and rows that are roads. Intersections are the crossings.
        self.road_x = set(range(0, width, road_spacing))
        self.road_y = set(range(0, height, road_spacing))
        self.road_positions = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if self.is_road((x, y))
        ]

        # ------------------------------------------------------------------
        # 3. Shared references and statistics
        # ------------------------------------------------------------------
        # Agents are collected here so other agents can talk to them with a
        # simple direct method call instead of scanning the grid.
        self.landfill: LandfillAgent | None = None
        self.bins: list[SmartBinAgent] = []
        self.trucks: list[TruckAgent] = []

        # Simple counters used for console logs and statistics.
        self.step_count = 0
        self.waste_collected = 0   # waste picked up from bins
        self.waste_dumped = 0      # waste delivered to the landfill

        # Human-readable events produced during the current tick. run.py
        # prints (and then discards) them; keeping logging out of the agents
        # makes the simulation usable with or without a console.
        self.messages: list[str] = []

        # ------------------------------------------------------------------
        # 4. Populate the world
        # ------------------------------------------------------------------
        self._place_landfill()
        self._place_bins(n_bins)

        # Smallest amount of waste that can make a bin request a truck.
        # Trucks use this to decide when they are effectively "full"
        # (i.e. cannot empty even one bin any more).
        self.min_bin_waste = int(
            min((b.threshold * b.capacity for b in self.bins), default=0)
        )

        self._place_trucks(n_trucks)

        # ------------------------------------------------------------------
        # 5. Data collection (one row per reported tick)
        # ------------------------------------------------------------------
        self.datacollector = DataCollector(
            model_reporters={
                "Collected": lambda m: m.waste_collected,
                "Dumped": lambda m: m.waste_dumped,
                "FullBins": lambda m: sum(1 for b in m.bins if b.is_full),
                "AvgBinFill": lambda m: (
                    sum(b.fill_ratio for b in m.bins) / len(m.bins) if m.bins else 0.0
                ),
                "TruckLoad": lambda m: sum(t.load for t in m.trucks),
                "TruckUtilization": lambda m: (
                    sum(t.load for t in m.trucks)
                    / sum(t.capacity for t in m.trucks)
                    if m.trucks
                    else 0.0
                ),
                "LandfillQueue": lambda m: m.landfill.queue_length,
                "Served": lambda m: m.landfill.processed_trucks,
            }
        )
        # Record the initial state as well (step 0).
        self.datacollector.collect(self)

    # ----------------------------------------------------------------------
    # Road network helpers
    # ----------------------------------------------------------------------
    def is_road(self, pos: tuple[int, int]) -> bool:
        """True when ``pos`` lies on a road (or at an intersection)."""
        x, y = pos
        return x in self.road_x or y in self.road_y

    def is_intersection(self, pos: tuple[int, int]) -> bool:
        """True when ``pos`` is where a vertical and a horizontal road cross."""
        x, y = pos
        return x in self.road_x and y in self.road_y

    def next_road_step(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> tuple[int, int]:
        """Return the next cell on a shortest road path from start to goal.

        A breadth-first search over the road cells keeps trucks on the
        streets (they cannot cut through city blocks). The grid is tiny
        (hundreds of cells), so recomputing this every tick is cheap and
        avoids storing stale paths when a truck switches target.
        """
        if start == goal or goal is None:
            return start

        frontier: deque[tuple[int, int]] = deque([start])
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while frontier:
            current = frontier.popleft()
            if current == goal:
                # Walk backwards to the cell right after ``start``.
                step = current
                while came_from[step] is not None and came_from[step] != start:
                    step = came_from[step]
                return step

            cx, cy = current
            neighbours = [
                (cx + 1, cy),
                (cx - 1, cy),
                (cx, cy + 1),
                (cx, cy - 1),
            ]
            # Shuffle so equal-length routes vary between trucks/runs.
            self.random.shuffle(neighbours)

            for nxt in neighbours:
                nx, ny = nxt
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and self.is_road(nxt)
                    and nxt not in came_from
                ):
                    came_from[nxt] = current
                    frontier.append(nxt)

        # No road path found (should not happen in a normal map).
        return start

    # ----------------------------------------------------------------------
    # Placement helpers
    # ----------------------------------------------------------------------
    def _random_empty_road_cell(self) -> tuple[int, int]:
        """Return a random road cell that currently contains no agent."""
        candidates = [
            pos
            for pos in self.road_positions
            if not self.grid.get_cell_list_contents([pos])
        ]
        if not candidates:
            raise RuntimeError(
                "No empty road cell available: reduce n_bins/n_trucks "
                "or reduce road_spacing."
            )
        return self.random.choice(candidates)

    def _place_landfill(self) -> None:
        """Create the landfill on the intersection nearest to the centre."""
        centre_x = min(self.road_x, key=lambda x: abs(x - self.width // 2))
        centre_y = min(self.road_y, key=lambda y: abs(y - self.height // 2))

        landfill = LandfillAgent(self.next_id(), self)
        self.grid.place_agent(landfill, (centre_x, centre_y))
        self.schedule.add(landfill)

        self.landfill = landfill

    def _place_bins(self, n_bins: int) -> None:
        """Scatter ``n_bins`` smart bins on random empty road cells."""
        for _ in range(n_bins):
            bin_agent = SmartBinAgent(self.next_id(), self)
            self.grid.place_agent(bin_agent, self._random_empty_road_cell())
            self.schedule.add(bin_agent)
            self.bins.append(bin_agent)

    def _place_trucks(self, n_trucks: int) -> None:
        """Place ``n_trucks`` collection trucks on random empty road cells."""
        for _ in range(n_trucks):
            truck = TruckAgent(self.next_id(), self)
            self.grid.place_agent(truck, self._random_empty_road_cell())
            self.schedule.add(truck)
            self.trucks.append(truck)

    # ----------------------------------------------------------------------
    # Logging helper
    # ----------------------------------------------------------------------
    def log(self, message: str) -> None:
        """Record one event for the current tick (printed by run.py)."""
        self.messages.append(message)

    # ----------------------------------------------------------------------
    # Simulation step
    # ----------------------------------------------------------------------
    def step(self) -> None:
        """Advance the whole simulation by one tick.

        ``RandomActivation`` calls ``step()`` on every agent exactly once,
        in a fresh random order each tick. Mesa's scheduler automatically
        advances ``model._steps`` (used by the Solara front-end), so
        ``step_count`` is synchronised after the schedule has run.
        """
        self.messages.clear()   # only keep the events of the current tick

        self.schedule.step()

        # Keep our human-friendly counter in sync with Mesa's internal one.
        self.step_count = self._steps

        # Record one row of statistics for this tick.
        self.datacollector.collect(self)

