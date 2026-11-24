"""
agents.py
=========
Task 2 - Agent Classes

Defines the three agent types of the smart-city waste-management system:

* ``SmartBinAgent``    - Proactive agent: watches its own fill level and will
                         start a Contract Net Protocol when it gets full.
* ``TruckAgent``       - Utility-based agent: bids for collection jobs and
                         drives around the grid to do them.
* ``LandfillAgent``     - Reactive coordinator: owns a FIFO queue that decides
                         which full truck may dump its waste.

Task 3 implements the Contract Net Protocol (CFP -> propose -> accept) using
plain direct method calls between agents. Task 4 adds truck movement and bin
collection. Task 5 adds the landfill FIFO queue that coordinates full trucks
(a "social law": only one truck may dump at a time).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional

import mesa

if TYPE_CHECKING:  # imported only for type hints (avoids circular imports)
    from model import WasteManagementModel


def manhattan_distance(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> int:
    """Grid distance between two cells (|dx| + |dy|).

    Manhattan distance is enough for this simulation: trucks move one cell
    per tick along the axes, so this is the real number of ticks needed.
    """
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


class SmartBinAgent(mesa.Agent):
    """A static bin that monitors its own waste level.

    Parameters
    ----------
    unique_id : int
        Mesa identifier for this agent.
    model : WasteManagementModel
        The simulation the bin belongs to.
    capacity : int
        Maximum amount of waste the bin can hold.
    threshold : float
        Fraction of ``capacity`` that triggers a Call for Proposal (0.8 = 80%).
    generation_rate : int
        Waste units added to the bin on every tick (citizens throwing rubbish).
    """

    def __init__(
        self,
        unique_id: int,
        model: "WasteManagementModel",
        capacity: int = 100,
        threshold: float = 0.8,
        generation_rate: int = 2,
    ) -> None:
        super().__init__(unique_id, model)

        # --- Physical properties -----------------------------------------
        self.capacity = capacity
        self.waste_level = self.model.random.randint(0, int(capacity * 0.6))
        self.generation_rate = generation_rate

        # --- Contract Net Protocol state ---------------------------------
        self.threshold = threshold
        self.task_open = False        # True while a CFP is awaiting bids
        self.assigned_truck: Optional["TruckAgent"] = None

    @property
    def fill_ratio(self) -> float:
        """Current fill level as a fraction of the bin capacity (0.0 - 1.0)."""
        return self.waste_level / self.capacity

    @property
    def is_full(self) -> bool:
        """True when the bin has reached the CFP threshold (e.g. 80%)."""
        return self.fill_ratio >= self.threshold

    def step(self) -> None:
        """Proactive behaviour: generate waste, then start a CFP if full."""
        self.waste_level = min(self.capacity, self.waste_level + self.generation_rate)
        if self.is_full and self.assigned_truck is None:
            self.broadcast_cfp()

    # ------------------------------------------------------------------
    # Contract Net Protocol - initiator side
    # ------------------------------------------------------------------
    def broadcast_cfp(self) -> None:
        """Send a Call for Proposal to every truck and evaluate the replies.

        Direct method calls act as the message transport: each truck is asked
        for a proposal synchronously, which keeps the logic easy to follow
        (no message queues or event bus needed).
        """
        # Never run two negotiations for the same bin at the same time.
        if self.task_open or self.assigned_truck is not None:
            return

        self.task_open = True

        # Collect bids. ``None`` means "this truck declines the job".
        bids: dict["TruckAgent", float] = {}
        for truck in self.model.trucks:
            cost = truck.propose(self)
            if cost is not None:
                bids[truck] = cost

        winner = self.accept_proposal(bids)
        if winner is None:
            # Nobody could help right now (all trucks busy/too full).
            # Re-open the opportunity on the next tick.
            self.task_open = False

    def accept_proposal(self, bids: dict["TruckAgent", float]) -> Optional["TruckAgent"]:
        """Award the task to the truck with the lowest bid (lowest cost).

        Returns the winning truck, or ``None`` when there were no bids.
        """
        if not bids:
            return None

        winner = min(bids, key=bids.get)
        winning_cost = bids[winner]

        self.task_open = False
        self.assigned_truck = winner

        # Tell the winner it has the job (direct method call = message).
        winner.receive_accept_proposal(self)

        self.model.log(
            f"Bin {self.unique_id} is full ({self.waste_level}/{self.capacity}) "
            f"-> Truck {winner.unique_id} won the bid (cost {winning_cost:.1f})"
        )
        return winner


class TruckAgent(mesa.Agent):
    """A collection truck that moves on the grid and carries waste.

    Parameters
    ----------
    unique_id : int
        Mesa identifier for this agent.
    model : WasteManagementModel
        The simulation the truck belongs to.
    capacity : int
        Maximum load the truck can carry before it must visit the landfill.
    """

    # Possible truck states (plain strings keep the state machine obvious).
    IDLE = "idle"                 # nothing to do
    TO_BIN = "to_bin"             # driving to an assigned bin
    COLLECTING = "collecting"     # standing on the bin, loading waste
    TO_LANDFILL = "to_landfill"   # full, driving to the landfill
    QUEUED = "queued"             # waiting in the landfill FIFO queue
    DUMPING = "dumping"           # unloading at the landfill

    def __init__(
        self,
        unique_id: int,
        model: "WasteManagementModel",
        capacity: int = 100,
    ) -> None:
        super().__init__(unique_id, model)

        # --- Physical properties -----------------------------------------
        self.capacity = capacity
        self.load = 0                 # waste currently carried
        self.speed = 1                # cells moved per tick

        # --- Task / state-machine properties ------------------------------
        self.state = self.IDLE
        self.target_bin: Optional["SmartBinAgent"] = None
        self.last_bid = 0             # most recent bid sent (for logging)

    # ------------------------------------------------------------------
    # Convenience properties used by the bidding and movement logic
    # ------------------------------------------------------------------
    @property
    def free_capacity(self) -> int:
        """How much more waste the truck can still pick up."""
        return self.capacity - self.load

    @property
    def is_full(self) -> bool:
        """True when the truck can no longer collect any waste."""
        return self.free_capacity <= 0

    @property
    def is_busy(self) -> bool:
        """True when the truck is already committed to a job/queue."""
        return self.state != self.IDLE

    @property
    def needs_dump(self) -> bool:
        """True when the truck should go to the landfill.

        A truck is considered "full" in practice when it is physically full
        **or** when it no longer has enough room to empty even one full bin.
        This prevents a truck from idling forever at, for example, 95/100.
        """
        return self.is_full or self.free_capacity < self.model.min_bin_waste

    # ------------------------------------------------------------------
    # Contract Net Protocol - responder side
    # ------------------------------------------------------------------
    def propose(self, bin_agent: "SmartBinAgent") -> Optional[float]:
        """Return this truck's bid for collecting ``bin_agent``, or ``None``.

        Bid = Manhattan distance to the bin + a penalty for the load the
        truck is already carrying. Both terms are expressed in "cell units"
        so neither one completely dominates the other:

        * farther bins cost more (longer trip),
        * fuller trucks cost more (less room left = worse candidate).
        """
        # A busy truck cannot take a second job, and a truck must be able to
        # take *all* the waste in the bin so the bin can be emptied in one go.
        if self.is_busy or self.free_capacity < bin_agent.waste_level:
            return None

        distance = manhattan_distance(self.pos, bin_agent.pos)

        # Map the used capacity (0..capacity) onto a 0..max_dimension penalty.
        max_dimension = max(self.model.width, self.model.height)
        load_penalty = (self.load / self.capacity) * max_dimension

        self.last_bid = round(distance + load_penalty, 2)
        return self.last_bid

    def receive_accept_proposal(self, bin_agent: "SmartBinAgent") -> None:
        """Accept the job: remember the target and start driving (Task 4)."""
        self.target_bin = bin_agent
        self.state = self.TO_BIN

    def step(self) -> None:
        """Truck behaviour, chosen by the current state."""
        if self.state == self.TO_BIN:
            self._service_bin()
        elif self.state == self.TO_LANDFILL:
            self._drive_to_landfill()
        elif self.state == self.QUEUED:
            # Waiting in the landfill FIFO queue. The landfill releases us.
            pass
        elif self.state == self.IDLE and self.needs_dump:
            # Safety net (e.g. a truck that ended a job with little room).
            self.state = self.TO_LANDFILL

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------
    def _move_towards(self, target_pos: tuple[int, int]) -> None:
        """Move a single cell closer to ``target_pos`` (Manhattan style).

        One axis is corrected per tick: first X, then Y. This is enough for
        this simulation and guarantees the truck never overshoots the target.
        """
        x, y = self.pos
        target_x, target_y = target_pos

        new_x, new_y = x, y
        if x != target_x:
            new_x += 1 if target_x > x else -1
        elif y != target_y:
            new_y += 1 if target_y > y else -1

        if (new_x, new_y) != (x, y):
            # MultiGrid.move_agent removes the truck from its old cell and
            # places it on the new one; the bin can share the cell.
            self.model.grid.move_agent(self, (new_x, new_y))

    def _service_bin(self) -> None:
        """Drive to the assigned bin and empty it on arrival."""
        bin_agent = self.target_bin
        if bin_agent is None:
            # Safety net: no target means nothing to do.
            self.state = self.IDLE
            return

        # Not there yet -> take one step.
        if self.pos != bin_agent.pos:
            self._move_towards(bin_agent.pos)
            return

        # --- Arrived: collect as much as fits -------------------------
        collected = min(self.free_capacity, bin_agent.waste_level)
        self.load += collected
        bin_agent.waste_level -= collected
        self.model.waste_collected += collected

        # Release the bin so it can ask for another truck later.
        bin_agent.assigned_truck = None
        bin_agent.task_open = False
        self.target_bin = None

        self.model.log(
            f"Truck {self.unique_id} collecting at Bin {bin_agent.unique_id}: "
            f"+{collected} waste ({self.load}/{self.capacity})"
        )

        if self.needs_dump:
            self.state = self.TO_LANDFILL
            self.model.log(
                f"Truck {self.unique_id} is FULL ({self.load}/{self.capacity}) "
                f"-> heading to the landfill"
            )
        else:
            self.state = self.IDLE

    # ------------------------------------------------------------------
    # Landfill coordination (shared resource)
    # ------------------------------------------------------------------
    def _drive_to_landfill(self) -> None:
        """Drive to the landfill, then join its FIFO queue."""
        landfill = self.model.landfill
        if landfill is None:
            self.state = self.IDLE
            return

        # Still on the road -> move one cell closer.
        if self.pos != landfill.pos:
            self._move_towards(landfill.pos)
            return

        # Arrived: request permission to dump (the "social law").
        landfill.enqueue(self)
        self.state = self.QUEUED
        self.model.log(
            f"Truck {self.unique_id} requests to dump at Landfill "
            f"{landfill.unique_id} (queue position {landfill.queue_length})"
        )


class LandfillAgent(mesa.Agent):
    """The city dump. Coordinates full trucks through a FIFO queue.

    This agent is a *reactive coordinator*: it does not look for work, it only
    reacts to trucks that request permission to dump.
    """

    def __init__(self, unique_id: int, model: "WasteManagementModel") -> None:
        super().__init__(unique_id, model)

        # --- Shared-resource coordination --------------------------------
        # ``deque`` gives O(1) append (arrival) and popleft (FIFO service).
        self.queue: deque["TruckAgent"] = deque()
        self.state = "idle"           # "idle" or "serving"
        self.processed_trucks = 0     # statistics for the console log
        self.total_waste_received = 0

    def enqueue(self, truck: "TruckAgent") -> None:
        """Register a full truck at the back of the FIFO queue.

        Called by the truck when it reaches the landfill (Task 5).
        """
        if truck not in self.queue:
            self.queue.append(truck)

    @property
    def queue_length(self) -> int:
        """Number of trucks currently waiting to dump."""
        return len(self.queue)

    def step(self) -> None:
        """Serve exactly one queued truck per tick (FIFO social law).

        Serving a single truck per tick guarantees that two trucks never
        unload at the same time, which is the collision-avoidance rule of
        this shared resource.
        """
        if not self.queue:
            self.state = "idle"
            return

        self.state = "serving"
        truck: "TruckAgent" = self.queue.popleft()

        # Unload everything the truck is carrying and free its capacity.
        dumped = truck.load
        truck.load = 0
        truck.target_bin = None
        truck.last_bid = 0
        truck.state = TruckAgent.IDLE

        # Update statistics.
        self.processed_trucks += 1
        self.total_waste_received += dumped
        self.model.waste_dumped += dumped

        self.model.log(
            f"Landfill {self.unique_id} served Truck {truck.unique_id}: "
            f"dumped {dumped} waste (queue left: {self.queue_length})"
        )
