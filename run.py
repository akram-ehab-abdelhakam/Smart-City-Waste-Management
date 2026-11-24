"""
run.py
======
Task 6 - Simulation & Console Logging

Runs the smart-city waste-management model for a fixed number of ticks and
prints the agent interactions in a readable, step-by-step log - no GUI
required.

Usage
-----
    python run.py                  # 100 steps, default world
    python run.py --steps 50       # shorter run
    python run.py --seed 7         # different random layout

Each tick is printed only when something interesting happened, for example::

    Step 012
      - Bin 7 is full (81/100) -> Truck 9 won the bid (cost 7.0)
      - Truck 9 collecting at Bin 7: +95 waste (95/100)
      - Truck 9 is FULL (95/100) -> heading to the landfill
"""

import argparse

from model import WasteManagementModel


def parse_args() -> argparse.Namespace:
    """Read optional command-line settings (all have sensible defaults)."""
    parser = argparse.ArgumentParser(
        description="Smart City Waste Management - Mesa multi-agent simulation"
    )
    parser.add_argument("--steps", type=int, default=100, help="ticks to simulate")
    parser.add_argument("--width", type=int, default=20, help="grid width")
    parser.add_argument("--height", type=int, default=20, help="grid height")
    parser.add_argument("--bins", type=int, default=6, help="number of smart bins")
    parser.add_argument("--trucks", type=int, default=3, help="number of trucks")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def print_header(args: argparse.Namespace) -> None:
    """Print the run configuration."""
    print("=" * 64)
    print("Smart City Waste Management - Multi-Agent Simulation")
    print(
        f"Grid {args.width}x{args.height} | bins={args.bins} | "
        f"trucks={args.trucks} | steps={args.steps} | seed={args.seed}"
    )
    print("=" * 64)


def print_summary(model: WasteManagementModel) -> None:
    """Print final statistics once the simulation stops."""
    waste_in_trucks = sum(truck.load for truck in model.trucks)

    print("\n" + "=" * 64)
    print("Simulation finished")
    print("=" * 64)
    print(f"Steps run          : {model.step_count}")
    print(f"Waste collected    : {model.waste_collected}")
    print(f"Waste dumped       : {model.waste_dumped}")
    print(f"Waste still in trucks: {waste_in_trucks}")
    print(f"Landfill services  : {model.landfill.processed_trucks}")
    print(f"Landfill queue     : {model.landfill.queue_length}")

    print("\nBins")
    for bin_agent in model.bins:
        status = "FULL" if bin_agent.is_full else "ok"
        print(
            f"  Bin {bin_agent.unique_id:<2} level={bin_agent.waste_level:>3}"
            f"/{bin_agent.capacity} ({status})"
        )

    print("\nTrucks")
    for truck in model.trucks:
        print(
            f"  Truck {truck.unique_id:<2} state={truck.state:<12} "
            f"load={truck.load:>3}/{truck.capacity} pos={truck.pos}"
        )


def main() -> None:
    args = parse_args()

    model = WasteManagementModel(
        width=args.width,
        height=args.height,
        n_bins=args.bins,
        n_trucks=args.trucks,
        seed=args.seed,
    )

    print_header(args)

    # Main simulation loop -------------------------------------------------
    for step in range(1, args.steps + 1):
        model.step()

        # Only print ticks that produced events, so the log stays readable.
        if model.messages:
            print(f"\nStep {step:03d}")
            for message in model.messages:
                print(f"  - {message}")

    print_summary(model)


if __name__ == "__main__":
    main()
