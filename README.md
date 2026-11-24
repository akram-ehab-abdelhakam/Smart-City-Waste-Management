# 🏙️ Smart City Waste Management - Multi-Agent System (MAS)

An autonomous, decentralized multi-agent simulation built with **Python**, **Mesa**, and **Solara**. This project models a smart city waste management ecosystem where agents coordinate dynamically without a central controller to optimize waste collection.

## 🚀 Key Features & AI Concepts Applied

This project practically implements core Agent-Based Systems (ABS) and Distributed AI concepts:
* **Contract Net Protocol (CNP):** Used for decentralized task allocation. Bins initiate a Call for Proposal (CFP), and trucks submit bids based on their current state and distance.
* **Proactive Agents:** Smart bins monitor their own internal state and take the initiative to request collection when a critical threshold is reached.
* **Utility-Based Decision Making:** Collection trucks calculate their expected utility (cost) using a heuristic combining Manhattan distance and remaining capacity.
* **Shared Resource Coordination (Social Laws):** The landfill acts as a reactive coordinator, managing a FIFO queue to prevent collisions and enforce orderly dumping.

## 🤖 The Agents

1. **SmartBinAgent (Proactive Initiator)**
   * Static agents that generate waste over time.
   * Triggers a CFP broadcast when waste levels reach ≥80%.
   * Evaluates incoming bids from trucks and awards the task to the lowest bidder.

2. **TruckAgent (Utility-Based Contractor)**
   * Mobile agents patrolling the city grid.
   * Evaluates CFPs and submits bids combining `Manhattan Distance + Load Penalty`.
   * Navigates to assigned bins, collects waste, and dynamically routes to the landfill when full.

3. **LandfillAgent (Reactive Coordinator)**
   * The central dumping facility.
   * Maintains a strictly enforced First-In-First-Out (FIFO) queue, allowing only one truck to dump per time step.

## 🛠️ Installation & Requirements

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install mesa>=2.4.0 solara matplotlib