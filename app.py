"""
app.py
======
Task 7 - Solara Visualization

Browser front-end for the Smart City Waste Management simulation.

Why this file does not use ``JupyterViz`` directly
--------------------------------------------------
Mesa 2.4 ships its Solara integration as ``mesa.experimental.jupyter_viz.
JupyterViz`` (``SolaraViz`` only exists in Mesa 3.x). When served by
``solara run``, JupyterViz switches to a "browser" layout that builds its
cards with ``ipyvuetify``: those widgets do not display with the Solara
version installed here, which leaves the grid area blank (only the sidebar
worked). This module therefore uses the same underlying stack that Mesa's
SolaraViz uses in Mesa 3.x - native Solara components plus
``solara.FigureMatplotlib`` for the grid - and keeps the same
``agent_portrayal`` / model-parameter design.

Run it with::

    python -m solara run app.py

Then open http://localhost:8765 in the browser.
"""

import time

from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch, Rectangle

import solara

from agents import LandfillAgent, SmartBinAgent, TruckAgent
from model import WasteManagementModel


# Fixed world size (as requested).
WIDTH = 20
HEIGHT = 20

# Delay between automatic ticks while playing (seconds).
PLAY_INTERVAL = 0.2


# ----------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------
BIN_GREEN = "#43a047"     # empty / filling normally
BIN_YELLOW = "#f9a825"    # getting full
BIN_RED = "#d32f2f"       # full and waiting for a truck

LANDFILL_BROWN = "#5d4037"

TRUCK_COLORS = {
    TruckAgent.IDLE: "#1e88e5",          # blue
    TruckAgent.TO_BIN: "#fb8c00",        # orange
    TruckAgent.COLLECTING: "#fdd835",    # yellow (working on site)
    TruckAgent.TO_LANDFILL: "#8e24aa",   # purple
    TruckAgent.QUEUED: "#757575",        # grey (waiting in the FIFO queue)
    TruckAgent.DUMPING: "#ab47bc",       # light purple
}


# ----------------------------------------------------------------------
# Agent portrayal
# ----------------------------------------------------------------------
def agent_portrayal(agent) -> dict:
    """Map an agent to its visual representation.

    Returns a dictionary with a shape, a colour and a human-readable label.
    The ``color``/``size`` entries also keep it compatible with Mesa's
    built-in matplotlib/Altair drawers.
    """
    # --- Smart bins: circles, colour = fill level ---------------------
    if isinstance(agent, SmartBinAgent):
        if agent.is_full:
            color = BIN_RED
            status = "full"
        elif agent.fill_ratio >= 0.5:
            color = BIN_YELLOW
            status = "filling"
        else:
            color = BIN_GREEN
            status = "ok"

        return {
            "shape": "circle",
            "color": color,
            "size": 120,
            "status": status,
            "waste": agent.waste_level,
            "label": f"Bin {agent.unique_id} ({agent.waste_level}/{agent.capacity})",
        }

    # --- Trucks: rectangles, colour = state ---------------------------
    if isinstance(agent, TruckAgent):
        color = TRUCK_COLORS.get(agent.state, "#424242")
        return {
            "shape": "rect",
            "color": color,
            "size": 160,
            "status": agent.state,
            "load": agent.load,
            "label": (
                f"Truck {agent.unique_id} ({agent.state}, "
                f"{agent.load}/{agent.capacity})"
            ),
        }

    # --- Landfill: big static square ----------------------------------
    if isinstance(agent, LandfillAgent):
        return {
            "shape": "square",
            "color": LANDFILL_BROWN,
            "size": 600,
            "status": "landfill",
            "queue": agent.queue_length,
            "label": f"Landfill (queue: {agent.queue_length})",
        }

    # --- Fallback (should not happen) ---------------------------------
    return {"shape": "circle", "color": "#9e9e9e", "size": 80}


# ----------------------------------------------------------------------
# Custom space drawer (real circles / rectangles / squares)
# ----------------------------------------------------------------------
def _draw_space(model, agent_portrayal_func) -> Figure:
    """Build the matplotlib figure for the current model state.

    Mesa's built-in matplotlib drawer only supports scatter circles, so this
    custom drawer is used to honour the requested shapes (circle bins,
    rectangle trucks, square landfill).
    """
    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.subplots()

    # Collect every agent once per redraw.
    agents = []
    for cell_contents, _pos in model.grid.coord_iter():
        agents.extend(cell_contents)

    # Draw the landfill first (background layer), then bins, then trucks.
    for agent in agents:
        if isinstance(agent, LandfillAgent) and agent.pos is not None:
            x, y = agent.pos
            ax.add_patch(
                Rectangle(
                    (x - 0.45, y - 0.45),
                    0.9,
                    0.9,
                    facecolor=LANDFILL_BROWN,
                    edgecolor="#3e2723",
                    linewidth=1.2,
                    zorder=1,
                )
            )

    for agent in agents:
        if isinstance(agent, SmartBinAgent) and agent.pos is not None:
            data = agent_portrayal_func(agent)
            x, y = agent.pos
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=0.32,
                    facecolor=data["color"],
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=2,
                )
            )

    for agent in agents:
        if isinstance(agent, TruckAgent) and agent.pos is not None:
            data = agent_portrayal_func(agent)
            x, y = agent.pos
            ax.add_patch(
                Rectangle(
                    (x - 0.40, y - 0.26),
                    0.80,
                    0.52,
                    facecolor=data["color"],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )
            )

    # Grid cosmetics.
    ax.set_xlim(-0.5, model.width - 0.5)
    ax.set_ylim(-0.5, model.height - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Step {model.step_count}", fontsize=10)

    # Simple legend so the colour code is readable without extra docs.
    legend_handles = [
        Patch(color=BIN_GREEN, label="Bin (ok)"),
        Patch(color=BIN_YELLOW, label="Bin (filling)"),
        Patch(color=BIN_RED, label="Bin (full)"),
        Patch(color=TRUCK_COLORS[TruckAgent.IDLE], label="Truck (idle)"),
        Patch(color=TRUCK_COLORS[TruckAgent.TO_BIN], label="Truck (to bin)"),
        Patch(color=TRUCK_COLORS[TruckAgent.TO_LANDFILL], label="Truck (to landfill)"),
        Patch(color=LANDFILL_BROWN, label="Landfill"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=7,
        frameon=False,
    )

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Solara entry point
# ----------------------------------------------------------------------
@solara.component
def Page():
    """Root component picked up by ``solara run app.py``."""
    # --- Reactive UI state --------------------------------------------
    n_bins = solara.use_reactive(6)
    n_trucks = solara.use_reactive(3)
    reset_counter = solara.use_reactive(0)
    playing = solara.use_reactive(False)
    tick = solara.use_reactive(0)  # forces a redraw after a manual step

    # The model is rebuilt whenever a parameter or the reset counter changes.
    model = solara.use_memo(
        lambda: WasteManagementModel(
            width=WIDTH,
            height=HEIGHT,
            n_bins=n_bins.value,
            n_trucks=n_trucks.value,
            seed=42,
        ),
        dependencies=[n_bins.value, n_trucks.value, reset_counter.value],
    )

    # --- Control callbacks --------------------------------------------
    def start():
        playing.value = True

    def pause():
        playing.value = False

    def step_once():
        playing.value = False
        model.step()
        tick.value = model.step_count

    def reset():
        playing.value = False
        reset_counter.value += 1
        tick.value = 0

    def on_bins(value: int):
        playing.value = False
        n_bins.value = value
        tick.value = 0

    def on_trucks(value: int):
        playing.value = False
        n_trucks.value = value
        tick.value = 0

    # --- Automatic play loop ------------------------------------------
    # ``use_thread`` runs this generator in a background thread and asks
    # Solara to re-render after every ``yield``. The thread is cancelled
    # automatically when ``playing`` or the model changes.
    def play_loop(cancel):
        while playing.value and not cancel.is_set():
            model.step()
            yield None
            time.sleep(PLAY_INTERVAL)

    solara.use_thread(play_loop, dependencies=[playing.value, model])

    # --- Layout -------------------------------------------------------
    with solara.AppBar():
        solara.AppBarTitle("Smart City Waste Management")

    with solara.Sidebar():
        with solara.Card("Controls", margin=1):
            solara.SliderInt(
                "Number of bins",
                value=n_bins.value,
                min=1,
                max=20,
                step=1,
                on_value=on_bins,
            )
            solara.SliderInt(
                "Number of trucks",
                value=n_trucks.value,
                min=1,
                max=10,
                step=1,
                on_value=on_trucks,
            )
            with solara.Row():
                if playing.value:
                    solara.Button("Pause", color="primary", on_click=pause)
                else:
                    solara.Button("Start", color="primary", on_click=start)
                solara.Button("Step", color="secondary", on_click=step_once)
                solara.Button("Reset", color="error", on_click=reset)
            solara.Markdown(
                f"**Step:** {model.step_count}  \n"
                f"**Grid:** {WIDTH} x {HEIGHT} (fixed)  \n"
                f"**Landfill queue:** {model.landfill.queue_length}"
            )

        with solara.Card("Legend", margin=1):
            solara.Markdown(
                "- Green / yellow / red circle: bin ok / filling / full\n"
                "- Blue rectangle: truck idle\n"
                "- Orange rectangle: truck driving to a bin\n"
                "- Purple rectangle: truck driving to the landfill\n"
                "- Grey rectangle: truck queued at the landfill\n"
                "- Brown square: landfill"
            )

    with solara.Card("City Grid", margin=2):
        fig = _draw_space(model, agent_portrayal)
        solara.FigureMatplotlib(
            fig,
            format="png",
            dependencies=[tick.value, model.step_count],
        )
