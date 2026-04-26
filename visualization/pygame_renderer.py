# visualization/pygame_renderer.py
#
# Handles all Pygame drawing for the traffic simulation.
#
# This file only READS simulation state — it never modifies anything.
# The RL agent never interacts with this file at all.
# It exists purely so WE can watch what is happening.

import sys
import pygame
from utils.config import VisualizationConfig as VC, SimConfig


class PygameRenderer:
    """
    Draws the traffic simulation to a Pygame window.
    
    Created lazily — only when render_mode="human" is set.
    During training (500k steps) this class is never instantiated,
    keeping training as fast as possible.
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (VC.WINDOW_WIDTH, VC.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Traffic RL — Adaptive Signal Control")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", VC.FONT_SIZE)

    def render(self, road):
        """
        Main render function. Call once per simulation step.

        PARAMETERS:
          road — the Road object containing all simulation state

        ORDER MATTERS — DRAW IN STRICT LAYERS:
          Earlier versions did "for each intersection: draw roads then box then
          lights then vehicles" all in one pass. That's wrong for grid_size > 1
          because each intersection's road strips span the entire screen, so
          drawing roads for intersection (1,0) would paint over the intersection
          BOX of (0,0) that was drawn a moment earlier — the box gets erased
          and you see a continuous road where an intersection should be (the
          "underpass/bridge" visual bug).
          Fix: do ALL roads across ALL intersections first (pass 1), THEN all
          intersection boxes (pass 2), THEN all lights and vehicles (pass 3).
          That way every intersection box lands on top of every road strip.

        LAYER ORDER (background → foreground):
          1. Background fill
          2. Roads + center lines (every intersection)
          3. Intersection boxes (every intersection)
          4. Traffic lights + vehicles (every intersection)
          5. HUD text — always on top so it stays readable
        """
        # Handle window close button and ESC key
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    sys.exit()

        # --- layer 1: clear screen ---
        self.screen.fill(VC.COLOR_BACKGROUND)

        # --- layer 2: roads (and their center lines) for every intersection ---
        for intersection in road.intersections:
            self._draw_roads(intersection)

        # --- layer 3: intersection boxes on top of all roads ---
        for intersection in road.intersections:
            self._draw_intersection_box(intersection)

        # --- layer 4: traffic lights + vehicles on top of everything ---
        for intersection in road.intersections:
            self._draw_traffic_lights(intersection)
            self._draw_vehicles(intersection)

        # --- layer 5: HUD always on top ---
        self._draw_hud(road)

        # Push frame to screen
        pygame.display.flip()

        # Cap at FPS — slows simulation to human-watchable speed
        self.clock.tick(VC.FPS)

    def _draw_roads(self, intersection):
        """
        Draw the road surface as two crossing rectangles.
        One vertical strip (NS road) and one horizontal strip (EW road).
        """
        cx = int(intersection.cx)
        cy = int(intersection.cy)
        road_width = SimConfig.LANE_WIDTH * SimConfig.LANES_PER_DIRECTION * 2

        # North-South road — vertical strip full height of screen
        ns_rect = pygame.Rect(
            cx - road_width // 2, 0,
            road_width, VC.WINDOW_HEIGHT
        )
        pygame.draw.rect(self.screen, VC.COLOR_ROAD, ns_rect)

        # East-West road — horizontal strip full width of screen
        ew_rect = pygame.Rect(
            0, cy - road_width // 2,
            VC.WINDOW_WIDTH, road_width
        )
        pygame.draw.rect(self.screen, VC.COLOR_ROAD, ew_rect)

        # Draw dashed center lines on both roads
        self._draw_center_lines(cx, cy)

    def _draw_center_lines(self, cx, cy):
        """Draw dashed white lines down the center of each road."""
        dash = 15    # length of each dash in pixels
        gap  = 10    # gap between dashes
        color = VC.COLOR_LANE_MARKING

        # Vertical center line (down the NS road)
        y = 0
        while y < VC.WINDOW_HEIGHT:
            end_y = min(y + dash, VC.WINDOW_HEIGHT)
            pygame.draw.line(self.screen, color, (cx, y), (cx, end_y), 2)
            y += dash + gap

        # Horizontal center line (across the EW road)
        x = 0
        while x < VC.WINDOW_WIDTH:
            end_x = min(x + dash, VC.WINDOW_WIDTH)
            pygame.draw.line(self.screen, color, (x, cy), (end_x, cy), 2)
            x += dash + gap

    def _draw_intersection_box(self, intersection):
        """
        Draw the darker square where the two roads cross.
        This is the area vehicles pass through when crossing.
        """
        cx = int(intersection.cx)
        cy = int(intersection.cy)
        half = intersection.box_size // 2

        box = pygame.Rect(
            cx - half, cy - half,
            intersection.box_size, intersection.box_size
        )
        pygame.draw.rect(self.screen, VC.COLOR_INTERSECTION, box)

    def _draw_traffic_lights(self, intersection):
        """
        Draw 4 small circles near the intersection showing
        the current signal state for each direction.

        Positions:
          North light — above intersection
          South light — below intersection
          East light  — right of intersection
          West light  — left of intersection
        """
        from simulation.traffic_light import Phase

        cx = int(intersection.cx)
        cy = int(intersection.cy)
        phase = intersection.traffic_light.phase
        offset = intersection.box_size // 2 + 20
        radius = 8

        # Determine color for NS and EW directions
        if phase == Phase.GREEN_NS:
            ns_color = VC.COLOR_LIGHT_GREEN
            ew_color = VC.COLOR_LIGHT_RED
        elif phase == Phase.GREEN_EW:
            ns_color = VC.COLOR_LIGHT_RED
            ew_color = VC.COLOR_LIGHT_GREEN
        elif phase == Phase.YELLOW_NS:
            ns_color = VC.COLOR_LIGHT_YELLOW
            ew_color = VC.COLOR_LIGHT_RED
        elif phase == Phase.YELLOW_EW:
            ns_color = VC.COLOR_LIGHT_RED
            ew_color = VC.COLOR_LIGHT_YELLOW
        else:
            # ALL_RED phases
            ns_color = VC.COLOR_LIGHT_RED
            ew_color = VC.COLOR_LIGHT_RED

        # Draw each of the 4 light positions
        lights = {
            "N": ((cx, cy - offset), ns_color),
            "S": ((cx, cy + offset), ns_color),
            "E": ((cx + offset, cy), ew_color),
            "W": ((cx - offset, cy), ew_color),
        }

        for direction, (pos, color) in lights.items():
            # Dark background circle
            pygame.draw.circle(self.screen, VC.COLOR_LIGHT_OFF, pos, radius + 3)
            # Colored light on top
            pygame.draw.circle(self.screen, color, pos, radius)

    def _draw_vehicles(self, intersection):
        """
        Draw all vehicles — both those waiting in lanes
        and those crossing through the intersection.

        Vehicles in lanes are drawn as solid opaque rects. Vehicles in the
        box are drawn semi-transparent (see VC.BOX_RENDER_ALPHA) so that
        perpendicular crossings blend into each other visually instead of
        stacking as opaque "crashes." The sim itself never sees this —
        physics and metrics are computed from the true (x, y), unchanged.
        """
        # Vehicles in approach lanes — drawn solid at their real coordinates
        for lane_list in intersection.lanes.values():
            for lane in lane_list:
                for vehicle in lane.vehicles:
                    if vehicle.active:
                        self._draw_single_vehicle(vehicle)

        # Vehicles crossing the intersection box — drawn with alpha blending
        for vehicle in intersection.crossing_vehicles:
            if vehicle.active:
                self._draw_single_vehicle(vehicle, alpha=VC.BOX_RENDER_ALPHA)

    # --- Spawn-origin → color lookup (PHASE 2) ---
    # Keyed by vehicle.spawn_direction (the ORIGINAL heading, locked at spawn).
    # A car whose spawn_direction is "N" is heading north, which means it
    # entered from the SOUTH edge → FROM_S color. Same logic for the others.
    # This mapping is a class-level constant so we don't rebuild the dict on
    # every _draw_single_vehicle call (we draw 100+ cars per frame).
    _SPAWN_COLOR = {
        "N": VC.COLOR_VEHICLE_FROM_S,  # heading north ← came from south edge
        "S": VC.COLOR_VEHICLE_FROM_N,  # heading south ← came from north edge
        "E": VC.COLOR_VEHICLE_FROM_W,  # heading east  ← came from west edge
        "W": VC.COLOR_VEHICLE_FROM_E,  # heading west  ← came from east edge
    }

    def _draw_single_vehicle(self, vehicle, alpha: int = None):
        """
        Draw one vehicle as a colored rectangle.

        PARAMETERS:
          vehicle — the Vehicle to draw
          alpha   — optional 0-255 opacity. None = fully opaque (normal
                    pygame.draw.rect path, fastest). When a value is given
                    (e.g. VC.BOX_RENDER_ALPHA for cars inside the
                    intersection box), we route through a SRCALPHA
                    temporary surface so the paint actually blends with
                    whatever is already on screen — that's the whole point
                    of making crossing cars semi-transparent, so two
                    overlapping rects composite instead of stacking opaquely.
                    Slightly slower per call; only used for crossing cars
                    (typically <30 on screen at once on a 3x3 grid), so the
                    cost is invisible at 30 FPS.

        COLOR CODING (by SPAWN ORIGIN, not current heading):
          Cyan   — entered from south edge (spawned heading N)
          Pink   — entered from north edge (spawned heading S)
          Orange — entered from west edge  (spawned heading E)
          Lime   — entered from east edge  (spawned heading W)

        Why spawn origin and not current direction?
          Once Phase 2 turning kicked in, a single car can flip through all
          four headings across a 3x3 grid. If we colored by current heading
          the car would change color mid-trip, which is useless for watching
          flow. Coloring by origin lets you track one car visually the whole
          way across the grid, and also gives an instant eyeball on balance —
          "am I getting equal inflow from all four edges?".

        Stopped vehicles are drawn slightly darker so you can visually see
        where queues are forming.
        """
        rx, ry, rw, rh = vehicle.get_rect()

        # Fall back to the old heading-based color if spawn_direction is
        # missing (defensive — shouldn't happen once all vehicles go through
        # the current constructor, but saves us a crash if some test stub
        # constructs a bare Vehicle without it).
        origin = getattr(vehicle, "spawn_direction", vehicle.direction)
        color = self._SPAWN_COLOR.get(origin, VC.COLOR_VEHICLE_NS)

        # Darken color slightly if stopped
        if vehicle.is_stopped:
            color = tuple(max(0, c - 50) for c in color)

        w = max(2, int(rw))
        h = max(2, int(rh))

        if alpha is None:
            # --- Opaque path (lane cars) — fastest, straight to the screen ---
            pygame.draw.rect(self.screen, color, (int(rx), int(ry), w, h))
        else:
            # --- Alpha-blended path (crossing cars) ---
            # pygame.draw.rect on self.screen doesn't honor a 4th alpha
            # channel — we have to paint onto a SRCALPHA surface first and
            # blit it. Two overlapping semi-transparent blits composite
            # correctly because the screen surface retains the first paint
            # as the "below" layer when the second blit lands on top.
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.fill((color[0], color[1], color[2], alpha))
            self.screen.blit(surf, (int(rx), int(ry)))

    def _draw_hud(self, road):
        """
        Draw heads-up display with real-time metrics in top left corner.

        Shows step count, current phase, queue lengths per direction,
        and total waiting time — the key metrics for understanding
        what the RL agent is responding to.
        """
        inter   = road.intersections[0]
        queues  = inter.get_queue_lengths()
        metrics = road.get_metrics()

        lines = [
            f"Step    : {metrics['step']:>6}",
            f"Phase   : {inter.traffic_light.phase.name}",
            f"Waiting : {metrics['total_waiting_time']:>8.0f}",
            f"Queue   : {metrics['total_queue_length']:>6}",
            f"N:{queues['N']:>3}  S:{queues['S']:>3}",
            f"E:{queues['E']:>3}  W:{queues['W']:>3}",
            f"Spawned : {metrics['total_spawned']:>6}",
        ]

        # Semi-transparent dark background for readability
        hud_w = 210
        hud_h = len(lines) * 22 + 14
        hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surface.fill((20, 20, 20, 180))
        self.screen.blit(hud_surface, (8, 8))

        # Draw each line of text
        for i, line in enumerate(lines):
            text = self.font.render(line, True, VC.COLOR_TEXT)
            self.screen.blit(text, (15, 15 + i * 22))

    def close(self):
        """Shut down Pygame cleanly."""
        pygame.quit()

       