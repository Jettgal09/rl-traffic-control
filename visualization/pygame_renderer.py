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

        ORDER MATTERS:
          We draw from background to foreground.
          Roads first, then intersection box on top,
          then traffic lights, then vehicles on top of everything,
          then HUD text last so it's always readable.
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

        # Clear screen with background color
        self.screen.fill(VC.COLOR_BACKGROUND)

        # Draw each intersection and everything around it
        for intersection in road.intersections:
            self._draw_roads(intersection)
            self._draw_intersection_box(intersection)
            self._draw_traffic_lights(intersection)
            self._draw_vehicles(intersection)

        # Draw HUD on top of everything
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
        """
        # Vehicles in approach lanes
        for lane_list in intersection.lanes.values():
            for lane in lane_list:
                for vehicle in lane.vehicles:
                    if vehicle.active:
                        self._draw_single_vehicle(vehicle)

        # Vehicles crossing the intersection box
        for vehicle in intersection.crossing_vehicles:
            if vehicle.active:
                self._draw_single_vehicle(vehicle)

    def _draw_single_vehicle(self, vehicle):
        """
        Draw one vehicle as a colored rectangle.

        COLOR CODING:
          Blue   — North or South traveling vehicles
          Orange — East or West traveling vehicles

        Stopped vehicles are drawn slightly darker
        so you can visually see where queues are forming.
        """
        rx, ry, rw, rh = vehicle.get_rect()
        color = (
            VC.COLOR_VEHICLE_NS
            if vehicle.direction in ("N", "S")
            else VC.COLOR_VEHICLE_EW
        )

        # Darken color slightly if stopped
        if vehicle.is_stopped:
            color = tuple(max(0, c - 50) for c in color)

        pygame.draw.rect(
            self.screen,
            color,
            (int(rx), int(ry), max(2, int(rw)), max(2, int(rh)))
        )

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

       