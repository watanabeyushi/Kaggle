"""
Orbit Wars - Strategic v3 agent

This version keeps the accurate moving-target hit logic from v1 and layers on:
- speed-threshold aware waiting
- mission scoring that values early optionality
- future-position based target evaluation
- rescue / recapture / reinforcement heuristics
- gateway planet bonuses for chaining future captures
"""

import math
from collections import defaultdict

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD_MIN = 0.0
BOARD_MAX = 100.0
LAUNCH_CLEARANCE = 0.1
BASE_DEFENSE_MARGIN = 8
AIM_ITERATIONS = 3
DEFAULT_MAX_SPEED = 6.0
STRICT_VALIDATION_THRESHOLD = 55.0
WAIT_LOOKAHEAD = 3
ETA_IMPROVEMENT_STEPS = 4
MAX_FORECAST_TURNS = 18
CLUSTER_RADIUS = 24.0
GATEWAY_RADIUS = 28.0


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        value = obs.get(key, default)
    else:
        value = getattr(obs, key, default)
    return default if value is None else value


def _build_angular_velocity_map(angular_velocity):
    if isinstance(angular_velocity, dict):
        return {int(k): float(v) for k, v in angular_velocity.items()}, 0.0
    if isinstance(angular_velocity, (list, tuple)):
        return {i: float(v) for i, v in enumerate(angular_velocity)}, 0.0
    if angular_velocity is None:
        return {}, 0.0
    return {}, float(angular_velocity)


def _is_orbiting_initial(planet_row):
    if not isinstance(planet_row, (list, tuple)) or len(planet_row) < 7:
        return False
    x = float(planet_row[2])
    y = float(planet_row[3])
    radius = float(planet_row[4])
    orbital_radius = math.hypot(x - SUN_X, y - SUN_Y)
    return orbital_radius + radius < 50.0


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class PlanetState:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production", "is_comet")

    def __init__(self, record, is_comet=False):
        (
            self.id,
            self.owner,
            self.x,
            self.y,
            self.radius,
            self.ships,
            self.production,
        ) = record
        self.is_comet = is_comet


class FleetState:
    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")

    def __init__(self, record):
        self.id, self.owner, self.x, self.y, self.angle, self.from_planet_id, self.ships = record


class SpatialBody:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production")

    def __init__(self, planet, x, y):
        self.id = int(planet.id)
        self.owner = int(planet.owner)
        self.x = float(x)
        self.y = float(y)
        self.radius = float(planet.radius)
        self.ships = int(planet.ships)
        self.production = int(planet.production)


class GameState:
    def __init__(self, obs):
        self.player = int(_obs_get(obs, "player", 0))
        self.step = int(_obs_get(obs, "step", 0) or 0)
        self.initial_planets = _obs_get(obs, "initial_planets", []) or []
        angular_velocity = _obs_get(obs, "angular_velocity", 0.0)
        self.angular_velocity_map, self.global_angular_velocity = _build_angular_velocity_map(
            angular_velocity
        )
        comet_ids = set(_obs_get(obs, "comet_planet_ids", []) or [])
        self.planets = [PlanetState(p, p[0] in comet_ids) for p in (_obs_get(obs, "planets", []) or [])]
        self.fleets = [FleetState(f) for f in (_obs_get(obs, "fleets", []) or [])]
        self.planets_by_id = {planet.id: planet for planet in self.planets}
        self.my_planets = [planet for planet in self.planets if planet.owner == self.player]
        self.targets = [planet for planet in self.planets if planet.owner != self.player]
        self.enemy_planets = [planet for planet in self.planets if planet.owner not in (-1, self.player)]

    def get_planet(self, planet_id):
        return self.planets_by_id.get(planet_id)

    def orbit_row(self, planet):
        if 0 <= int(planet.id) < len(self.initial_planets):
            row = self.initial_planets[int(planet.id)]
            if isinstance(row, (list, tuple)) and len(row) >= 7:
                return row
        return None

    def is_orbiting(self, planet):
        if planet.is_comet:
            return False
        row = self.orbit_row(planet)
        return _is_orbiting_initial(row)

    def angular_velocity(self, planet):
        return float(self.angular_velocity_map.get(int(planet.id), self.global_angular_velocity))

    def predict_position(self, planet, future_turns):
        future_turns = max(0.0, float(future_turns))
        row = self.orbit_row(planet)
        if row is None or not self.is_orbiting(planet):
            return float(planet.x), float(planet.y)

        x0 = float(row[2])
        y0 = float(row[3])
        omega = self.angular_velocity(planet)
        if abs(omega) < 1e-12:
            return x0, y0

        orbital_radius = math.hypot(x0 - SUN_X, y0 - SUN_Y)
        theta0 = math.atan2(y0 - SUN_Y, x0 - SUN_X)
        theta = theta0 + omega * (self.step + future_turns)
        return SUN_X + orbital_radius * math.cos(theta), SUN_Y + orbital_radius * math.sin(theta)

    def future_body(self, planet, future_turns):
        x, y = self.predict_position(planet, future_turns)
        return SpatialBody(planet, x, y)


def launch_point(body, angle, clearance=LAUNCH_CLEARANCE):
    launch_radius = float(body.radius) + float(clearance)
    return body.x + math.cos(angle) * launch_radius, body.y + math.sin(angle) * launch_radius


def closest_point_on_segment(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = clamp(t, 0.0, 1.0)
    return x1 + t * dx, y1 + t * dy


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    cx, cy = closest_point_on_segment(px, py, x1, y1, x2, y2)
    return math.hypot(px - cx, py - cy)


def ray_circle_hit_distance(src_x, src_y, dir_x, dir_y, center_x, center_y, radius):
    dx = center_x - src_x
    dy = center_y - src_y
    proj = dx * dir_x + dy * dir_y
    if proj < 0:
        return None

    perp_sq = dx * dx + dy * dy - proj * proj
    radius_sq = radius * radius
    if perp_sq > radius_sq:
        return None

    return max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))


def is_in_bounds(x, y):
    return BOARD_MIN <= x <= BOARD_MAX and BOARD_MIN <= y <= BOARD_MAX


def estimate_fleet_speed(num_ships, max_speed=DEFAULT_MAX_SPEED):
    ships = max(1, int(num_ships))
    if ships == 1:
        return 1.0
    speed_ratio = math.log(ships) / math.log(1000)
    speed_ratio = clamp(speed_ratio, 0.0, 1.0)
    return 1.0 + (max_speed - 1.0) * (speed_ratio ** 1.5)


def ships_for_speed(required_speed, max_speed=DEFAULT_MAX_SPEED):
    required_speed = clamp(float(required_speed), 1.0, max_speed)
    if required_speed <= 1.0 + 1e-9:
        return 1
    ratio = (required_speed - 1.0) / (max_speed - 1.0)
    ratio = clamp(ratio, 0.0, 1.0)
    log_ratio = ratio ** (1.0 / 1.5)
    ships = math.exp(math.log(1000.0) * log_ratio)
    return max(1, int(math.ceil(ships)))


def fleet_position_after_time(launch_x, launch_y, angle, num_ships, time_elapsed):
    speed = estimate_fleet_speed(num_ships)
    return (
        launch_x + math.cos(angle) * speed * time_elapsed,
        launch_y + math.sin(angle) * speed * time_elapsed,
    )


def segment_hits_sun(src_x, src_y, dst_x, dst_y):
    dx = dst_x - src_x
    dy = dst_y - src_y
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return math.hypot(src_x - SUN_X, src_y - SUN_Y) <= SUN_RADIUS

    t = ((SUN_X - src_x) * dx + (SUN_Y - src_y) * dy) / seg_len_sq
    t = clamp(t, 0.0, 1.0)
    closest_x = src_x + t * dx
    closest_y = src_y + t * dy
    return math.hypot(closest_x - SUN_X, closest_y - SUN_Y) <= SUN_RADIUS


def validate_intercept_solution(
    state,
    source,
    target,
    num_ships,
    angle,
    launch_delay,
    max_turns,
    validation_mode="strict",
):
    source_body = state.future_body(source, launch_delay)
    launch_x, launch_y = launch_point(source_body, angle)
    speed = estimate_fleet_speed(num_ships)
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    prev_x, prev_y = launch_x, launch_y

    for turn in range(1, max(1, int(max_turns)) + 1):
        curr_x, curr_y = fleet_position_after_time(launch_x, launch_y, angle, num_ships, turn)
        target_before_x, target_before_y = state.predict_position(target, launch_delay + turn - 1)
        target_after_x, target_after_y = state.predict_position(target, launch_delay + turn)

        hit_distance = ray_circle_hit_distance(
            prev_x,
            prev_y,
            dir_x,
            dir_y,
            target_before_x,
            target_before_y,
            target.radius,
        )
        if hit_distance is not None and hit_distance <= speed + 1e-6:
            hit_x = prev_x + dir_x * hit_distance
            hit_y = prev_y + dir_y * hit_distance
            return {
                "valid": True,
                "time": float(turn - 1 + (hit_distance / max(speed, 1e-9))),
                "eta": int(turn),
                "pred_x": float(target_before_x),
                "pred_y": float(target_before_y),
                "aim_x": float(hit_x),
                "aim_y": float(hit_y),
                "launch_x": float(launch_x),
                "launch_y": float(launch_y),
                "angle": float(angle),
                "uses_sweep": False,
                "launch_delay": int(launch_delay),
            }

        if validation_mode == "relaxed":
            relaxed_hit_distance = ray_circle_hit_distance(
                prev_x,
                prev_y,
                dir_x,
                dir_y,
                target_after_x,
                target_after_y,
                target.radius,
            )
            if relaxed_hit_distance is not None and relaxed_hit_distance <= speed + 1e-6:
                hit_x = prev_x + dir_x * relaxed_hit_distance
                hit_y = prev_y + dir_y * relaxed_hit_distance
                return {
                    "valid": True,
                    "time": float(turn - 1 + (relaxed_hit_distance / max(speed, 1e-9))),
                    "eta": int(turn),
                    "pred_x": float(target_after_x),
                    "pred_y": float(target_after_y),
                    "aim_x": float(hit_x),
                    "aim_y": float(hit_y),
                    "launch_x": float(launch_x),
                    "launch_y": float(launch_y),
                    "angle": float(angle),
                    "uses_sweep": False,
                    "launch_delay": int(launch_delay),
                }

        if point_to_segment_distance(
            curr_x, curr_y, target_before_x, target_before_y, target_after_x, target_after_y
        ) <= target.radius + 1e-6:
            sweep_hit_x, sweep_hit_y = closest_point_on_segment(
                curr_x, curr_y, target_before_x, target_before_y, target_after_x, target_after_y
            )
            return {
                "valid": True,
                "time": float(turn),
                "eta": int(turn),
                "pred_x": float(target_after_x),
                "pred_y": float(target_after_y),
                "aim_x": float(sweep_hit_x),
                "aim_y": float(sweep_hit_y),
                "launch_x": float(launch_x),
                "launch_y": float(launch_y),
                "angle": float(angle),
                "uses_sweep": True,
                "launch_delay": int(launch_delay),
            }

        if not is_in_bounds(curr_x, curr_y):
            break

        prev_x, prev_y = curr_x, curr_y

    return None


def estimate_precise_intercept(
    state,
    source,
    target,
    num_ships,
    launch_delay=0,
    validation_mode="strict",
):
    source_body = state.future_body(source, launch_delay)
    target_x0, target_y0 = state.predict_position(target, launch_delay)
    speed = estimate_fleet_speed(num_ships)
    time_to_hit = max(0.0, math.hypot(target_x0 - source_body.x, target_y0 - source_body.y) / max(speed, 1e-9))
    angle = math.atan2(target_y0 - source_body.y, target_x0 - source_body.x)
    launch_x, launch_y = launch_point(source_body, angle)
    pred_x, pred_y = target_x0, target_y0

    for _ in range(AIM_ITERATIONS):
        pred_x, pred_y = state.predict_position(target, launch_delay + time_to_hit)
        angle = math.atan2(pred_y - source_body.y, pred_x - source_body.x)
        launch_x, launch_y = launch_point(source_body, angle)
        time_to_hit = math.hypot(pred_x - launch_x, pred_y - launch_y) / max(speed, 1e-9)

    eta_turns = max(1, int(math.ceil(time_to_hit)))
    direct_validation = validate_intercept_solution(
        state,
        source,
        target,
        num_ships,
        angle,
        launch_delay,
        max_turns=max(1, eta_turns + 2),
        validation_mode=validation_mode,
    )

    sweep_validation = None
    sweep_start_x, sweep_start_y = state.predict_position(target, launch_delay + max(0.0, eta_turns - 1))
    sweep_end_x, sweep_end_y = state.predict_position(target, launch_delay + eta_turns)
    sweep_motion = math.hypot(sweep_end_x - sweep_start_x, sweep_end_y - sweep_start_y)
    if sweep_motion > 1e-6:
        sweep_aim_x, sweep_aim_y = closest_point_on_segment(
            pred_x, pred_y, sweep_start_x, sweep_start_y, sweep_end_x, sweep_end_y
        )
        sweep_angle = math.atan2(sweep_aim_y - source_body.y, sweep_aim_x - source_body.x)
        sweep_validation = validate_intercept_solution(
            state,
            source,
            target,
            num_ships,
            sweep_angle,
            launch_delay,
            max_turns=max(1, eta_turns + 2),
            validation_mode=validation_mode,
        )

    if direct_validation is not None and (
        sweep_validation is None or direct_validation["time"] <= sweep_validation["time"]
    ):
        return direct_validation
    if sweep_validation is not None:
        return sweep_validation

    return {
        "valid": False,
        "time": float(time_to_hit),
        "eta": int(max(1, math.ceil(time_to_hit))),
        "pred_x": float(pred_x),
        "pred_y": float(pred_y),
        "aim_x": float(pred_x),
        "aim_y": float(pred_y),
        "launch_x": float(launch_x),
        "launch_y": float(launch_y),
        "angle": float(angle),
        "uses_sweep": False,
        "launch_delay": int(launch_delay),
    }


def infer_fleet_target_and_eta(state, fleet):
    best_planet = None
    best_eta = None
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = estimate_fleet_speed(fleet.ships)

    for planet in state.planets:
        eta_time = math.hypot(planet.x - fleet.x, planet.y - fleet.y) / max(speed, 1e-9)
        hit_distance = None

        for _ in range(AIM_ITERATIONS):
            pred_x, pred_y = state.predict_position(planet, eta_time)
            hit_distance = ray_circle_hit_distance(
                fleet.x, fleet.y, dir_x, dir_y, pred_x, pred_y, planet.radius
            )
            if hit_distance is None:
                break
            eta_time = hit_distance / max(speed, 1e-9)

        if hit_distance is None:
            continue

        eta_turns = max(1, int(math.ceil(eta_time)))
        if best_eta is None or eta_turns < best_eta:
            best_eta = eta_turns
            best_planet = planet

    return best_planet, best_eta


def build_arrivals_by_target(state):
    arrivals_by_target = defaultdict(list)
    for fleet in state.fleets:
        target_planet, fleet_eta = infer_fleet_target_and_eta(state, fleet)
        if target_planet is None or fleet_eta is None:
            continue
        arrivals_by_target[target_planet.id].append((int(fleet_eta), int(fleet.owner), int(fleet.ships)))
    return arrivals_by_target


def resolve_arrival_event(owner, garrison, arrivals):
    by_owner = defaultdict(int)
    for attacker_owner, ships in arrivals:
        by_owner[int(attacker_owner)] += int(ships)

    if not by_owner:
        return int(owner), max(0, int(garrison))

    sorted_forces = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_forces[0]

    if len(sorted_forces) > 1:
        second_ships = sorted_forces[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return int(owner), max(0, int(garrison))

    if int(owner) == int(survivor_owner):
        return int(owner), int(garrison) + survivor_ships

    remaining_garrison = int(garrison) - survivor_ships
    if remaining_garrison < 0:
        return int(survivor_owner), -remaining_garrison
    return int(owner), remaining_garrison


def forecast_planet_state(planet, eta_turns, arrivals, planned_arrivals, player):
    owner = int(planet.owner)
    garrison = int(planet.ships)
    arrivals_by_turn = defaultdict(list)

    for eta, owner_id, ships in arrivals:
        if eta <= eta_turns:
            arrivals_by_turn[int(eta)].append((int(owner_id), int(ships)))
    for eta, ships in planned_arrivals:
        if eta <= eta_turns:
            arrivals_by_turn[int(eta)].append((int(player), int(ships)))

    for turn in range(1, int(eta_turns) + 1):
        if owner != -1:
            garrison += int(planet.production)
        turn_arrivals = arrivals_by_turn.get(turn, [])
        if turn_arrivals:
            owner, garrison = resolve_arrival_event(owner, garrison, turn_arrivals)

    return owner, garrison


class WorldModel:
    def __init__(self, state):
        self.state = state
        self.arrivals_by_target = build_arrivals_by_target(state)
        self.enemy_arrivals_by_target = defaultdict(list)
        self.friendly_arrivals_by_target = defaultdict(list)
        for target_id, arrivals in self.arrivals_by_target.items():
            for eta, owner, ships in arrivals:
                if owner == self.state.player:
                    self.friendly_arrivals_by_target[target_id].append((eta, owner, ships))
                else:
                    self.enemy_arrivals_by_target[target_id].append((eta, owner, ships))

    def forecast_state(self, planet, eta_turns, planned_arrivals_by_target):
        planned = planned_arrivals_by_target.get(planet.id, [])
        arrivals = self.arrivals_by_target.get(planet.id, [])
        return forecast_planet_state(planet, eta_turns, arrivals, planned, self.state.player)

    def projected_loss_turn(self, planet, planned_arrivals_by_target, max_turns=MAX_FORECAST_TURNS):
        if planet.owner != self.state.player:
            return None
        for turn in range(1, max_turns + 1):
            owner, _ = self.forecast_state(planet, turn, planned_arrivals_by_target)
            if owner != self.state.player:
                return turn
        return None

    def soonest_enemy_eta(self, planet_id):
        arrivals = self.enemy_arrivals_by_target.get(planet_id, [])
        if not arrivals:
            return None
        return min(eta for eta, _, _ in arrivals)

    def distance_to_closest_enemy(self, planet, future_turns=0.0):
        if not self.state.enemy_planets:
            return float("inf")
        px, py = self.state.predict_position(planet, future_turns)
        best = float("inf")
        for enemy in self.state.enemy_planets:
            ex, ey = self.state.predict_position(enemy, future_turns)
            best = min(best, math.hypot(ex - px, ey - py))
        return best

    def local_cluster_stats(self, planet, future_turns=0.0):
        px, py = self.state.predict_position(planet, future_turns)
        count = 0
        prod_sum = 0.0
        best_gap = float("inf")
        for other in self.state.planets:
            if other.id == planet.id:
                continue
            ox, oy = self.state.predict_position(other, future_turns)
            dist = math.hypot(ox - px, oy - py)
            best_gap = min(best_gap, dist)
            if dist <= CLUSTER_RADIUS:
                count += 1
                prod_sum += float(other.production)
        return count, prod_sum, best_gap

    def enemy_competition_gap(self, source, target, future_turns=0.0):
        tx, ty = self.state.predict_position(target, future_turns)
        sx, sy = self.state.predict_position(source, 0.0)
        my_dist = math.hypot(tx - sx, ty - sy)
        enemy_best = float("inf")
        for enemy in self.state.enemy_planets:
            ex, ey = self.state.predict_position(enemy, 0.0)
            enemy_best = min(enemy_best, math.hypot(tx - ex, ty - ey))
        if enemy_best == float("inf"):
            return 20.0
        return enemy_best - my_dist

    def gateway_branch_score(self, anchor, future_turns):
        ax, ay = self.state.predict_position(anchor, future_turns)
        best = 0.0
        for other in self.state.targets:
            if other.id == anchor.id:
                continue
            ox, oy = self.state.predict_position(other, future_turns)
            dist = math.hypot(ox - ax, oy - ay)
            if dist > GATEWAY_RADIUS:
                continue
            candidate = (other.production * 4.0 + max(0.0, 12.0 - other.ships) * 0.4) / max(6.0, dist)
            best = max(best, candidate)
        return best

    def gateway_bonus(self, target, eta_turns):
        samples = [2, 3, 5, 8, 12, 18]
        if self.state.step >= 80:
            samples.extend([24, 32])
        if self.state.step >= 160:
            samples.extend([48, 64])

        bonus = 0.0
        for dt in samples:
            horizon = eta_turns + dt
            branch = self.gateway_branch_score(target, horizon)
            growth = target.production * min(12, dt) * 0.04
            bonus = max(bonus, branch + growth)
        return bonus


def dynamic_defense_margin(state, world, source):
    margin = BASE_DEFENSE_MARGIN + int(source.production)
    if state.step < 60:
        margin += 2
    if world.distance_to_closest_enemy(source) < 18.0:
        margin += 4
    if len(state.my_planets) <= 3:
        margin += 3
    return max(4, margin)


def available_to_send(state, world, source, wait_turns=0):
    future_ships = int(source.ships) + int(source.production) * int(wait_turns)
    return max(0, future_ships - dynamic_defense_margin(state, world, source))


def compute_validation_pressure(target, ships_needed, eta_turns, mission_type):
    ownership_multiplier = 1.5 if int(target.owner) != -1 else 1.0
    mission_multiplier = 1.35 if mission_type in ("capture", "recapture", "snipe") else 0.9
    value = max(1.0, float(target.production))
    cost = max(1.0, float(ships_needed) * max(1.0, float(eta_turns)))
    return ownership_multiplier * mission_multiplier * cost / value


def choose_validation_mode(target, ships_needed, eta_turns, mission_type):
    pressure = compute_validation_pressure(target, ships_needed, eta_turns, mission_type)
    return "strict" if pressure >= STRICT_VALIDATION_THRESHOLD else "relaxed"


def threshold_ship_options(distance, min_ships, available):
    min_ships = max(1, int(min_ships))
    available = max(min_ships, int(available))
    options = {min_ships}
    base_eta = max(1, int(math.ceil(distance / max(estimate_fleet_speed(min_ships), 1e-9))))
    for improved_eta in range(max(1, base_eta - ETA_IMPROVEMENT_STEPS), base_eta):
        required_speed = distance / max(1, improved_eta)
        ships = ships_for_speed(required_speed)
        if min_ships <= ships <= available:
            options.add(ships)
    for delta in (5, 10, 20, 35):
        ships = min_ships + delta
        if ships <= available:
            options.add(ships)
    options.add(available)
    return sorted(options)


def choose_preferred_launch(
    state,
    world,
    source,
    target,
    min_ships,
    mission_type,
    max_wait=WAIT_LOOKAHEAD,
):
    best_now = None
    best_future = None

    for wait_turns in range(0, max_wait + 1):
        available = available_to_send(state, world, source, wait_turns)
        if available < min_ships:
            continue

        source_x, source_y = state.predict_position(source, wait_turns)
        target_x, target_y = state.predict_position(target, wait_turns)
        distance = math.hypot(target_x - source_x, target_y - source_y)
        ship_options = threshold_ship_options(distance, min_ships, available)

        local_best = None
        for ships in ship_options:
            probe = estimate_precise_intercept(
                state,
                source,
                target,
                ships,
                launch_delay=wait_turns,
                validation_mode=choose_validation_mode(target, ships, 8, mission_type),
            )
            if not probe.get("valid", False):
                continue
            if segment_hits_sun(probe["launch_x"], probe["launch_y"], probe["aim_x"], probe["aim_y"]):
                continue

            candidate = {
                "ships": int(ships),
                "eta": int(probe["eta"]),
                "wait": int(wait_turns),
                "arrival_turn": int(wait_turns + probe["eta"]),
                "angle": float(probe["angle"]),
                "solution": probe,
            }
            if local_best is None:
                local_best = candidate
                continue
            if candidate["arrival_turn"] < local_best["arrival_turn"]:
                local_best = candidate
                continue
            if candidate["arrival_turn"] == local_best["arrival_turn"] and candidate["ships"] < local_best["ships"]:
                local_best = candidate

        if local_best is None:
            continue

        if wait_turns == 0:
            best_now = local_best
        elif best_future is None or local_best["arrival_turn"] < best_future["arrival_turn"]:
            best_future = local_best
        elif (
            local_best["arrival_turn"] == best_future["arrival_turn"]
            and local_best["ships"] < best_future["ships"]
        ):
            best_future = local_best

    if best_future is not None and (
        best_now is None or best_future["arrival_turn"] < best_now["arrival_turn"]
    ):
        return None
    return best_now


def tempo_multiplier(step, eta_turns):
    eta_turns = max(1, int(eta_turns))
    base = 1.0 / (1.0 + 0.12 * eta_turns)
    if step < 60:
        base *= 1.0 / (1.0 + 0.07 * eta_turns)
    elif step < 120:
        base *= 1.0 / (1.0 + 0.04 * eta_turns)
    return base


def source_optionality_cost(state, world, source):
    early_weight = clamp((70.0 - state.step) / 70.0, 0.0, 1.0)
    frontline = 1.0 / max(8.0, world.distance_to_closest_enemy(source))
    scarcity = 1.4 if len(state.my_planets) <= 3 else (1.15 if len(state.my_planets) <= 5 else 1.0)
    production_weight = 1.0 + 0.25 * float(source.production)
    return (5.0 + 8.0 * early_weight * scarcity * frontline) * production_weight


def target_stability_cost(state, world, source, target, eta_turns):
    competition_gap = world.enemy_competition_gap(source, target, eta_turns)
    instability = 0.0
    if competition_gap < 0.0:
        instability += min(14.0, -competition_gap * 0.45)
    if target.owner not in (-1, state.player):
        instability += 1.5
    enemy_eta = world.soonest_enemy_eta(target.id)
    if enemy_eta is not None and enemy_eta <= eta_turns + 2:
        instability += 2.5
    if world.distance_to_closest_enemy(target, eta_turns) < 15.0:
        instability += 3.0
    return instability


def target_value(state, world, source, target, eta_turns):
    cluster_count, cluster_prod, nearest_gap = world.local_cluster_stats(target, eta_turns)
    gateway = world.gateway_bonus(target, eta_turns)
    competition_gap = world.enemy_competition_gap(source, target, eta_turns)

    base_value = float(target.production) * 8.0
    future_income = float(target.production) * max(4.0, 18.0 - float(eta_turns)) * 0.55
    low_garrison_bonus = max(0.0, 16.0 - float(target.ships)) * 0.45
    owner_bonus = 2.5 if target.owner not in (-1, state.player) else 0.0
    cluster_bonus = 1.0 + 0.05 * cluster_count + 0.015 * cluster_prod
    density_bonus = 1.0 if nearest_gap == float("inf") else 1.0 + max(0.0, 12.0 - nearest_gap) * 0.01
    competition_multiplier = 1.0 + clamp(competition_gap / 50.0, -0.25, 0.2)

    value = (base_value + future_income + low_garrison_bonus + owner_bonus + gateway)
    value *= cluster_bonus * density_bonus * competition_multiplier * tempo_multiplier(state.step, eta_turns)
    return value


def defended_planet_value(state, world, planet):
    cluster_count, cluster_prod, _ = world.local_cluster_stats(planet, 0.0)
    frontline_pressure = 1.0 / max(8.0, world.distance_to_closest_enemy(planet))
    return (
        float(planet.production) * 10.0
        + float(planet.ships) * 0.15
        + cluster_prod * 0.8
        + cluster_count * 1.5
        + 40.0 * frontline_pressure
    )


def build_capture_candidate(state, world, source, target, planned_arrivals_by_target):
    initial_need = max(1, int(target.ships) + 1)
    launch = choose_preferred_launch(state, world, source, target, initial_need, "capture")
    if launch is None:
        return None

    owner_at_eta, garrison_at_eta = world.forecast_state(target, launch["arrival_turn"], planned_arrivals_by_target)
    if owner_at_eta == state.player:
        return None

    ships_needed = int(garrison_at_eta) + 1
    launch = choose_preferred_launch(state, world, source, target, ships_needed, "capture")
    if launch is None:
        return None

    owner_at_eta, garrison_at_eta = world.forecast_state(target, launch["arrival_turn"], planned_arrivals_by_target)
    if owner_at_eta == state.player:
        return None

    ships_needed = int(garrison_at_eta) + 1
    if launch["ships"] < ships_needed:
        return None

    strategic_value = target_value(state, world, source, target, launch["arrival_turn"])
    strategic_value -= source_optionality_cost(state, world, source)
    strategic_value -= target_stability_cost(state, world, source, target, launch["arrival_turn"])
    score = strategic_value / max(1.0, launch["ships"] * max(1, launch["arrival_turn"]))
    if score <= 0.0:
        return None

    return {
        "type": "capture",
        "source_id": int(source.id),
        "target_id": int(target.id),
        "ships": int(ships_needed),
        "eta": int(launch["eta"]),
        "wait": int(launch["wait"]),
        "arrival_turn": int(launch["arrival_turn"]),
        "angle": float(launch["angle"]),
        "score": float(score),
    }


def desired_defense_garrison(state, world, target, desired_eta):
    margin = BASE_DEFENSE_MARGIN + int(target.production) * 2
    if world.distance_to_closest_enemy(target, desired_eta) < 18.0:
        margin += 6
    if state.step < 80:
        margin += 2
    return margin


def build_rescue_candidate(state, world, source, target, planned_arrivals_by_target):
    if source.id == target.id:
        return None

    soonest_enemy = world.soonest_enemy_eta(target.id)
    loss_turn = world.projected_loss_turn(target, planned_arrivals_by_target, max_turns=12)
    if soonest_enemy is None and loss_turn is None:
        return None

    desired_eta = 6
    if soonest_enemy is not None:
        desired_eta = min(desired_eta, max(1, soonest_enemy))
    if loss_turn is not None:
        desired_eta = min(desired_eta, max(1, loss_turn))

    owner_at_eta, garrison_at_eta = world.forecast_state(target, desired_eta, planned_arrivals_by_target)
    if owner_at_eta != state.player:
        return None

    desired_garrison = desired_defense_garrison(state, world, target, desired_eta)
    ships_needed = max(1, desired_garrison - int(garrison_at_eta))
    if ships_needed <= 0:
        return None

    launch = choose_preferred_launch(state, world, source, target, ships_needed, "rescue", max_wait=2)
    if launch is None:
        return None
    if loss_turn is not None and launch["arrival_turn"] > loss_turn:
        return None

    if launch["ships"] < ships_needed:
        return None

    saved_value = defended_planet_value(state, world, target)
    saved_value += 5.0 if loss_turn is not None else 0.0
    saved_value -= source_optionality_cost(state, world, source) * 0.7
    score = saved_value / max(1.0, launch["ships"] * max(1, launch["arrival_turn"]))
    if score <= 0.0:
        return None

    return {
        "type": "rescue",
        "source_id": int(source.id),
        "target_id": int(target.id),
        "ships": int(ships_needed),
        "eta": int(launch["eta"]),
        "wait": int(launch["wait"]),
        "arrival_turn": int(launch["arrival_turn"]),
        "angle": float(launch["angle"]),
        "score": float(score + 0.2),
    }


def build_recapture_candidate(state, world, source, target, planned_arrivals_by_target):
    loss_turn = world.projected_loss_turn(target, planned_arrivals_by_target, max_turns=12)
    if loss_turn is None:
        return None

    min_probe_need = max(1, int(target.ships) + 1)
    launch = choose_preferred_launch(state, world, source, target, min_probe_need, "recapture")
    if launch is None:
        return None
    if launch["arrival_turn"] < loss_turn + 1 or launch["arrival_turn"] > loss_turn + 4:
        return None

    owner_at_eta, garrison_at_eta = world.forecast_state(target, launch["arrival_turn"], planned_arrivals_by_target)
    if owner_at_eta == state.player:
        return None

    ships_needed = int(garrison_at_eta) + 1
    launch = choose_preferred_launch(state, world, source, target, ships_needed, "recapture")
    if launch is None:
        return None
    if launch["arrival_turn"] < loss_turn + 1 or launch["arrival_turn"] > loss_turn + 4:
        return None

    owner_at_eta, garrison_at_eta = world.forecast_state(target, launch["arrival_turn"], planned_arrivals_by_target)
    if owner_at_eta == state.player:
        return None
    ships_needed = int(garrison_at_eta) + 1
    if launch["ships"] < ships_needed:
        return None

    strategic_value = defended_planet_value(state, world, target) + target_value(
        state, world, source, target, launch["arrival_turn"]
    )
    strategic_value -= source_optionality_cost(state, world, source)
    score = strategic_value / max(1.0, launch["ships"] * max(1, launch["arrival_turn"]))
    if score <= 0.0:
        return None

    return {
        "type": "recapture",
        "source_id": int(source.id),
        "target_id": int(target.id),
        "ships": int(ships_needed),
        "eta": int(launch["eta"]),
        "wait": int(launch["wait"]),
        "arrival_turn": int(launch["arrival_turn"]),
        "angle": float(launch["angle"]),
        "score": float(score + 0.12),
    }


def better_candidate(lhs, rhs):
    if rhs is None:
        return lhs
    if lhs is None:
        return rhs
    if rhs["score"] != lhs["score"]:
        return rhs if rhs["score"] > lhs["score"] else lhs
    if rhs["arrival_turn"] != lhs["arrival_turn"]:
        return rhs if rhs["arrival_turn"] < lhs["arrival_turn"] else lhs
    return rhs if rhs["ships"] < lhs["ships"] else lhs


def record_candidate(moves, used_source_ids, planned_arrivals_by_target, candidate):
    moves.append([candidate["source_id"], candidate["angle"], candidate["ships"]])
    used_source_ids.add(candidate["source_id"])
    planned_arrivals_by_target[candidate["target_id"]].append(
        (int(candidate["arrival_turn"]), int(candidate["ships"]))
    )


def source_priority(state, world, planet):
    available = available_to_send(state, world, planet, 0)
    enemy_dist = world.distance_to_closest_enemy(planet)
    return (available, -enemy_dist, planet.production)


def agent(obs):
    state = GameState(obs)
    if not state.my_planets:
        return []

    world = WorldModel(state)
    moves = []
    planned_arrivals_by_target = defaultdict(list)
    used_source_ids = set()
    ordered_sources = sorted(
        state.my_planets,
        key=lambda planet: source_priority(state, world, planet),
        reverse=True,
    )

    for source in ordered_sources:
        if source.id in used_source_ids or available_to_send(state, world, source, 0) <= 0:
            continue
        best = None
        for target in state.my_planets:
            if target.id == source.id:
                continue
            best = better_candidate(best, build_rescue_candidate(state, world, source, target, planned_arrivals_by_target))
        if best is not None and best["score"] > 0.18:
            record_candidate(moves, used_source_ids, planned_arrivals_by_target, best)

    for source in ordered_sources:
        if source.id in used_source_ids or available_to_send(state, world, source, 0) <= 0:
            continue
        best = None
        for target in state.my_planets:
            if target.id == source.id:
                continue
            best = better_candidate(
                best,
                build_recapture_candidate(state, world, source, target, planned_arrivals_by_target),
            )
        if best is not None and best["score"] > 0.12:
            record_candidate(moves, used_source_ids, planned_arrivals_by_target, best)

    for source in ordered_sources:
        if source.id in used_source_ids or available_to_send(state, world, source, 0) <= 0:
            continue
        best = None
        for target in state.targets:
            best = better_candidate(best, build_capture_candidate(state, world, source, target, planned_arrivals_by_target))
        if best is not None and best["score"] > 0.05:
            record_candidate(moves, used_source_ids, planned_arrivals_by_target, best)

    return moves
