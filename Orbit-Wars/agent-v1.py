import math
from collections import defaultdict

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
LAUNCH_CLEARANCE = 0.1
DEFENSE_MARGIN = 10
AIM_ITERATIONS = 3
DEFAULT_MAX_SPEED = 6.0
MAX_FRONTLINE_PLANETS = 3


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _build_angular_velocity_map(angular_velocity):
    if isinstance(angular_velocity, dict):
        return {int(k): float(v) for k, v in angular_velocity.items()}
    if isinstance(angular_velocity, (list, tuple)):
        return {i: float(v) for i, v in enumerate(angular_velocity)}
    return {}


def _is_orbiting_initial(planet_row):
    if len(planet_row) < 7:
        return False
    x, y, radius = float(planet_row[2]), float(planet_row[3]), float(planet_row[4])
    orbital_radius = math.hypot(x - SUN_X, y - SUN_Y)
    return orbital_radius + radius < 50.0


def predict_planet_position(planet, future_turns, step, initial_planets, angular_velocity_map):
    future_turns = max(0.0, float(future_turns))
    base = None
    if isinstance(initial_planets, (list, tuple)) and 0 <= planet.id < len(initial_planets):
        row = initial_planets[planet.id]
        if isinstance(row, (list, tuple)) and len(row) >= 7:
            base = row

    if base is None or not _is_orbiting_initial(base):
        return planet.x, planet.y

    x0, y0 = float(base[2]), float(base[3])
    omega = float(angular_velocity_map.get(planet.id, 0.0))
    if abs(omega) < 1e-12:
        return x0, y0

    orbital_radius = math.hypot(x0 - SUN_X, y0 - SUN_Y)
    theta0 = math.atan2(y0 - SUN_Y, x0 - SUN_X)
    theta = theta0 + omega * (step + future_turns)
    return SUN_X + orbital_radius * math.cos(theta), SUN_Y + orbital_radius * math.sin(theta)


def launch_point(planet, angle, clearance=LAUNCH_CLEARANCE):
    launch_radius = float(planet.radius) + float(clearance)
    return (
        float(planet.x) + math.cos(angle) * launch_radius,
        float(planet.y) + math.sin(angle) * launch_radius,
    )


def closest_point_on_segment(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    return x1 + t * dx, y1 + t * dy


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    closest_x, closest_y = closest_point_on_segment(px, py, x1, y1, x2, y2)
    return math.hypot(px - closest_x, py - closest_y)


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


def estimate_fleet_speed(num_ships, max_speed=DEFAULT_MAX_SPEED):
    ships = max(1, int(num_ships))
    if ships == 1:
        return 1.0
    speed_ratio = math.log(ships) / math.log(1000)
    speed_ratio = max(0.0, min(1.0, speed_ratio))
    return 1.0 + (max_speed - 1.0) * (speed_ratio ** 1.5)


def estimate_eta_turns(src_x, src_y, dst_x, dst_y, num_ships):
    distance = math.hypot(dst_x - src_x, dst_y - src_y)
    speed = estimate_fleet_speed(num_ships)
    return max(1, int(math.ceil(distance / max(speed, 1e-9))))


def estimate_precise_intercept(
    source, target, num_ships, step, initial_planets, angular_velocity_map
):
    speed = estimate_fleet_speed(num_ships)
    time_to_hit = max(0.0, math.hypot(target.x - source.x, target.y - source.y) / max(speed, 1e-9))
    angle = math.atan2(target.y - source.y, target.x - source.x)
    launch_x, launch_y = launch_point(source, angle)
    pred_x, pred_y = target.x, target.y

    for _ in range(AIM_ITERATIONS):
        pred_x, pred_y = predict_planet_position(
            target, time_to_hit, step, initial_planets, angular_velocity_map
        )
        angle = math.atan2(pred_y - source.y, pred_x - source.x)
        launch_x, launch_y = launch_point(source, angle)
        time_to_hit = math.hypot(pred_x - launch_x, pred_y - launch_y) / max(speed, 1e-9)

    eta_turns = max(1, int(math.ceil(time_to_hit)))
    sweep_start_x, sweep_start_y = predict_planet_position(
        target, max(0.0, eta_turns - 1), step, initial_planets, angular_velocity_map
    )
    sweep_end_x, sweep_end_y = predict_planet_position(
        target, eta_turns, step, initial_planets, angular_velocity_map
    )
    sweep_aim_x, sweep_aim_y = closest_point_on_segment(
        pred_x, pred_y, sweep_start_x, sweep_start_y, sweep_end_x, sweep_end_y
    )
    uses_sweep = math.hypot(sweep_end_x - sweep_start_x, sweep_end_y - sweep_start_y) > 1e-6

    if uses_sweep:
        angle = math.atan2(sweep_aim_y - source.y, sweep_aim_x - source.x)
        launch_x, launch_y = launch_point(source, angle)
        time_to_hit = math.hypot(sweep_aim_x - launch_x, sweep_aim_y - launch_y) / max(speed, 1e-9)
        eta_turns = max(1, int(math.ceil(time_to_hit)))
        pred_x, pred_y = predict_planet_position(
            target, time_to_hit, step, initial_planets, angular_velocity_map
        )
        sweep_start_x, sweep_start_y = predict_planet_position(
            target, max(0.0, eta_turns - 1), step, initial_planets, angular_velocity_map
        )
        sweep_end_x, sweep_end_y = predict_planet_position(
            target, eta_turns, step, initial_planets, angular_velocity_map
        )
        sweep_aim_x, sweep_aim_y = closest_point_on_segment(
            pred_x, pred_y, sweep_start_x, sweep_start_y, sweep_end_x, sweep_end_y
        )
        angle = math.atan2(sweep_aim_y - source.y, sweep_aim_x - source.x)
        launch_x, launch_y = launch_point(source, angle)

    aim_x, aim_y = (sweep_aim_x, sweep_aim_y) if uses_sweep else (pred_x, pred_y)
    return {
        "time": float(time_to_hit),
        "eta": int(max(1, math.ceil(time_to_hit))),
        "pred_x": float(pred_x),
        "pred_y": float(pred_y),
        "aim_x": float(aim_x),
        "aim_y": float(aim_y),
        "launch_x": float(launch_x),
        "launch_y": float(launch_y),
        "angle": float(angle),
        "uses_sweep": uses_sweep,
        "sweep_start_x": float(sweep_start_x),
        "sweep_start_y": float(sweep_start_y),
        "sweep_end_x": float(sweep_end_x),
        "sweep_end_y": float(sweep_end_y),
    }


def estimate_target_garrison(target, eta_turns):
    if target.owner == -1:
        return int(target.ships)
    return int(target.ships) + int(target.production) * int(eta_turns)


def infer_fleet_target_and_eta(fleet, planets, step, initial_planets, angular_velocity_map):
    best_planet = None
    best_eta = None
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = estimate_fleet_speed(fleet.ships)

    for planet in planets:
        eta_time = math.hypot(planet.x - fleet.x, planet.y - fleet.y) / max(speed, 1e-9)
        hit_distance = None

        for _ in range(AIM_ITERATIONS):
            pred_x, pred_y = predict_planet_position(
                planet, eta_time, step, initial_planets, angular_velocity_map
            )
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


def build_arrivals_by_target(fleets, planets, step, initial_planets, angular_velocity_map):
    arrivals_by_target = defaultdict(list)
    for fleet in fleets:
        target_planet, fleet_eta = infer_fleet_target_and_eta(
            fleet, planets, step, initial_planets, angular_velocity_map
        )
        if target_planet is None or fleet_eta is None:
            continue
        arrivals_by_target[target_planet.id].append(
            (int(fleet_eta), int(fleet.owner), int(fleet.ships))
        )
    return arrivals_by_target


def estimate_friendly_inbound_ships(
    target_id, eta_turns, friendly_arrivals_by_target, planned_arrivals_by_target=None
):
    inbound_ships = 0

    for fleet_eta, _, ships in friendly_arrivals_by_target.get(target_id, []):
        if fleet_eta <= eta_turns:
            inbound_ships += int(ships)

    if planned_arrivals_by_target is not None:
        for fleet_eta, ships in planned_arrivals_by_target.get(target_id, []):
            if fleet_eta <= eta_turns:
                inbound_ships += int(ships)

    return inbound_ships


def get_available_to_send(planet):
    return max(0, int(planet.ships) - DEFENSE_MARGIN)


def compute_attack_need(
    target, eta_turns, friendly_arrivals_by_target, planned_arrivals_by_target, garrison_override=None
):
    predicted_garrison = (
        int(garrison_override)
        if garrison_override is not None
        else estimate_target_garrison(target, eta_turns)
    )
    friendly_inbound = estimate_friendly_inbound_ships(
        target.id, eta_turns, friendly_arrivals_by_target, planned_arrivals_by_target
    )
    return max(0, predicted_garrison + 1 - friendly_inbound)


def segment_hits_sun(src_x, src_y, dst_x, dst_y, sun_x=SUN_X, sun_y=SUN_Y, sun_radius=SUN_RADIUS):
    dx = dst_x - src_x
    dy = dst_y - src_y
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return math.hypot(src_x - sun_x, src_y - sun_y) <= sun_radius

    t = ((sun_x - src_x) * dx + (sun_y - src_y) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    closest_x = src_x + t * dx
    closest_y = src_y + t * dy
    return math.hypot(closest_x - sun_x, closest_y - sun_y) <= sun_radius


def build_regular_attack_candidate(
    source,
    target,
    step,
    initial_planets,
    angular_velocity_map,
    friendly_arrivals_by_target,
    planned_arrivals_by_target,
):
    available_to_send = get_available_to_send(source)
    if available_to_send <= 0:
        return None

    initial_ships_needed = max(1, int(target.ships) + 1)
    solution = estimate_precise_intercept(
        source,
        target,
        initial_ships_needed,
        step,
        initial_planets,
        angular_velocity_map,
    )
    eta_turns = solution["eta"]
    ships_needed = compute_attack_need(
        target, eta_turns, friendly_arrivals_by_target, planned_arrivals_by_target
    )
    if ships_needed <= 0 or ships_needed > available_to_send:
        return None

    solution = estimate_precise_intercept(
        source,
        target,
        ships_needed,
        step,
        initial_planets,
        angular_velocity_map,
    )
    eta_turns = solution["eta"]
    ships_needed = compute_attack_need(
        target, eta_turns, friendly_arrivals_by_target, planned_arrivals_by_target
    )
    if ships_needed <= 0 or ships_needed > available_to_send:
        return None

    solution = estimate_precise_intercept(
        source,
        target,
        ships_needed,
        step,
        initial_planets,
        angular_velocity_map,
    )
    eta_turns = solution["eta"]
    if segment_hits_sun(
        solution["launch_x"], solution["launch_y"], solution["aim_x"], solution["aim_y"]
    ):
        return None

    roi_score = target.production / max(1.0, ships_needed * solution["time"])
    return {
        "type": "roi",
        "source_id": int(source.id),
        "target_id": int(target.id),
        "ships": int(ships_needed),
        "eta": int(eta_turns),
        "angle": solution["angle"],
        "score": roi_score,
    }


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


def build_intercept_windows(target, enemy_arrivals, player):
    if not enemy_arrivals:
        return []

    arrivals_by_turn = defaultdict(list)
    for eta_turns, owner, ships in enemy_arrivals:
        if owner == player:
            continue
        arrivals_by_turn[int(eta_turns)].append((int(owner), int(ships)))

    if not arrivals_by_turn:
        return []

    windows = []
    owner = int(target.owner)
    garrison = int(target.ships)
    max_turn = max(arrivals_by_turn)

    for turn in range(1, max_turn + 1):
        if owner != -1:
            garrison += int(target.production)

        turn_arrivals = arrivals_by_turn.get(turn, [])
        if not turn_arrivals:
            continue

        previous_owner = owner
        owner, garrison = resolve_arrival_event(owner, garrison, turn_arrivals)

        if previous_owner != owner and previous_owner != player and owner != player:
            desired_eta = turn + 1
            if desired_eta in arrivals_by_turn:
                continue
            arrival_garrison = garrison + (int(target.production) if owner != -1 else 0)
            windows.append(
                {
                    "arrival_eta": int(desired_eta),
                    "garrison": int(arrival_garrison),
                    "score": target.production / max(1.0, arrival_garrison),
                }
            )

    return windows


def build_intercept_candidate(
    source,
    target,
    window,
    step,
    initial_planets,
    angular_velocity_map,
    friendly_arrivals_by_target,
    planned_arrivals_by_target,
):
    available_to_send = get_available_to_send(source)
    if available_to_send <= 0:
        return None

    desired_eta = int(window["arrival_eta"])
    ships_needed = compute_attack_need(
        target,
        desired_eta,
        friendly_arrivals_by_target,
        planned_arrivals_by_target,
        garrison_override=window["garrison"],
    )
    if ships_needed <= 0 or ships_needed > available_to_send:
        return None

    solution = estimate_precise_intercept(
        source,
        target,
        ships_needed,
        step,
        initial_planets,
        angular_velocity_map,
    )
    if solution["eta"] != desired_eta:
        return None

    if segment_hits_sun(
        solution["launch_x"], solution["launch_y"], solution["aim_x"], solution["aim_y"]
    ):
        return None

    intercept_score = window["score"] / max(1.0, ships_needed)
    return {
        "type": "intercept",
        "source_id": int(source.id),
        "target_id": int(target.id),
        "ships": int(ships_needed),
        "eta": int(desired_eta),
        "angle": solution["angle"],
        "score": intercept_score,
    }


def min_distance_to_targets(planet, targets):
    if not targets:
        return float("inf")
    return min(math.hypot(planet.x - target.x, planet.y - target.y) for target in targets)


def classify_frontline_planets(my_planets, targets):
    if not my_planets:
        return [], []

    sorted_planets = sorted(my_planets, key=lambda planet: min_distance_to_targets(planet, targets))
    frontline_count = max(1, min(MAX_FRONTLINE_PLANETS, len(sorted_planets)))
    return sorted_planets[:frontline_count], sorted_planets[frontline_count:]


def build_supply_candidate(
    source,
    frontline_target,
    step,
    targets,
    initial_planets,
    angular_velocity_map,
    friendly_arrivals_by_target,
    planned_arrivals_by_target,
):
    available_to_send = get_available_to_send(source)
    if available_to_send <= 0 or source.id == frontline_target.id:
        return None

    probe_ships = max(1, min(available_to_send, DEFENSE_MARGIN))
    solution = estimate_precise_intercept(
        source,
        frontline_target,
        probe_ships,
        step,
        initial_planets,
        angular_velocity_map,
    )
    eta_turns = solution["eta"]
    incoming_support = estimate_friendly_inbound_ships(
        frontline_target.id, eta_turns, friendly_arrivals_by_target, planned_arrivals_by_target
    )
    predicted_frontline_ships = (
        int(frontline_target.ships) + int(frontline_target.production) * eta_turns + incoming_support
    )
    desired_frontline_ships = max(DEFENSE_MARGIN * 3, int(frontline_target.production) * 4)
    deficit = desired_frontline_ships - predicted_frontline_ships
    if deficit <= 0:
        return None

    ships_to_send = min(available_to_send, int(deficit))
    if ships_to_send <= 0:
        return None

    solution = estimate_precise_intercept(
        source,
        frontline_target,
        ships_to_send,
        step,
        initial_planets,
        angular_velocity_map,
    )
    eta_turns = solution["eta"]
    incoming_support = estimate_friendly_inbound_ships(
        frontline_target.id, eta_turns, friendly_arrivals_by_target, planned_arrivals_by_target
    )
    predicted_frontline_ships = (
        int(frontline_target.ships) + int(frontline_target.production) * eta_turns + incoming_support
    )
    deficit = max(0, desired_frontline_ships - predicted_frontline_ships)
    ships_to_send = min(available_to_send, int(deficit))
    if ships_to_send <= 0:
        return None

    solution = estimate_precise_intercept(
        source,
        frontline_target,
        ships_to_send,
        step,
        initial_planets,
        angular_velocity_map,
    )
    eta_turns = solution["eta"]
    if segment_hits_sun(
        solution["launch_x"], solution["launch_y"], solution["aim_x"], solution["aim_y"]
    ):
        return None

    pressure = 1.0 / max(1.0, min_distance_to_targets(frontline_target, targets))
    score = (ships_to_send * pressure) / max(1.0, solution["time"])
    return {
        "type": "supply",
        "source_id": int(source.id),
        "target_id": int(frontline_target.id),
        "ships": int(ships_to_send),
        "eta": int(eta_turns),
        "angle": solution["angle"],
        "score": score,
    }


def nearest_planet_sniper(obs):
    moves = []
    player = _obs_get(obs, "player", 0)
    step = int(_obs_get(obs, "step", 0) or 0)
    raw_planets = _obs_get(obs, "planets", [])
    raw_fleets = _obs_get(obs, "fleets", [])
    initial_planets = _obs_get(obs, "initial_planets", [])
    angular_velocity = _obs_get(obs, "angular_velocity", [])
    angular_velocity_map = _build_angular_velocity_map(angular_velocity)
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]

    my_planets = [planet for planet in planets if planet.owner == player]
    targets = [planet for planet in planets if planet.owner != player]
    if not my_planets or not targets:
        return moves

    arrivals_by_target = build_arrivals_by_target(
        fleets, planets, step, initial_planets, angular_velocity_map
    )
    friendly_arrivals_by_target = defaultdict(list)
    enemy_arrivals_by_target = defaultdict(list)
    for target_id, arrivals in arrivals_by_target.items():
        for eta_turns, owner, ships in arrivals:
            if owner == player:
                friendly_arrivals_by_target[target_id].append((eta_turns, owner, ships))
            else:
                enemy_arrivals_by_target[target_id].append((eta_turns, owner, ships))

    intercept_windows_by_target = {
        target.id: build_intercept_windows(
            target, enemy_arrivals_by_target.get(target.id, []), player
        )
        for target in targets
    }
    frontline_planets, rear_planets = classify_frontline_planets(my_planets, targets)
    planned_arrivals_by_target = defaultdict(list)
    used_source_ids = set()

    for source in sorted(my_planets, key=lambda planet: get_available_to_send(planet), reverse=True):
        if source.id in used_source_ids or get_available_to_send(source) <= 0:
            continue

        best_candidate = None
        for target in targets:
            for window in intercept_windows_by_target.get(target.id, []):
                candidate = build_intercept_candidate(
                    source,
                    target,
                    window,
                    step,
                    initial_planets,
                    angular_velocity_map,
                    friendly_arrivals_by_target,
                    planned_arrivals_by_target,
                )
                if candidate is None:
                    continue
                if best_candidate is None or candidate["score"] > best_candidate["score"]:
                    best_candidate = candidate

        if best_candidate is not None:
            moves.append([best_candidate["source_id"], best_candidate["angle"], best_candidate["ships"]])
            planned_arrivals_by_target[best_candidate["target_id"]].append(
                (best_candidate["eta"], best_candidate["ships"])
            )
            used_source_ids.add(best_candidate["source_id"])

    for source in sorted(my_planets, key=lambda planet: get_available_to_send(planet), reverse=True):
        if source.id in used_source_ids or get_available_to_send(source) <= 0:
            continue

        best_candidate = None
        for target in targets:
            candidate = build_regular_attack_candidate(
                source,
                target,
                step,
                initial_planets,
                angular_velocity_map,
                friendly_arrivals_by_target,
                planned_arrivals_by_target,
            )
            if candidate is None:
                continue
            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

        if best_candidate is not None:
            moves.append([best_candidate["source_id"], best_candidate["angle"], best_candidate["ships"]])
            planned_arrivals_by_target[best_candidate["target_id"]].append(
                (best_candidate["eta"], best_candidate["ships"])
            )
            used_source_ids.add(best_candidate["source_id"])

    for source in sorted(rear_planets, key=lambda planet: get_available_to_send(planet), reverse=True):
        if source.id in used_source_ids or get_available_to_send(source) <= 0:
            continue

        best_candidate = None
        for frontline_target in frontline_planets:
            candidate = build_supply_candidate(
                source,
                frontline_target,
                step,
                targets,
                initial_planets,
                angular_velocity_map,
                friendly_arrivals_by_target,
                planned_arrivals_by_target,
            )
            if candidate is None:
                continue
            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

        if best_candidate is not None:
            moves.append([best_candidate["source_id"], best_candidate["angle"], best_candidate["ships"]])
            planned_arrivals_by_target[best_candidate["target_id"]].append(
                (best_candidate["eta"], best_candidate["ships"])
            )
            used_source_ids.add(best_candidate["source_id"])

    return moves