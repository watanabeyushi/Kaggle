import math
from collections import defaultdict

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD_MIN = 0.0
BOARD_MAX = 100.0
LAUNCH_CLEARANCE = 0.1
DEFENSE_MARGIN = 10
AIM_ITERATIONS = 3
DEFAULT_MAX_SPEED = 6.0
MAX_FRONTLINE_PLANETS = 3
STRICT_VALIDATION_THRESHOLD = 40.0
EARLY_GAME_TURNS = 40
MAX_WAIT_TURNS = 4
WAIT_ADVANTAGE_MARGIN = 0.75
GATEWAY_SAMPLE_TURNS = (2, 3, 5, 8, 12, 18)
GATEWAY_LONG_SAMPLE_TURNS = (24, 32, 48, 64)
SPEED_THRESHOLD_LEVELS = (1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
INTERCEPT_SEARCH_RADIUS = 2
INTERCEPT_SEARCH_EXPANSION = 6
MAX_SPEED_OPTION_CANDIDATES = 5
MAX_REGULAR_TARGETS = 6
MAX_ROUGH_ATTACK_ETA = 28
EARLY_PEACE_SEND_MARGIN = 3
EARLY_PEACE_TARGETS = 10
EARLY_PEACE_SUPPLY_STEP = 20
EARLY_PAYBACK_HORIZON = 18.0
TIME_OFFSET_EXPERIMENTS = (-1, 0, 1)


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


def predict_planet_position(planet, future_turns, step, initial_planets, angular_velocity_map, time_offset=0):
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
    theta = theta0 + omega * (float(step) + future_turns + float(time_offset))
    return SUN_X + orbital_radius * math.cos(theta), SUN_Y + orbital_radius * math.sin(theta)


def board_time_offset(board_state=None):
    if not board_state:
        return -1
    return int(board_state.get("time_offset", 0) or 0)


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


def is_in_bounds(x, y):
    return BOARD_MIN <= x <= BOARD_MAX and BOARD_MIN <= y <= BOARD_MAX


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def solve_launch_solution(
    source,
    target_x,
    target_y,
    num_ships,
    step,
    initial_planets,
    angular_velocity_map,
    iterations=AIM_ITERATIONS,
):
    speed = estimate_fleet_speed(num_ships)
    angle = math.atan2(target_y - source.y, target_x - source.x)
    launch_x, launch_y = float(source.x), float(source.y)
    travel_time = math.hypot(float(target_x) - float(launch_x), float(target_y) - float(launch_y)) / max(
        speed, 1e-9
    )
    return {
        "angle": float(angle),
        "time": float(travel_time),
        "launch_x": float(launch_x),
        "launch_y": float(launch_y),
    }


def fleet_position_after_time(launch_x, launch_y, angle, num_ships, time_elapsed):
    speed = estimate_fleet_speed(num_ships)
    return (
        launch_x + math.cos(angle) * speed * time_elapsed,
        launch_y + math.sin(angle) * speed * time_elapsed,
    )


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


def max_intercept_turns(num_ships):
    speed = estimate_fleet_speed(num_ships)
    board_diagonal = math.hypot(BOARD_MAX - BOARD_MIN, BOARD_MAX - BOARD_MIN)
    return max(1, int(math.ceil(board_diagonal / max(speed, 1e-9))) + 2)


def candidate_intercept_turns(source, target, num_ships, step, initial_planets, angular_velocity_map, time_offset=0):
    max_turns = max_intercept_turns(num_ships)
    base_solution = solve_launch_solution(
        source, target.x, target.y, num_ships, step, initial_planets, angular_velocity_map, iterations=1
    )
    if base_solution is None:
        base_eta = estimate_eta_turns(source.x, source.y, target.x, target.y, num_ships)
    else:
        base_eta = max(1, int(math.ceil(base_solution["time"])))
    pred_x, pred_y = predict_planet_position(
        target, max(0, base_eta - 1), step, initial_planets, angular_velocity_map, time_offset=time_offset
    )
    solved = solve_launch_solution(
        source, pred_x, pred_y, num_ships, step, initial_planets, angular_velocity_map, iterations=1
    )
    if solved is None:
        refined_eta = estimate_eta_turns(source.x, source.y, pred_x, pred_y, num_ships)
    else:
        refined_eta = max(1, int(math.ceil(solved["time"])))
    seeds = {max(1, base_eta), max(1, refined_eta)}
    candidates = set()
    for seed in seeds:
        for delta in range(-INTERCEPT_SEARCH_RADIUS, INTERCEPT_SEARCH_RADIUS + 1):
            turn = seed + delta
            if 1 <= turn <= max_turns:
                candidates.add(turn)
        for delta in range(INTERCEPT_SEARCH_RADIUS + 1, INTERCEPT_SEARCH_EXPANSION + 1, 2):
            low_turn = seed - delta
            high_turn = seed + delta
            if 1 <= low_turn <= max_turns:
                candidates.add(low_turn)
            if 1 <= high_turn <= max_turns:
                candidates.add(high_turn)
    if not candidates:
        return [max(1, min(max_turns, base_eta))]
    return sorted(candidates)


def solution_path_endpoint(solution, num_ships):
    speed = estimate_fleet_speed(num_ships)
    travel_time = max(0.0, float(solution["time"]))
    return (
        float(solution["launch_x"]) + math.cos(float(solution["angle"])) * speed * travel_time,
        float(solution["launch_y"]) + math.sin(float(solution["angle"])) * speed * travel_time,
    )


def estimate_precise_intercept(
    source,
    target,
    num_ships,
    step,
    initial_planets,
    angular_velocity_map,
    validation_mode="strict",
    time_offset=0,
):
    best_solution = None
    best_key = None

    for turn in candidate_intercept_turns(
        source, target, num_ships, step, initial_planets, angular_velocity_map, time_offset=time_offset
    ):
        pred_x, pred_y = predict_planet_position(
            target, turn - 1, step, initial_planets, angular_velocity_map, time_offset=time_offset
        )
        launch_solution = solve_launch_solution(
            source, pred_x, pred_y, num_ships, step, initial_planets, angular_velocity_map
        )
        if launch_solution is None:
            continue
        solution = validate_intercept_solution(
            source,
            target,
            num_ships,
            launch_solution["angle"],
            step,
            initial_planets,
            angular_velocity_map,
            max_turns=turn,
            validation_mode=validation_mode,
            time_offset=time_offset,
        )
        if solution is None:
            continue
        solution["turn_hint"] = int(turn)
        solution["turn_gap"] = abs(int(solution["eta"]) - int(turn))
        candidate_key = (
            int(solution["eta"]),
            float(solution["time"]),
            abs(int(solution["eta"]) - turn),
        )
        if best_solution is None or candidate_key < best_key:
            best_solution = solution
            best_key = candidate_key

    if best_solution is not None:
        return best_solution

    fallback_solution = solve_launch_solution(
        source, target.x, target.y, num_ships, step, initial_planets, angular_velocity_map, iterations=1
    )
    if fallback_solution is None:
        fallback_turn = max(1, estimate_eta_turns(source.x, source.y, target.x, target.y, num_ships))
    else:
        fallback_turn = max(1, int(math.ceil(fallback_solution["time"])))
    pred_x, pred_y = predict_planet_position(
        target, fallback_turn - 1, step, initial_planets, angular_velocity_map, time_offset=time_offset
    )
    launch_solution = solve_launch_solution(
        source, pred_x, pred_y, num_ships, step, initial_planets, angular_velocity_map
    )
    if launch_solution is None:
        angle = math.atan2(pred_y - source.y, pred_x - source.x)
        launch_x, launch_y = float(source.x), float(source.y)
        speed = estimate_fleet_speed(num_ships)
        time_to_hit = math.hypot(pred_x - launch_x, pred_y - launch_y) / max(speed, 1e-9)
    else:
        angle = launch_solution["angle"]
        launch_x = launch_solution["launch_x"]
        launch_y = launch_solution["launch_y"]
        time_to_hit = launch_solution["time"]
    return {
        "valid": False,
        "time": float(time_to_hit),
        "eta": int(fallback_turn),
        "pred_x": float(pred_x),
        "pred_y": float(pred_y),
        "aim_x": float(pred_x),
        "aim_y": float(pred_y),
        "launch_x": float(launch_x),
        "launch_y": float(launch_y),
        "angle": float(angle),
        "uses_sweep": False,
        "turn_hint": int(fallback_turn),
        "turn_gap": 0,
    }


def validate_intercept_solution(
    source,
    target,
    num_ships,
    angle,
    step,
    initial_planets,
    angular_velocity_map,
    max_turns,
    validation_mode="strict",
    time_offset=0,
):
    launch_x, launch_y = float(source.x), float(source.y)
    prev_x, prev_y = launch_x, launch_y

    for turn in range(1, max(1, int(max_turns)) + 1):
        curr_x, curr_y = fleet_position_after_time(launch_x, launch_y, angle, num_ships, turn)
        target_before_x, target_before_y = predict_planet_position(
            target, turn - 1, step, initial_planets, angular_velocity_map, time_offset=time_offset
        )

        if segment_hits_sun(prev_x, prev_y, curr_x, curr_y):
            return None

        if not is_in_bounds(curr_x, curr_y):
            break

        if math.hypot(curr_x - target_before_x, curr_y - target_before_y) <= float(target.radius) + 1e-6:
            return {
                "valid": True,
                "time": float(turn),
                "eta": int(turn),
                "pred_x": float(target_before_x),
                "pred_y": float(target_before_y),
                "aim_x": float(curr_x),
                "aim_y": float(curr_y),
                "launch_x": float(launch_x),
                "launch_y": float(launch_y),
                "angle": float(angle),
                "uses_sweep": False,
            }

        prev_x, prev_y = curr_x, curr_y

    return None


def estimate_target_garrison(target, eta_turns):
    if target.owner == -1:
        return int(target.ships)
    return int(target.ships) + int(target.production) * int(eta_turns)


def infer_fleet_target_and_eta(fleet, planets, step, initial_planets, angular_velocity_map):
    return infer_fleet_target_and_eta_with_offset(
        fleet, planets, step, initial_planets, angular_velocity_map, time_offset=0
    )


def infer_fleet_target_and_eta_with_offset(
    fleet, planets, step, initial_planets, angular_velocity_map, time_offset=0
):
    best_planet = None
    best_eta = None
    source_planet_id = int(getattr(fleet, "from_planet_id", -1))
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = estimate_fleet_speed(fleet.ships)
    max_turns = max(1, int(math.ceil(math.hypot(BOARD_MAX, BOARD_MAX) / max(speed, 1e-9))) + 2)

    for planet in planets:
        if int(planet.id) == source_planet_id:
            continue
        prev_x, prev_y = float(fleet.x), float(fleet.y)
        for turn in range(1, max_turns + 1):
            pred_x, pred_y = predict_planet_position(
                planet, turn - 1, step, initial_planets, angular_velocity_map, time_offset=time_offset
            )
            curr_x = float(fleet.x) + dir_x * speed * turn
            curr_y = float(fleet.y) + dir_y * speed * turn
            if segment_hits_sun(prev_x, prev_y, curr_x, curr_y):
                break
            if not is_in_bounds(curr_x, curr_y):
                break
            if math.hypot(curr_x - pred_x, curr_y - pred_y) <= float(planet.radius) + 1e-6:
                if best_eta is None or turn < best_eta:
                    best_eta = turn
                    best_planet = planet
                break
            prev_x, prev_y = curr_x, curr_y

    return best_planet, best_eta


def build_arrivals_by_target(fleets, planets, step, initial_planets, angular_velocity_map, time_offset=0):
    arrivals_by_target = defaultdict(list)
    for fleet in fleets:
        target_planet, fleet_eta = infer_fleet_target_and_eta_with_offset(
            fleet, planets, step, initial_planets, angular_velocity_map, time_offset=time_offset
        )
        if target_planet is None or fleet_eta is None:
            continue
        arrivals_by_target[target_planet.id].append(
            (int(fleet_eta), int(fleet.owner), int(fleet.ships))
        )
    return arrivals_by_target


def compare_single_target_time_offsets(
    source,
    target,
    num_ships,
    step,
    initial_planets,
    angular_velocity_map,
    offsets=TIME_OFFSET_EXPERIMENTS,
    validation_mode="strict",
):
    results = []
    for time_offset in offsets:
        solution = estimate_precise_intercept(
            source,
            target,
            num_ships,
            step,
            initial_planets,
            angular_velocity_map,
            validation_mode=validation_mode,
            time_offset=time_offset,
        )
        result = {
            "offset": int(time_offset),
            "valid": bool(solution.get("valid", False)),
            "eta": int(solution["eta"]),
            "time": float(solution["time"]),
            "turn_hint": int(solution.get("turn_hint", solution["eta"])),
            "turn_gap": int(solution.get("turn_gap", 0)),
            "pred_x": float(solution["pred_x"]),
            "pred_y": float(solution["pred_y"]),
            "aim_x": float(solution["aim_x"]),
            "aim_y": float(solution["aim_y"]),
            "angle": float(solution["angle"]),
            "rank_key": (
                0 if solution.get("valid", False) else 1,
                int(solution.get("turn_gap", 0)),
                int(solution["eta"]),
                float(solution["time"]),
            ),
        }
        results.append(result)
    return sorted(results, key=lambda result: result["rank_key"])


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


def is_early_peace(board_state):
    return bool(board_state and board_state.get("early_peace", False))


def expansion_pressure(board_state):
    if not board_state:
        return 0.0
    return clamp(float(board_state.get("expansion_pressure", 0.0)), 0.0, 1.0)


def compute_expansion_pressure(step, neutral_planets, enemy_arrivals_by_target=None):
    step_factor = clamp((EARLY_GAME_TURNS - float(step)) / EARLY_GAME_TURNS, 0.0, 1.0)
    neutral_factor = clamp(float(neutral_planets) / 8.0, 0.0, 1.0)
    enemy_arrival_count = 0
    if enemy_arrivals_by_target:
        enemy_arrival_count = sum(len(arrivals) for arrivals in enemy_arrivals_by_target.values())
    threat_factor = clamp(float(enemy_arrival_count) / 6.0, 0.0, 1.0)
    return clamp(step_factor * (0.45 + 0.55 * neutral_factor) * (1.0 - 0.7 * threat_factor), 0.0, 1.0)


def effective_defense_margin(planet, board_state=None):
    pressure = expansion_pressure(board_state)
    if pressure <= 1e-6:
        return DEFENSE_MARGIN
    production_margin = max(EARLY_PEACE_SEND_MARGIN, min(5, int(getattr(planet, "production", 0)) + 1))
    if board_state is not None and len(board_state.get("my_planets", [])) <= 2:
        production_margin += 1
    relaxed_margin = min(DEFENSE_MARGIN, production_margin)
    blended_margin = DEFENSE_MARGIN + (relaxed_margin - DEFENSE_MARGIN) * pressure
    return max(1, int(round(blended_margin)))


def get_available_to_send(planet, board_state=None):
    return max(0, int(planet.ships) - effective_defense_margin(planet, board_state))


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


def compute_validation_pressure(target, ships_needed, eta_turns):
    ownership_multiplier = 1.5 if int(target.owner) != -1 else 1.0
    value = max(1.0, float(target.production))
    cost = max(1.0, float(ships_needed) * max(1.0, float(eta_turns)))
    return ownership_multiplier * cost / value


def choose_validation_mode(target, ships_needed, eta_turns, candidate_type):
    return "strict"


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


def make_planet_proxy(planet, x, y, ships=None):
    return type(
        "PlanetProxy",
        (),
        {
            "id": int(planet.id),
            "owner": int(planet.owner),
            "x": float(x),
            "y": float(y),
            "radius": float(planet.radius),
            "ships": int(planet.ships if ships is None else ships),
            "production": int(planet.production),
        },
    )()


def predicted_distance_between_planets(planet_a, planet_b, future_turns, board_state):
    ax, ay = predict_planet_position(
        planet_a,
        future_turns,
        board_state["step"],
        board_state["initial_planets"],
        board_state["angular_velocity_map"],
        time_offset=board_time_offset(board_state),
    )
    bx, by = predict_planet_position(
        planet_b,
        future_turns,
        board_state["step"],
        board_state["initial_planets"],
        board_state["angular_velocity_map"],
        time_offset=board_time_offset(board_state),
    )
    return math.hypot(ax - bx, ay - by)


def nearest_predicted_distance_to_group(target, planets, future_turns, board_state, exclude_ids=None):
    exclude_ids = set() if exclude_ids is None else set(exclude_ids)
    distances = [
        predicted_distance_between_planets(target, planet, future_turns, board_state)
        for planet in planets
        if planet.id not in exclude_ids and planet.id != target.id
    ]
    if not distances:
        return float("inf")
    return min(distances)


def local_cluster_stats(target, board_state, future_turns=0):
    distances = []
    for planet in board_state["targets"]:
        if planet.id == target.id:
            continue
        distances.append(predicted_distance_between_planets(target, planet, future_turns, board_state))
    if not distances:
        return {"nearby_count": 0, "avg_distance": BOARD_MAX, "spread_score": 0.0}
    distances.sort()
    local_distances = distances[:4]
    avg_distance = sum(local_distances) / len(local_distances)
    nearby_count = sum(1 for distance in local_distances if distance <= 22.0)
    spread_score = 1.0 / (1.0 + avg_distance / 18.0)
    return {
        "nearby_count": int(nearby_count),
        "avg_distance": float(avg_distance),
        "spread_score": float(spread_score),
    }


def tempo_multiplier(eta_turns, step):
    eta_turns = max(1.0, float(eta_turns))
    early_pressure = 1.0 + clamp((EARLY_GAME_TURNS - float(step)) / EARLY_GAME_TURNS, 0.0, 1.0) * 0.9
    horizon = max(6.0, 16.0 / early_pressure)
    return 1.0 / (1.0 + eta_turns / horizon)


def gateway_sample_turns(step):
    samples = list(GATEWAY_SAMPLE_TURNS)
    if int(step) < EARLY_GAME_TURNS:
        samples.extend(GATEWAY_LONG_SAMPLE_TURNS[:2])
    elif int(step) < EARLY_GAME_TURNS * 2:
        samples.append(GATEWAY_LONG_SAMPLE_TURNS[0])
    return tuple(samples)


def best_gateway_followup(target, future_turns, board_state, exclude_ids=None):
    exclude_ids = set() if exclude_ids is None else set(exclude_ids)
    tx, ty = predict_planet_position(
        target,
        future_turns,
        board_state["step"],
        board_state["initial_planets"],
        board_state["angular_velocity_map"],
        time_offset=board_time_offset(board_state),
    )
    best_planet = None
    best_score = 0.0
    followup_turns = max(0, int(future_turns))
    future_window = max(0.0, EARLY_GAME_TURNS - float(board_state["step"] + followup_turns))
    for planet in board_state["targets"]:
        if planet.id == target.id or planet.id in exclude_ids:
            continue
        px, py = predict_planet_position(
            planet,
            future_turns,
            board_state["step"],
            board_state["initial_planets"],
            board_state["angular_velocity_map"],
            time_offset=board_time_offset(board_state),
        )
        distance = math.hypot(tx - px, ty - py)
        production = max(1.0, float(planet.production))
        score = production * (1.0 + future_window / 60.0) / (1.0 + distance / 18.0)
        if int(planet.owner) != -1:
            score *= 1.1
        if score > best_score:
            best_score = score
            best_planet = planet
    return best_planet, best_score


def gateway_bonus(target, eta_turns, board_state):
    sample_scores = []
    for sample_turn in gateway_sample_turns(board_state["step"]):
        _, followup_score = best_gateway_followup(
            target, max(1, int(eta_turns) + int(sample_turn)), board_state
        )
        if followup_score > 0.0:
            sample_scores.append(followup_score)
    if not sample_scores:
        return 0.0
    best_scores = sorted(sample_scores, reverse=True)[:3]
    pressure = expansion_pressure(board_state)
    eta_factor = 1.0 / (1.0 + max(1.0, float(eta_turns)) / 14.0)
    weight = 1.0 + pressure * 0.9 * eta_factor
    return (sum(best_scores) / max(1.0, 8.0 * len(best_scores))) * weight


def gateway_branch_score(target, eta_turns, board_state):
    branch_scores = []
    for sample_turn in gateway_sample_turns(board_state["step"])[:4]:
        followup_turn = max(1, int(eta_turns) + int(sample_turn))
        first_planet, first_score = best_gateway_followup(target, followup_turn, board_state)
        if first_planet is None or first_score <= 0.0:
            continue
        _, second_score = best_gateway_followup(
            first_planet,
            followup_turn + max(2, int(sample_turn) // 2),
            board_state,
            exclude_ids={target.id, first_planet.id},
        )
        branch_scores.append(first_score * 0.35 + second_score * 0.18)
    if not branch_scores:
        return 0.0
    pressure = expansion_pressure(board_state)
    eta_factor = 1.0 / (1.0 + max(1.0, float(eta_turns)) / 16.0)
    return (max(branch_scores) / 8.0) * (1.0 + pressure * 0.8 * eta_factor)


def payback_bonus(target, ships_to_send, eta_turns, board_state):
    pressure = expansion_pressure(board_state)
    if pressure <= 1e-6:
        return 0.0
    production = max(1.0, float(target.production))
    payback_turns = float(ships_to_send) / production
    total_delay = float(eta_turns) + payback_turns
    recovery_factor = 1.0 / (1.0 + total_delay / EARLY_PAYBACK_HORIZON)
    neutral_factor = 1.15 if int(target.owner) == -1 else 0.85
    return pressure * neutral_factor * recovery_factor * (0.45 + 0.08 * production)


def target_value(target, eta_turns, board_state, mission_type="attack"):
    eta_turns = max(1, int(math.ceil(eta_turns)))
    pressure = expansion_pressure(board_state)
    base_production = max(1.0, float(target.production))
    future_window = max(0.0, EARLY_GAME_TURNS - float(board_state["step"] + eta_turns))
    production_value = base_production * (1.0 + future_window / 18.0)
    tempo_value = tempo_multiplier(eta_turns, board_state["step"])
    cluster = local_cluster_stats(target, board_state, future_turns=eta_turns)
    cluster_multiplier = 1.0 + 0.18 * cluster["nearby_count"] + 0.55 * cluster["spread_score"]
    enemy_distance = nearest_predicted_distance_to_group(
        target, board_state["enemy_planets"], eta_turns, board_state
    )
    gateway_weight = 0.25 + pressure * 0.18
    branch_weight = 0.12 + pressure * 0.1
    gateway_multiplier = 1.0 + gateway_weight * gateway_bonus(target, eta_turns, board_state)
    gateway_multiplier += branch_weight * gateway_branch_score(target, eta_turns, board_state)
    if pressure > 1e-6:
        production_value *= 1.0 + pressure * (0.15 + future_window / 80.0)

    if mission_type == "supply":
        frontline_pressure = 1.0 / max(8.0, enemy_distance)
        owned_multiplier = 1.0 + frontline_pressure * 6.0
        owned_multiplier *= 1.0 - pressure * 0.3
        return production_value * (0.7 + tempo_value * 0.6) * cluster_multiplier * owned_multiplier

    my_distance = nearest_predicted_distance_to_group(
        target, board_state["my_planets"], eta_turns, board_state
    )
    neutral_distance = nearest_predicted_distance_to_group(
        target, board_state["neutral_planets"], eta_turns, board_state, exclude_ids={target.id}
    )
    competition_multiplier = clamp((enemy_distance + 8.0) / (my_distance + 8.0), 0.75, 1.35)
    neutral_multiplier = 1.0
    if int(target.owner) == -1 and not math.isinf(neutral_distance):
        neutral_multiplier += 0.25 / (1.0 + neutral_distance / 18.0)
        neutral_multiplier += pressure * (0.18 / (1.0 + eta_turns / 10.0))
    owner_multiplier = 1.12 if int(target.owner) != -1 else 1.0
    return (
        production_value
        * tempo_value
        * cluster_multiplier
        * competition_multiplier
        * neutral_multiplier
        * owner_multiplier
        * gateway_multiplier
    )


def source_optionality_cost(source, ships_to_send, available_to_send, board_state):
    available_to_send = max(1.0, float(available_to_send))
    commit_ratio = clamp(float(ships_to_send) / available_to_send, 0.0, 1.5)
    frontline_distance = min_distance_to_targets(source, board_state["targets"])
    frontline_pressure = 14.0 / max(14.0, frontline_distance + 4.0)
    early_factor = 0.6 + clamp(
        (EARLY_GAME_TURNS - float(board_state["step"])) / EARLY_GAME_TURNS, 0.0, 1.0
    ) * 0.8
    scarcity_factor = 2.5 / max(2.0, float(len(board_state["my_planets"])))
    production_factor = max(1.0, float(source.production)) / 12.0
    reserve_ratio = max(0.0, (available_to_send - float(ships_to_send)) / available_to_send)
    flexibility_penalty = commit_ratio * (
        0.05 + 0.08 * frontline_pressure + 0.05 * scarcity_factor + 0.04 * production_factor
    )
    cost = flexibility_penalty * early_factor * (1.0 - 0.35 * reserve_ratio)
    cost *= 1.0 - 0.55 * expansion_pressure(board_state)
    return cost


def target_stability_cost(target, eta_turns, board_state, mission_type="attack"):
    eta_turns = max(1.0, float(eta_turns))
    early_factor = 1.0 + clamp(
        (EARLY_GAME_TURNS - float(board_state["step"])) / EARLY_GAME_TURNS, 0.0, 1.0
    ) * 0.8
    my_distance = nearest_predicted_distance_to_group(
        target, board_state["my_planets"], eta_turns, board_state
    )
    enemy_distance = nearest_predicted_distance_to_group(
        target, board_state["enemy_planets"], eta_turns, board_state
    )
    if math.isinf(enemy_distance):
        enemy_distance = BOARD_MAX
    competition_penalty = max(0.0, (my_distance - enemy_distance) / 50.0)
    frontline_penalty = (12.0 / max(12.0, enemy_distance + 2.0)) * 0.08
    eta_penalty = min(0.12, eta_turns / 80.0)
    owner_penalty = 0.05 if int(target.owner) != -1 and mission_type != "supply" else 0.0
    if mission_type == "supply":
        competition_penalty *= 0.5
        frontline_penalty *= 0.6
    cost = (competition_penalty + frontline_penalty + eta_penalty + owner_penalty) * early_factor
    pressure = expansion_pressure(board_state)
    if pressure > 1e-6:
        cost *= 1.0 - pressure * (0.6 if mission_type == "attack" else 0.8)
    return cost


def mission_score(
    source,
    target,
    ships_to_send,
    eta_turns,
    solution_time,
    available_to_send,
    board_state,
    mission_type="attack",
    opportunity_multiplier=1.0,
):
    value = target_value(target, eta_turns, board_state, mission_type=mission_type)
    value *= max(0.1, float(opportunity_multiplier))
    production_scale = max(6.0, max(1.0, float(target.production)) * 3.0)
    effective_cost = max(1.0, float(solution_time)) * (0.6 + float(ships_to_send) / production_scale)
    base_score = value / effective_cost
    source_cost = source_optionality_cost(source, ships_to_send, available_to_send, board_state)
    stability_cost = target_stability_cost(target, eta_turns, board_state, mission_type=mission_type)
    if mission_type == "supply":
        source_cost *= 0.75
        stability_cost *= 0.6
    elif mission_type == "attack":
        pressure = expansion_pressure(board_state)
        if pressure > 1e-6:
            base_score *= 1.0 + pressure * (
                min(0.4, float(target.production) / 10.0) + 0.18 / (1.0 + float(eta_turns) / 6.0)
            )
            base_score += payback_bonus(target, ships_to_send, eta_turns, board_state)
    return base_score - source_cost - stability_cost


def estimate_required_ships_for_speed(target_speed, max_speed=DEFAULT_MAX_SPEED):
    target_speed = clamp(float(target_speed), 1.0, float(max_speed))
    if target_speed <= 1.0 + 1e-9 or max_speed <= 1.0 + 1e-9:
        return 1
    ratio = (target_speed - 1.0) / (float(max_speed) - 1.0)
    ratio = clamp(ratio, 0.0, 1.0)
    exponent = ratio ** (2.0 / 3.0)
    return max(1, int(math.ceil(math.exp(math.log(1000) * exponent))))


def speed_threshold_options(min_ships, max_ships, source_production=0):
    min_ships = max(1, int(min_ships))
    max_ships = max(min_ships, int(max_ships))
    options = {min_ships, max_ships}
    growth_step = max(1, int(source_production))
    for multiplier in range(1, 4):
        options.add(min(max_ships, min_ships + growth_step * multiplier))
    for target_speed in SPEED_THRESHOLD_LEVELS:
        ships = estimate_required_ships_for_speed(target_speed)
        if min_ships <= ships <= max_ships:
            options.add(ships)
    return sorted(options)


def limited_speed_threshold_options(min_ships, max_ships, source_production=0):
    options = speed_threshold_options(min_ships, max_ships, source_production)
    if len(options) <= MAX_SPEED_OPTION_CANDIDATES:
        return options
    selected = []
    for idx in (0, 1, 2, len(options) // 2, len(options) - 1):
        if 0 <= idx < len(options):
            value = options[idx]
            if value not in selected:
                selected.append(value)
    return selected


def evaluate_regular_attack_option(
    source,
    target,
    ships_to_send,
    wait_turns,
    step,
    initial_planets,
    angular_velocity_map,
    friendly_arrivals_by_target,
    planned_arrivals_by_target,
    board_state,
):
    available_after_wait = get_available_to_send(source, board_state) + int(source.production) * int(wait_turns)
    ships_to_send = int(ships_to_send)
    if ships_to_send <= 0 or ships_to_send > available_after_wait:
        return None

    launch_source = source
    launch_step = int(step)
    if int(wait_turns) > 0:
        future_source_x, future_source_y = predict_planet_position(
            source,
            wait_turns,
            step,
            initial_planets,
            angular_velocity_map,
            time_offset=board_time_offset(board_state),
        )
        launch_source = make_planet_proxy(
            source,
            future_source_x,
            future_source_y,
            ships=int(source.ships) + int(source.production) * int(wait_turns),
        )
        launch_step = int(step) + int(wait_turns)

    solution = estimate_precise_intercept(
        launch_source,
        target,
        ships_to_send,
        launch_step,
        initial_planets,
        angular_velocity_map,
        time_offset=board_time_offset(board_state),
    )
    if not solution or not solution.get("valid", False):
        return None

    total_eta = int(wait_turns) + int(solution["eta"])
    ships_needed = compute_attack_need(
        target, total_eta, friendly_arrivals_by_target, planned_arrivals_by_target
    )
    if ships_needed <= 0 or ships_needed > ships_to_send:
        return None

    validation_mode = choose_validation_mode(target, ships_to_send, total_eta, "roi")
    path_end_x, path_end_y = solution_path_endpoint(solution, ships_to_send)
    if segment_hits_sun(solution["launch_x"], solution["launch_y"], path_end_x, path_end_y):
        return None

    effective_time = float(wait_turns) + float(solution["time"])
    score = mission_score(
        source,
        target,
        ships_to_send,
        total_eta,
        effective_time,
        available_after_wait,
        board_state,
        mission_type="attack",
    )
    if expansion_pressure(board_state) > 1e-6:
        if int(target.owner) == -1:
            score += max(
                0.0,
                expansion_pressure(board_state) * (float(target.production) * 0.16 - float(wait_turns) * 0.22),
            )
        score += max(
            0.0,
            expansion_pressure(board_state) * (0.24 - max(0, ships_to_send - ships_needed) * 0.015),
        )
    return {
        "type": "roi",
        "source_id": int(source.id),
        "target_id": int(target.id),
        "ships": int(ships_to_send),
        "ships_needed": int(ships_needed),
        "eta": int(total_eta),
        "angle": solution["angle"],
        "score": score,
        "validation_mode": validation_mode,
        "wait_turns": int(wait_turns),
        "effective_time": effective_time,
    }


def select_preferred_attack_plan(
    source,
    target,
    step,
    initial_planets,
    angular_velocity_map,
    friendly_arrivals_by_target,
    planned_arrivals_by_target,
    board_state,
):
    available_to_send = get_available_to_send(source, board_state)
    if available_to_send <= 0:
        return None

    base_min_ships = max(1, int(target.ships) + 1)
    rough_solution = solve_launch_solution(
        source, target.x, target.y, base_min_ships, step, initial_planets, angular_velocity_map, iterations=1
    )
    if rough_solution is None:
        rough_eta = estimate_eta_turns(source.x, source.y, target.x, target.y, base_min_ships)
    else:
        rough_eta = max(1, int(math.ceil(rough_solution["time"])))
    if rough_eta > MAX_ROUGH_ATTACK_ETA:
        return None
    immediate_best = None
    overall_best = None
    max_wait_turns = MAX_WAIT_TURNS if int(source.production) > 0 else 0

    for wait_turns in range(0, max_wait_turns + 1):
        available_after_wait = available_to_send + int(source.production) * wait_turns
        if available_after_wait < base_min_ships:
            continue
        for ships_to_send in limited_speed_threshold_options(
            base_min_ships, available_after_wait, source.production
        ):
            plan = evaluate_regular_attack_option(
                source,
                target,
                ships_to_send,
                wait_turns,
                step,
                initial_planets,
                angular_velocity_map,
                friendly_arrivals_by_target,
                planned_arrivals_by_target,
                board_state,
            )
            if plan is None:
                continue
            if wait_turns == 0 and (
                immediate_best is None
                or plan["effective_time"] < immediate_best["effective_time"] - 1e-6
                or (
                    abs(plan["effective_time"] - immediate_best["effective_time"]) <= 1e-6
                    and plan["ships"] < immediate_best["ships"]
                )
            ):
                immediate_best = plan
            if (
                overall_best is None
                or plan["effective_time"] < overall_best["effective_time"] - 1e-6
                or (
                    abs(plan["effective_time"] - overall_best["effective_time"]) <= 1e-6
                    and plan["ships"] < overall_best["ships"]
                )
            ):
                overall_best = plan

    if overall_best is None:
        return None
    if immediate_best is None:
        return overall_best
    if expansion_pressure(board_state) >= 0.55:
        return immediate_best
    if overall_best["wait_turns"] > 0 and (
        overall_best["effective_time"] + WAIT_ADVANTAGE_MARGIN < immediate_best["effective_time"]
    ):
        return overall_best
    return immediate_best


def build_regular_attack_candidate(
    source,
    target,
    step,
    initial_planets,
    angular_velocity_map,
    friendly_arrivals_by_target,
    planned_arrivals_by_target,
    board_state,
):
    return select_preferred_attack_plan(
        source,
        target,
        step,
        initial_planets,
        angular_velocity_map,
        friendly_arrivals_by_target,
        planned_arrivals_by_target,
        board_state,
    )


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
                    "opportunity_multiplier": 1.0
                    + (float(target.production) / max(4.0, float(arrival_garrison))),
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
    board_state,
):
    available_to_send = get_available_to_send(source, board_state)
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
    validation_mode = choose_validation_mode(target, ships_needed, desired_eta, "intercept")

    solution = estimate_precise_intercept(
        source,
        target,
        ships_needed,
        step,
        initial_planets,
        angular_velocity_map,
        validation_mode=validation_mode,
        time_offset=board_time_offset(board_state),
    )
    if not solution.get("valid", False):
        return None
    if solution["eta"] != desired_eta:
        return None

    path_end_x, path_end_y = solution_path_endpoint(solution, ships_needed)
    if segment_hits_sun(solution["launch_x"], solution["launch_y"], path_end_x, path_end_y):
        return None

    intercept_score = mission_score(
        source,
        target,
        ships_needed,
        desired_eta,
        solution["time"],
        available_to_send,
        board_state,
        mission_type="intercept",
        opportunity_multiplier=window.get("opportunity_multiplier", 1.0),
    )
    return {
        "type": "intercept",
        "source_id": int(source.id),
        "target_id": int(target.id),
        "ships": int(ships_needed),
        "eta": int(desired_eta),
        "angle": solution["angle"],
        "score": intercept_score,
        "validation_mode": validation_mode,
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
    board_state,
):
    available_to_send = get_available_to_send(source, board_state)
    if available_to_send <= 0 or source.id == frontline_target.id:
        return None
    expansion_bias = expansion_pressure(board_state)
    if expansion_bias > 0.45:
        if int(step) < EARLY_PEACE_SUPPLY_STEP:
            return None
        if board_state.get("neutral_planets"):
            return None

    probe_ships = max(1, min(available_to_send, effective_defense_margin(source, board_state)))
    solution = estimate_precise_intercept(
        source,
        frontline_target,
        probe_ships,
        step,
        initial_planets,
        angular_velocity_map,
        time_offset=board_time_offset(board_state),
    )
    if not solution.get("valid", False):
        return None
    eta_turns = solution["eta"]
    incoming_support = estimate_friendly_inbound_ships(
        frontline_target.id, eta_turns, friendly_arrivals_by_target, planned_arrivals_by_target
    )
    predicted_frontline_ships = (
        int(frontline_target.ships) + int(frontline_target.production) * eta_turns + incoming_support
    )
    desired_frontline_ships = max(DEFENSE_MARGIN * 3, int(frontline_target.production) * 4)
    if expansion_bias > 1e-6:
        desired_frontline_ships = max(
            effective_defense_margin(frontline_target, board_state) * 2,
            int(frontline_target.production) * 2,
        )
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
        time_offset=board_time_offset(board_state),
    )
    if not solution.get("valid", False):
        return None
    eta_turns = solution["eta"]
    incoming_support = estimate_friendly_inbound_ships(
        frontline_target.id, eta_turns, friendly_arrivals_by_target, planned_arrivals_by_target
    )
    path_end_x, path_end_y = solution_path_endpoint(solution, ships_to_send)
    if segment_hits_sun(solution["launch_x"], solution["launch_y"], path_end_x, path_end_y):
        return None

    frontline_pressure = 1.0 / max(1.0, min_distance_to_targets(frontline_target, targets))
    score = mission_score(
        source,
        frontline_target,
        ships_to_send,
        eta_turns,
        solution["time"],
        available_to_send,
        board_state,
        mission_type="supply",
        opportunity_multiplier=1.0 + frontline_pressure * 6.0,
    )
    if expansion_bias > 1e-6:
        score *= 1.0 - 0.45 * expansion_bias
    return {
        "type": "supply",
        "source_id": int(source.id),
        "target_id": int(frontline_target.id),
        "ships": int(ships_to_send),
        "eta": int(eta_turns),
        "angle": solution["angle"],
        "score": score,
        "validation_mode": "strict",
    }


def rank_candidates(candidates):
    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def nearest_planet_sniper(obs):
    moves = []
    player = _obs_get(obs, "player", 0)
    step = int(_obs_get(obs, "step", 0) or 0)
    time_offset = int(_obs_get(obs, "time_offset", 0) or 0)
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

    board_state = {
        "player": int(player),
        "step": int(step),
        "early_peace": int(step) < EARLY_GAME_TURNS,
        "expansion_pressure": 0.0,
        "planets": planets,
        "my_planets": my_planets,
        "targets": targets,
        "enemy_planets": [planet for planet in targets if int(planet.owner) not in (-1, int(player))],
        "neutral_planets": [planet for planet in targets if int(planet.owner) == -1],
        "initial_planets": initial_planets,
        "angular_velocity_map": angular_velocity_map,
        "time_offset": int(time_offset),
    }

    arrivals_by_target = build_arrivals_by_target(
        fleets, planets, step, initial_planets, angular_velocity_map, time_offset=time_offset
    )
    friendly_arrivals_by_target = defaultdict(list)
    enemy_arrivals_by_target = defaultdict(list)
    for target_id, arrivals in arrivals_by_target.items():
        for eta_turns, owner, ships in arrivals:
            if owner == player:
                friendly_arrivals_by_target[target_id].append((eta_turns, owner, ships))
            else:
                enemy_arrivals_by_target[target_id].append((eta_turns, owner, ships))
    board_state["expansion_pressure"] = compute_expansion_pressure(
        step, len(board_state["neutral_planets"]), enemy_arrivals_by_target
    )

    intercept_windows_by_target = {
        target.id: build_intercept_windows(
            target, enemy_arrivals_by_target.get(target.id, []), player
        )
        for target in targets
    }
    frontline_planets, rear_planets = classify_frontline_planets(my_planets, targets)
    planned_arrivals_by_target = defaultdict(list)
    used_source_ids = set()
    reserved_source_ids = set()

    for source in sorted(my_planets, key=lambda planet: get_available_to_send(planet, board_state), reverse=True):
        if (
            source.id in used_source_ids
            or source.id in reserved_source_ids
            or get_available_to_send(source, board_state) <= 0
        ):
            continue

        candidates = []
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
                    board_state,
                )
                if candidate is None:
                    continue
                candidates.append(candidate)

        ranked_candidates = rank_candidates(candidates)
        if ranked_candidates:
            best_candidate = ranked_candidates[0]
            moves.append([best_candidate["source_id"], best_candidate["angle"], best_candidate["ships"]])
            planned_arrivals_by_target[best_candidate["target_id"]].append(
                (best_candidate["eta"], best_candidate["ships"])
            )
            used_source_ids.add(best_candidate["source_id"])

    for source in sorted(my_planets, key=lambda planet: get_available_to_send(planet, board_state), reverse=True):
        if (
            source.id in used_source_ids
            or source.id in reserved_source_ids
            or get_available_to_send(source, board_state) <= 0
        ):
            continue

        candidates = []
        target_limit = max(
            MAX_REGULAR_TARGETS,
            int(round(MAX_REGULAR_TARGETS + (EARLY_PEACE_TARGETS - MAX_REGULAR_TARGETS) * expansion_pressure(board_state))),
        )
        candidate_targets = sorted(
            targets,
            key=lambda target: (
                math.hypot(source.x - target.x, source.y - target.y)
                / max(1.0, 1.0 + float(target.production) * (0.8 if int(target.owner) == -1 else 0.4)),
                0 if int(target.owner) == -1 else 1,
            ),
        )[:target_limit]
        for target in candidate_targets:
            candidate = build_regular_attack_candidate(
                source,
                target,
                step,
                initial_planets,
                angular_velocity_map,
                friendly_arrivals_by_target,
                planned_arrivals_by_target,
                board_state,
            )
            if candidate is None:
                continue
            candidates.append(candidate)

        ranked_candidates = rank_candidates(candidates)
        if ranked_candidates:
            best_candidate = ranked_candidates[0]
            if best_candidate.get("wait_turns", 0) > 0 and expansion_pressure(board_state) < 0.55:
                reserved_source_ids.add(best_candidate["source_id"])
                continue
            moves.append([best_candidate["source_id"], best_candidate["angle"], best_candidate["ships"]])
            planned_arrivals_by_target[best_candidate["target_id"]].append(
                (best_candidate["eta"], best_candidate["ships"])
            )
            used_source_ids.add(best_candidate["source_id"])

    for source in sorted(rear_planets, key=lambda planet: get_available_to_send(planet, board_state), reverse=True):
        if (
            source.id in used_source_ids
            or source.id in reserved_source_ids
            or get_available_to_send(source, board_state) <= 0
        ):
            continue

        candidates = []
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
                board_state,
            )
            if candidate is None:
                continue
            candidates.append(candidate)

        ranked_candidates = rank_candidates(candidates)
        if ranked_candidates:
            best_candidate = ranked_candidates[0]
            moves.append([best_candidate["source_id"], best_candidate["angle"], best_candidate["ships"]])
            planned_arrivals_by_target[best_candidate["target_id"]].append(
                (best_candidate["eta"], best_candidate["ships"])
            )
            used_source_ids.add(best_candidate["source_id"])

    return moves