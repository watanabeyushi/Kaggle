import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
DEFENSE_MARGIN = 10
AIM_ITERATIONS = 3
DEFAULT_MAX_SPEED = 6.0


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
    future_turns = max(0, int(future_turns))
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


def estimate_converged_intercept(
    src_x, src_y, target, num_ships, step, initial_planets, angular_velocity_map
):
    eta_turns = estimate_eta_turns(src_x, src_y, target.x, target.y, num_ships)
    pred_x, pred_y = target.x, target.y

    for _ in range(AIM_ITERATIONS):
        pred_x, pred_y = predict_planet_position(
            target, eta_turns, step, initial_planets, angular_velocity_map
        )
        eta_turns = estimate_eta_turns(src_x, src_y, pred_x, pred_y, num_ships)

    return eta_turns, pred_x, pred_y


def estimate_target_garrison(target, eta_turns):
    if target.owner == -1:
        return target.ships
    return target.ships + target.production * eta_turns


def infer_fleet_target_and_eta(fleet, planets):
    best_planet = None
    best_eta = None
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = estimate_fleet_speed(fleet.ships)

    for planet in planets:
        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue

        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue

        hit_distance = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        eta_turns = int(math.ceil(hit_distance / max(speed, 1e-9)))
        eta_turns = max(1, eta_turns)

        if best_eta is None or eta_turns < best_eta:
            best_eta = eta_turns
            best_planet = planet

    return best_planet, best_eta


def estimate_friendly_inbound_ships(target_id, eta_turns, friendly_fleets, planets):
    inbound_ships = 0
    for fleet in friendly_fleets:
        target_planet, fleet_eta = infer_fleet_target_and_eta(fleet, planets)
        if target_planet is None or fleet_eta is None:
            continue
        if target_planet.id != target_id:
            continue
        if fleet_eta > eta_turns:
            continue
        inbound_ships += int(fleet.ships)
    return inbound_ships


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

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    friendly_fleets = [f for f in fleets if f.owner == player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float('inf')
        for t in targets:
            dist = math.sqrt((mine.x - t.x)**2 + (mine.y - t.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        initial_ships_needed = nearest.ships + 1
        eta_turns, pred_x, pred_y = estimate_converged_intercept(
            mine.x,
            mine.y,
            nearest,
            initial_ships_needed,
            step,
            initial_planets,
            angular_velocity_map,
        )
        predicted_garrison = estimate_target_garrison(nearest, eta_turns)
        raw_ships_needed = predicted_garrison + 1
        friendly_inbound_ships = estimate_friendly_inbound_ships(
            nearest.id, eta_turns, friendly_fleets, planets
        )
        ships_needed = max(0, raw_ships_needed - friendly_inbound_ships)
        available_to_send = mine.ships - DEFENSE_MARGIN

        if ships_needed <= 0:
            continue

        # Only send if we can capture the target while keeping a reserve at home
        if available_to_send >= ships_needed:
            eta_turns, pred_x, pred_y = estimate_converged_intercept(
                mine.x,
                mine.y,
                nearest,
                ships_needed,
                step,
                initial_planets,
                angular_velocity_map,
            )
            predicted_garrison = estimate_target_garrison(nearest, eta_turns)
            raw_ships_needed = predicted_garrison + 1
            friendly_inbound_ships = estimate_friendly_inbound_ships(
                nearest.id, eta_turns, friendly_fleets, planets
            )
            ships_needed = max(0, raw_ships_needed - friendly_inbound_ships)

            if ships_needed <= 0 or available_to_send < ships_needed:
                continue

            if segment_hits_sun(mine.x, mine.y, pred_x, pred_y):
                continue
            # Calculate lead angle from our planet to predicted target position
            angle = math.atan2(pred_y - mine.y, pred_x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves