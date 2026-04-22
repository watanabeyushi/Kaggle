import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
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
    initial_planets = _obs_get(obs, "initial_planets", [])
    angular_velocity = _obs_get(obs, "angular_velocity", [])
    angular_velocity_map = _build_angular_velocity_map(angular_velocity)
    planets = [Planet(*p) for p in raw_planets]

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

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

        # How many ships do we need? Target's garrison + 1
        ships_needed = max(nearest.ships + 1, 20)

        # Only send if we have enough
        if mine.ships >= ships_needed:
            eta_turns = estimate_eta_turns(mine.x, mine.y, nearest.x, nearest.y, ships_needed)
            pred_x, pred_y = predict_planet_position(
                nearest, eta_turns, step, initial_planets, angular_velocity_map
            )
            if segment_hits_sun(mine.x, mine.y, pred_x, pred_y):
                continue
            # Calculate lead angle from our planet to predicted target position
            angle = math.atan2(pred_y - mine.y, pred_x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves