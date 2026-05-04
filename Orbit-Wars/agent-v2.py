"""
Orbit Wars - Accuracy-Focused Nearest Planet Agent

This agent keeps the original nearest-target strategy, but upgrades the
shot geometry with moving-target prediction, sun avoidance, and optional
mid-flight interception of enemy fleets that are heading toward our planets.
"""

import math

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 5.0
INNER_ORBIT_R = 30.0
SUN_MARGIN = 1.8
INTERCEPT_ANGLE_TOLERANCE = 0.28
SAFE_AIM_DELTAS = [0.08, -0.08, 0.16, -0.16, 0.28, -0.28, 0.45, -0.45]


def _read(obs, key, default=None):
    if hasattr(obs, key):
        value = getattr(obs, key)
        return default if value is None else value
    if isinstance(obs, dict):
        value = obs.get(key, default)
        return default if value is None else value
    return default


def fleet_speed(ships, cap=6.0):
    return min(1.0 + ships // 20, cap)


def hits_sun(sx, sy, angle, margin=SUN_MARGIN):
    dx = math.cos(angle)
    dy = math.sin(angle)
    t = (SUN_X - sx) * dx + (SUN_Y - sy) * dy
    if t < 0:
        return False
    hit_x = sx + t * dx
    hit_y = sy + t * dy
    return math.hypot(hit_x - SUN_X, hit_y - SUN_Y) < SUN_RADIUS + margin


class PlanetState:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production")

    def __init__(self, record):
        (
            self.id,
            self.owner,
            self.x,
            self.y,
            self.radius,
            self.ships,
            self.production,
        ) = record

    def dist(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def dist_xy(self, x, y):
        return math.hypot(self.x - x, self.y - y)

    def angle_to(self, other):
        return math.atan2(other.y - self.y, other.x - self.x)

    def angle_xy(self, x, y):
        return math.atan2(y - self.y, x - self.x)


class FleetState:
    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")

    def __init__(self, record):
        self.id, self.owner, self.x, self.y, self.angle, self.from_planet_id, self.ships = record


class GameState:
    def __init__(self, obs):
        self.my_id = _read(obs, "player", 0)
        self.ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
        self.planets = [PlanetState(p) for p in (_read(obs, "planets", []) or [])]
        self.fleets = [FleetState(f) for f in (_read(obs, "fleets", []) or [])]
        self._planets_by_id = {planet.id: planet for planet in self.planets}
        self.my_planets = [planet for planet in self.planets if planet.owner == self.my_id]
        self.targets = [planet for planet in self.planets if planet.owner != self.my_id]

    def get(self, planet_id):
        return self._planets_by_id.get(planet_id)

    def is_inner(self, planet):
        return planet.dist_xy(SUN_X, SUN_Y) < INNER_ORBIT_R

    def estimate_target(self, fleet):
        best_planet_id = None
        best_dist = float("inf")
        for planet in self.planets:
            angle = math.atan2(planet.y - fleet.y, planet.x - fleet.x)
            delta = abs((angle - fleet.angle + math.pi) % (2 * math.pi) - math.pi)
            if delta < INTERCEPT_ANGLE_TOLERANCE:
                dist = math.hypot(planet.x - fleet.x, planet.y - fleet.y)
                if dist < best_dist:
                    best_dist = dist
                    best_planet_id = planet.id
        return best_planet_id


class Predictor:
    def __init__(self, state):
        self.state = state

    def future_pos(self, planet, turns):
        if not self.state.is_inner(planet):
            return planet.x, planet.y
        orbit_radius = planet.dist_xy(SUN_X, SUN_Y)
        start_angle = math.atan2(planet.y - SUN_Y, planet.x - SUN_X)
        future_angle = start_angle + self.state.ang_vel * turns
        return (
            SUN_X + orbit_radius * math.cos(future_angle),
            SUN_Y + orbit_radius * math.sin(future_angle),
        )

    def discrete_intercept_turns(self, src, dst, ships, max_turns=50):
        speed = fleet_speed(ships)
        for turns in range(1, max_turns + 1):
            tx, ty = self.future_pos(dst, turns - 1)
            dist = math.hypot(tx - src.x, ty - src.y)
            if dist <= speed * turns:
                return turns, tx, ty
        tx, ty = self.future_pos(dst, max_turns - 1)
        return max_turns, tx, ty

    def intercept(self, src, dst, ships, iters=6):
        _, tx, ty = self.discrete_intercept_turns(src, dst, ships)
        return tx, ty

    def aim(self, src, dst, ships):
        if not self.state.is_inner(dst):
            return src.angle_to(dst)
        tx, ty = self.intercept(src, dst, ships)
        return src.angle_xy(tx, ty)

    def eta(self, src, dst, ships):
        turns, _, _ = self.discrete_intercept_turns(src, dst, ships)
        return turns

    def safe_aim(self, src, dst, ships):
        angle = self.aim(src, dst, ships)
        if not hits_sun(src.x, src.y, angle):
            return angle
        for delta in SAFE_AIM_DELTAS:
            candidate = angle + delta
            if not hits_sun(src.x, src.y, candidate):
                return candidate
        return angle


class FleetInterceptor:
    def __init__(self, state):
        self.state = state

    def fleet_at(self, fleet, turns):
        speed = fleet_speed(fleet.ships)
        return (
            fleet.x + math.cos(fleet.angle) * speed * turns,
            fleet.y + math.sin(fleet.angle) * speed * turns,
        )

    def find_window(self, fleet, src, our_ships):
        our_speed = fleet_speed(our_ships)
        for turns in range(1, 50):
            fx, fy = self.fleet_at(fleet, turns - 1)
            dist = math.hypot(fx - src.x, fy - src.y)
            if dist <= our_speed * turns:
                return fx, fy, turns
        return None

    def find_all(self):
        candidates = []
        for fleet in self.state.fleets:
            if fleet.owner in (-1, self.state.my_id):
                continue
            target_id = self.state.estimate_target(fleet)
            if target_id is None:
                continue
            target_planet = self.state.get(target_id)
            if target_planet is None or target_planet.owner != self.state.my_id:
                continue
            for src in self.state.my_planets:
                our_ships = max(5, src.ships // 3)
                if our_ships > fleet.ships:
                    window = self.find_window(fleet, src, our_ships)
                    if window is None:
                        continue
                    ix, iy, eta = window
                    angle = src.angle_xy(ix, iy)
                    if hits_sun(src.x, src.y, angle):
                        continue
                    candidates.append(
                        {
                            "src": src.id,
                            "our_ships": our_ships,
                            "enemy_ships": fleet.ships,
                            "eta": eta,
                            "angle": angle,
                        }
                    )
                    break
        return sorted(
            candidates,
            key=lambda item: (-item["enemy_ships"], item["eta"], item["src"]),
        )


def find_nearest_target(src, targets):
    nearest = None
    min_dist = float("inf")
    for target in targets:
        dist = src.dist(target)
        if dist < min_dist:
            min_dist = dist
            nearest = target
    return nearest


def agent(obs):
    state = GameState(obs)
    if not state.my_planets:
        return []

    predictor = Predictor(state)
    interceptor = FleetInterceptor(state)
    moves = []
    used_sources = set()

    for option in interceptor.find_all():
        src = state.get(option["src"])
        if src is None or src.id in used_sources:
            continue
        if src.ships >= option["our_ships"]:
            moves.append([src.id, option["angle"], option["our_ships"]])
            used_sources.add(src.id)

    if not state.targets:
        return moves

    for mine in state.my_planets:
        if mine.id in used_sources:
            continue
        nearest = find_nearest_target(mine, state.targets)
        if nearest is None:
            continue
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = predictor.safe_aim(mine, nearest, ships_needed)
            moves.append([mine.id, angle, ships_needed])

    return moves
