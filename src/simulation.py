"""Discrete-time simulation of a two-approach signalized intersection.
Vehicles arrive to each road following a Poisson process; the traffic
light alternates green between road 1 and road 2 each cycle, with the
green time of road 1 decided by the fuzzy controller and road 2 getting
the remainder of the fixed cycle time."""
import numpy as np

from . import config as cfg


class TrafficSimulation:
    def __init__(self, controller, arrival_rate_1, arrival_rate_2,
                 n_cycles=cfg.N_CYCLES, seed=None):
        self.controller = controller
        self.lam1 = arrival_rate_1
        self.lam2 = arrival_rate_2
        self.n_cycles = n_cycles
        self.rng = np.random.default_rng(seed)

    def _phase(self, q_green, q_red, duration, lam_green, lam_red):
        """Simulate `duration` seconds where `q_green` road has a green
        light and `q_red` road has a red light. Returns updated queues
        and accumulated metrics for this phase."""
        if duration <= 0:
            return q_green, q_red, 0.0, 0, 0.0

        arr_green = self.rng.poisson(lam_green, size=duration)
        arr_red = self.rng.poisson(lam_red, size=duration)

        queue_seconds = 0.0
        stops = 0
        served = 0.0

        for t in range(duration):
            q_green += arr_green[t]
            q_red += arr_red[t]

            depart = min(q_green, cfg.SERVICE_RATE)
            q_green -= depart
            served += depart

            if q_red > 0:
                stops += 1

            queue_seconds += q_green + q_red

        return q_green, q_red, queue_seconds, stops, served

    def run(self):
        q1, q2 = 0.0, 0.0
        queue_seconds_total = 0.0
        stops_total = 0
        served_total = 0.0
        seconds_total = 0

        q1_history, q2_history, g1_history = [], [], []

        for _ in range(self.n_cycles):
            g1_raw = self.controller.compute_green_time(q1, q2)
            g1 = int(round(min(max(g1_raw, cfg.GREEN_MIN), cfg.GREEN_MAX)))
            g2 = int(
                round(min(max(cfg.CYCLE_TIME - g1, cfg.GREEN_MIN), cfg.GREEN_MAX)))

            q1_history.append(q1)
            q2_history.append(q2)
            g1_history.append(g1)

            # phase 1: road 1 green, road 2 red
            q1, q2, qs1, s1, sv1 = self._phase(
                q1, q2, g1, self.lam1, self.lam2)
            queue_seconds_total += qs1
            stops_total += s1
            served_total += sv1
            seconds_total += g1

            # phase 2: road 2 green, road 1 red
            q2, q1, qs2, s2, sv2 = self._phase(
                q2, q1, g2, self.lam2, self.lam1)
            queue_seconds_total += qs2
            stops_total += s2
            served_total += sv2
            seconds_total += g2

        served_total = max(served_total, 1e-6)
        seconds_total = max(seconds_total, 1e-6)

        avg_wait = queue_seconds_total / served_total   # avg wait time per vehicle
        avg_queue = queue_seconds_total / seconds_total  # avg total queue length

        return {
            "W": avg_wait,
            "Q": avg_queue,
            "S": float(stops_total),
            "q1_history": q1_history,
            "q2_history": q2_history,
            "g1_history": g1_history,
        }
