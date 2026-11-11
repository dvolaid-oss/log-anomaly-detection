"""Synthetic SSH `auth.log` with injected attacks (Stage 9).

Why synthetic: real login logs contain real usernames and IP addresses (never publishable),
and nobody knows exactly which lines in them are attacks. Here we generate both the normal
traffic and the attacks ourselves, so the data is safe to share and the ground truth is exact.

Normal traffic
- ~30 fictional users with habits: weekday work hours, 1-2 usual machines (10.x addresses),
  mostly public-key logins, the occasional mistyped password followed by a success.
- Background internet scanning: every public SSH server gets a steady trickle of bots. Like
  real ones, they try `root` most of all, then service names (`admin`, `oracle`, ...) and
  common first names, some of which are real accounts here. A SOC treats this as noise, so it
  is labelled NORMAL.
  (Version 1 of this generator never tried `root` or real names, which made every outside
  "Failed password for <real account>" an attack: a giveaway real logs don't have. Fixed in v2;
  v1 results are archived in reports/auth/results_v1_flawed.md.)

Attacks (labelled per line with their type)
- brute_force       one IP, hundreds of failed passwords for one account within minutes
- password_spray    one IP, one attempt each against many accounts, spread over ~40 minutes
- compromise        dozens of failures for a real user, then "Accepted password": a break-in
- distributed       many IPs, each trying only 1-3 passwords: invisible per IP
- account_takeover  a real user logs in at night from a never-seen IP, with no failures

Safety: external addresses come only from the ranges reserved for documentation
(RFC 5737: 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24); all names are made up.
Everything is driven by one seeded random generator, so the output is identical every run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

USERS = [
    "alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
    "mallory", "niaj", "olivia", "peggy", "rupert", "sybil", "trent", "victor", "walter",
    "yasmin", "zoe", "liam", "noah", "emma", "mia", "lucas", "ava", "omar", "lena", "yuki",
]  # fmt: skip
BOT_NAMES = [
    "admin", "test", "user", "oracle", "postgres", "ubuntu", "guest", "git", "ftpuser",
    "support", "pi", "deploy", "hadoop", "mysql", "www", "backup", "jenkins", "minecraft",
]  # fmt: skip
COMMON_NAMES = [  # what bots try from leaked name lists; some are real accounts here
    "john", "david", "michael", "alice", "bob", "emma", "noah", "liam", "anna", "maria",
]  # fmt: skip
DOC_NETS = ["192.0.2", "198.51.100", "203.0.113"]  # RFC 5737 documentation ranges
ATTACKS = ["brute_force", "password_spray", "compromise", "distributed", "account_takeover"]


@dataclass
class Line:
    time: datetime
    text: str  # everything after "host sshd[pid]: "
    attack: str = ""  # "" = normal
    pid: int = 0


@dataclass
class Generator:
    rng: np.random.Generator
    host: str = "web01"
    lines: list[Line] = field(default_factory=list)

    # -------------------------------------------------------------- helpers
    def port(self) -> int:
        return int(self.rng.integers(32768, 61000))

    def pid(self) -> int:
        return int(self.rng.integers(1000, 65000))

    def external_ip(self) -> str:
        return f"{self.rng.choice(DOC_NETS)}.{int(self.rng.integers(1, 255))}"

    def add(self, t: datetime, text: str, attack: str = "", pid: int | None = None) -> None:
        self.lines.append(Line(t, text, attack, pid if pid is not None else self.pid()))

    def gap(self, low: float, high: float) -> timedelta:
        return timedelta(seconds=float(self.rng.uniform(low, high)))

    # -------------------------------------------------------------- building blocks
    def login(self, t: datetime, user: str, ip: str, method: str, attack: str = "") -> datetime:
        """A successful login, a session, and the logout. Returns the end time."""
        port, pid, uid = self.port(), self.pid(), 1000 + USERS.index(user) if user in USERS else 0
        self.add(t, f"Accepted {method} for {user} from {ip} port {port} ssh2", attack, pid)
        t += self.gap(0, 1)
        self.add(
            t,
            f"pam_unix(sshd:session): session opened for user {user}(uid={uid}) by (uid=0)",
            attack,
            pid,
        )  # noqa: E501
        t += timedelta(minutes=float(self.rng.uniform(2, 90)))
        self.add(
            t, f"Received disconnect from {ip} port {port}:11: disconnected by user", attack, pid
        )
        self.add(t, f"Disconnected from user {user} {ip} port {port}", attack, pid)
        self.add(t, f"pam_unix(sshd:session): session closed for user {user}", attack, pid)
        return t

    def failed(self, t: datetime, user: str, ip: str, attack: str = "", valid: bool = True) -> None:
        port = self.port()
        if valid:
            self.add(t, f"Failed password for {user} from {ip} port {port} ssh2", attack)
        else:
            pid = self.pid()
            self.add(t, f"Invalid user {user} from {ip} port {port}", attack, pid)
            self.add(
                t,
                f"Failed password for invalid user {user} from {ip} port {port} ssh2",
                attack,
                pid,
            )  # noqa: E501
            self.add(
                t + self.gap(0, 2),
                f"Connection closed by invalid user {user} {ip} port {port} [preauth]",
                attack,
                pid,
            )  # noqa: E501

    # -------------------------------------------------------------- normal traffic
    def user_day(self, day: datetime, user: str, profile: dict) -> None:
        weekend = day.weekday() >= 5
        for _ in range(int(self.rng.poisson(0.4 if weekend else profile["logins_per_day"]))):
            hour = self.rng.uniform(profile["start"], profile["end"])
            t = day + timedelta(hours=float(hour))
            ip = str(self.rng.choice(profile["ips"]))
            method = "publickey" if self.rng.random() < profile["key_share"] else "password"
            if method == "password" and self.rng.random() < 0.08:  # mistyped password
                self.failed(t, user, ip)
                t += self.gap(3, 20)
            self.login(t, user, ip, method)

    def background_scanning(self, day: datetime, per_day: float) -> None:
        for _ in range(int(self.rng.poisson(per_day))):
            t = day + timedelta(seconds=float(self.rng.uniform(0, 86400)))
            ip = self.external_ip()
            if self.rng.random() < 0.3:  # port scanner: connects, says nothing
                self.add(t, f"Connection closed by {ip} port {self.port()} [preauth]")
                continue
            for _ in range(int(self.rng.integers(1, 5))):
                user = self.bot_username()
                self.failed(t, user, ip, valid=user == "root" or user in USERS)
                t += self.gap(1, 15)

    def bot_username(self) -> str:
        """Bots try root most, then service accounts, then common first names."""
        kind = self.rng.random()
        if kind < 0.45:
            return "root"
        if kind < 0.85:
            return str(self.rng.choice(BOT_NAMES))
        return str(self.rng.choice(COMMON_NAMES))

    # -------------------------------------------------------------- attacks
    def brute_force(self, t: datetime) -> None:
        ip, user = self.external_ip(), str(self.rng.choice(["root", *USERS[:5]]))
        for _ in range(int(self.rng.integers(100, 400))):
            self.failed(t, user, ip, "brute_force")
            t += self.gap(0.5, 3)
        self.add(
            t,
            f"error: maximum authentication attempts exceeded for {user} from {ip} port {self.port()} ssh2 [preauth]",  # noqa: E501
            "brute_force",
        )  # noqa: E501
        self.add(
            t,
            f"Disconnecting authenticating user {user} {ip} port {self.port()}: Too many authentication failures [preauth]",  # noqa: E501
            "brute_force",
        )  # noqa: E501

    def password_spray(self, t: datetime) -> None:
        ip = self.external_ip()
        targets = list(self.rng.choice(USERS, size=int(self.rng.integers(15, 30)), replace=False))
        targets += list(self.rng.choice(BOT_NAMES, size=5, replace=False))
        for user in self.rng.permutation(targets):
            self.failed(t, str(user), ip, "password_spray", valid=user in USERS)
            t += self.gap(40, 120)  # slow on purpose, to stay under rate limits

    def compromise(self, t: datetime) -> None:
        ip, user = self.external_ip(), str(self.rng.choice(USERS))
        for _ in range(int(self.rng.integers(30, 80))):
            self.failed(t, user, ip, "compromise")
            t += self.gap(1, 5)
        self.login(t, user, ip, "password", "compromise")

    def distributed(self, t: datetime) -> None:
        user = str(self.rng.choice(["root", *USERS[:3]]))
        for _ in range(int(self.rng.integers(20, 50))):
            ip, start = self.external_ip(), t + self.gap(0, 7200)
            for _ in range(int(self.rng.integers(1, 4))):
                self.failed(start, user, ip, "distributed")
                start += self.gap(5, 60)

    def account_takeover(self, t: datetime) -> None:
        user = str(self.rng.choice(USERS))
        night = t.replace(hour=int(self.rng.integers(1, 5)), minute=int(self.rng.integers(0, 60)))
        self.login(night, user, self.external_ip(), "password", "account_takeover")

    # -------------------------------------------------------------- output
    def write(self, log_path: Path, labels_path: Path) -> None:
        self.lines.sort(key=lambda line: line.time)  # stable: equal times keep their order
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            log_path.open("w", encoding="utf-8", newline="\n") as log,
            labels_path.open("w", encoding="utf-8", newline="") as lab,
        ):
            labels = csv.writer(lab)
            labels.writerow(["line_id", "attack"])
            for i, line in enumerate(self.lines):
                stamp = f"{line.time:%b} {line.time.day:>2} {line.time:%H:%M:%S}"
                log.write(f"{stamp} {self.host} sshd[{line.pid}]: {line.text}\n")
                labels.writerow([i, line.attack])


def generate(config: dict[str, Any], report=print) -> tuple[Path, Path]:
    """Generate `auth.log` + `auth_labels.csv` into raw_dir, from `data.generate` settings."""
    settings = config["data"]["generate"]
    raw_dir = Path(config["paths"]["raw_dir"])
    log_path = raw_dir / config["data"]["log_file"]
    labels_path = raw_dir / config["data"]["label_file"]

    rng = np.random.default_rng(settings.get("seed", config.get("seed", 42)))
    gen = Generator(rng)
    start = datetime.fromisoformat(settings["start"])
    days = int(settings["days"])

    profiles = {}
    for user in USERS:
        n_ips = int(rng.integers(1, 3))
        profiles[user] = {
            "ips": [
                f"10.0.{int(rng.integers(0, 16))}.{int(rng.integers(2, 250))}" for _ in range(n_ips)
            ],
            "start": float(rng.uniform(7.5, 10)),
            "end": float(rng.uniform(16.5, 19)),
            "logins_per_day": float(rng.uniform(1.5, 4)),
            "key_share": float(rng.uniform(0.5, 0.95)),
        }

    for d in range(days):
        day = start + timedelta(days=d)
        for user in USERS:
            gen.user_day(day, user, profiles[user])
        gen.background_scanning(day, settings.get("scanners_per_day", 40))

    # Attacks: each type happens on random days across the whole period, at random times,
    # so every split (train / validation / test) sees every type.
    attack_rate = settings.get("attacks_per_type", 20)
    for attack in ATTACKS:
        for d in sorted(rng.choice(days, size=attack_rate, replace=False)):
            when = start + timedelta(days=int(d), seconds=float(rng.uniform(0, 86400)))
            getattr(gen, attack)(when)

    gen.write(log_path, labels_path)
    n_attack = sum(1 for line in gen.lines if line.attack)
    report(f"Generated {len(gen.lines):,} lines ({n_attack:,} attack lines), written to {log_path}")
    return log_path, labels_path
