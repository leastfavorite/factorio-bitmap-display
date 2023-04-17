from PIL import Image
import math
from sys import argv
from itertools import product
import random
import json
import zlib
import base64

CLIGHT1 = '\33[32m'
CLIGHT2 = '\33[92m'
CDARK1 = '\33[38;5;241m'
CDARK2 = '\33[38;5;246m'
CBOLD = '\33[1m'
CEND = '\33[0m'
CITALIC = '\33[3m'

# signal names. we avoid I since we use it for input.
COMBINATOR_ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ0123456789"

def encode_blueprint(blueprint):
    as_json = json.dumps(blueprint,separators=(',',':'))
    as_zcompressed = zlib.compress(as_json.encode("utf-8"), 9)
    as_b64 = base64.b64encode(as_zcompressed)
    return f"0{as_b64.decode()}"

def lamp_grid_to_blueprint(lamp_grid):
    blueprint = {
        "blueprint": {
            "icons": [
                {
                    "signal": {
                        "type": "item",
                        "name": "small-lamp"
                    },
                    "index": 1
                }
            ],
            "entities": [],
            "item": "blueprint",
            "version": 281479276920832
        }
    }

    h = len(lamp_grid)
    w = len(lamp_grid[0])

    for y, row in enumerate(lamp_grid):
        for x, lamp in enumerate(row):

            entity_number = y*w + x + 1

            connections = []
            if y > 0:
                connections.append(entity_number - w)
            if y < h-1:
                connections.append(entity_number + w)
            if y == h-1 and x != 0:
                connections.append(entity_number - 1)
            if y == h-1 and x != w+1:
                connections.append(entity_number + 1)
            connections = [{"entity_id":c} for c in connections]
            connections = {"1": {
                "red": connections,
                "green": connections
            }}

            blueprint["blueprint"]["entities"].append({
                "entity_number": entity_number,
                "name": "small-lamp",
                "position": {
                    "x": x + 0.5,
                    "y": y + 0.5
                },
                "control_behavior": lamp.as_control_behavior(),
                "connections": connections
            })

    return encode_blueprint(blueprint)


class Combinator(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_to_signal = {v:k for k,v in self.items()}

    def add(self, value):
        for signal in COMBINATOR_ALPHABET:
            if signal not in self:
                self[signal] = value
                return
        self[random.random()] = value

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.value_to_signal[v] = k

    def __getitem__(self, k):
        if type(k) is int:
            return self.value_to_signal[k]
        else:
            return super().__getitem__(k)

    def __contains__(self, k):
        return k in self.value_to_signal or super().__contains__(k)

    def __str__(self):
        ENTRIES_PER_ROW = 5
        ret = []
        ret.append(f"{CLIGHT2}{CBOLD}{'Combinator Values'.center(9 * ENTRIES_PER_ROW - 3)}{CEND}")
        ret.append(CDARK2 + "-------+-" * (ENTRIES_PER_ROW - 1) + f"------{CEND}")
        for i in range(0, len(self), ENTRIES_PER_ROW):
            ret.append(f" {CDARK2}|{CEND} ".join(
                f"{CLIGHT1}{CBOLD}{k}{CEND}{CDARK2}:{CEND}{CLIGHT2}{CITALIC}{v:4}{CEND}" \
                for k,v in list(self.items())[i:i+ENTRIES_PER_ROW]))
        return "\n".join(ret)

    def as_blueprint(self):
        COMBINATOR_SIZE = 20
        def _gen_filter(index, signal, value):
            return {
                "signal": {
                    "type": "virtual",
                    "name": f"signal-{signal}"
                },
                "count": value,
                "index": index
            }

        blueprint = {
            "blueprint": {
                "icons": [
                    {
                        "signal": {
                            "type": "item",
                            "name": "constant-combinator"
                        },
                        "index": 1
                    }
                ],
                "entities": [],
                "item": "blueprint",
                "version": 281479276920832
            }
        }

        for i, block_start in enumerate(range(0, len(self), COMBINATOR_SIZE), start=1):
            block = list(self.items())[block_start:block_start+COMBINATOR_SIZE]
            entity = {
                "entity_number": i,
                "name": "constant-combinator",
                "position": {
                    "x": i + 0.5,
                    "y": 0.5
                },
                "direction": 2,
                "control_behavior": {
                    "filters": [_gen_filter(i, k, v) for i, (k, v) in enumerate(block, start=1)]
                }
            }

            connections = []
            if block_start != 0:
                connections.append(i - 1)
            if block_start + 20 < len(self):
                connections.append(i + 1)
            if len(connections) > 0:
                entity["connections"] = {
                    "1": {
                        "red": [{"entity_id":c for c in connections}]
                    }
                }

            blueprint["blueprint"]["entities"].append(entity)

        return encode_blueprint(blueprint)

class Lamp:
    def __init__(self, a, op, b):
        self.a = a
        self.op = op
        self.b = b

    def __repr__(self):
        symbol = {"eq":"=", "neq":"≠", "gt":">"}[self.op]
        return f"Parsed{{{self.a}{symbol}{self.b}}}"

    def eval_at(self, combinator, i):
        evaluations = {k:((v >> i) & 1) for k,v in combinator.items()}
        evaluations["I"] = i

        a = evaluations[self.a] if type(self.a) is str else self.a
        b = evaluations[self.b] if type(self.b) is str else self.b
        if self.op == "eq": return a == b
        if self.op == "neq": return a != b
        if self.op == "gt": return a > b

    def eval_to_str(self, combinator, i):
        result = self.eval_at(combinator, i)
        color = CLIGHT1 if result else CDARK1
        color_int = CLIGHT2 if result else CDARK2

        wrap = lambda x: f"{color}{CBOLD}{x}{CEND}" if \
            type(x) is str else f"{color_int}{CITALIC}{x}{CEND}"

        symbol = {"eq":"=", "neq":"≠", "gt":">"}[self.op]
        symbol = f"{color}{symbol}{CEND}"

        return f"{wrap(self.a)}{symbol}{wrap(self.b)}"

    def as_control_behavior(self):
        ret = {
            "circuit_condition": {
                "first_signal": {
                    "type": "virtual",
                    "name": f"signal-{self.a}"
                },
                "second_signal": {
                    "type": "virtual",
                    "name": f"signal-{self.b}"
                },
                "constant": self.b,
                "comparator": {"eq":"=", "neq":"≠", "gt":">"}[self.op]
            },
            "use_colors": True
        }
        if type(self.b) is int:
            del ret["circuit_condition"]["second_signal"]
        else:
            del ret["circuit_condition"]["constant"]
        return ret

# converts a target set of numbers into a set of combinator values and
# lamp rules.
def values_to_lamps(values: set[int]) -> tuple[Combinator, dict[int, Lamp]]:
    _v = values.copy()

    trivial_lamps = {}
    if 0 in values:
        values.remove(0)
        trivial_lamps[0] = Lamp("I", "neq", "I")
    for i in range(int(math.log2(max(values))) + 1):
        exp = 2 ** i
        if exp in values:
            values.remove(exp)
        trivial_lamps[exp] = Lamp("I", "eq", i)

    def _find_shortcuts(values, cabinet_size, heuristic_shuffle):
        lamps = {}
        comb = Combinator()

        while len(values) > 0:
            # calculates a heuristic for number of shortcuts and a list of
            # all possible candidates that are compatible with this one
            def _list_shortcuts(value):
                targets = values - {value}
                intermediates = set(comb.values()) | targets

                # i dont know why this heuristic works better than just
                # checking all possible reachable targets. but it does,
                # and like.. it does *considerably*. maybe worth revisiting
                # now that some randomness is added to the heuristic
                heuristic = set()
                for i in intermediates:
                    if i ^ value in targets:
                        heuristic.add(min(i, i ^ value))
                    if value & ~i in targets:
                        heuristic.add(value & ~i)

                candidates = set()
                targets_reached = set()

                for i in intermediates:
                    reachable_targets = {value ^ i, value & ~i, i & ~value} & targets
                    if len(reachable_targets) > 0:
                        candidates.add(i)
                    targets_reached |= reachable_targets

                return candidates, len(heuristic) + random.gauss(0, heuristic_shuffle)

            shortcut_stats = {v:_list_shortcuts(v) for v in values}
            key = lambda v: shortcut_stats[v][1]

            candidate = max(values, key=key)
            cabinet = [candidate] + list(sorted(
                shortcut_stats[candidate][0] & values, key=key, reverse=True))

            # add new elements to combinator
            for a in cabinet[:cabinet_size + 1]:
                if a not in values:
                    continue

                values.remove(a)
                comb.add(a)
                lamps[a] = Lamp(comb[a], "eq", 1)

                for b in comb.values():
                    if a == b: continue
                    if a ^ b in values:
                        values.remove(a ^ b)
                        lamps[a ^ b] = Lamp(comb[a], "neq", comb[b])
                    if a & ~b in values:
                        values.remove(a & ~b)
                        lamps[a & ~b] = Lamp(comb[a], "gt", comb[b])
                    if b & ~a in values:
                        values.remove(b & ~a)
                        lamps[b & ~a] = Lamp(comb[b], "gt", comb[a])


        lamps.update(trivial_lamps)
        return lamps, comb


    lamps = None
    comb = None
    gaussian = [0, 0.01, 0.05, 0.075, 0.1, 0.3, 0.5, 1]
    for cabinet_size, gauss_stdev in product(range(30), gaussian):
        for _ in range(10):
            new_lamps, new_comb = _find_shortcuts(values.copy(), cabinet_size, gauss_stdev)
            if comb is None or len(new_comb) < len(comb):
                lamps, comb = new_lamps, new_comb
        print(".", end="", flush=True)
    print("")

    return lamps, comb

def image_to_lamps(grid: list[list[int]]):
    values = set(sum(grid, []))

    lamps, comb = values_to_lamps(values)
    print(f"Found result with {len(comb)} signals.\n")
    print(str(comb), end="\n\n")

    lamp_grid = [[lamps[x] for x in row] for row in grid]

    for i in range(int(math.log2(max(values))) + 1):
        print("\n".join(" ".join(
            x.eval_to_str(comb, i) for x in y) for y in lamp_grid), end="\n\n")

    return lamp_grid, comb

if __name__ == "__main__":
    if len(argv) == 1:
        print("Pass image as argument.")
        exit(1)

    im = Image.open(argv[1])
    n = set()

    process_pixel = lambda r, g, b: r >> 4 | g | b << 4
    grid = [[process_pixel(*im.getpixel((x,y))) for x in range(im.width)] for y in range(im.height)]
    lamps, comb = image_to_lamps(grid)

    if input(f"{CLIGHT1}{CBOLD}Do you want blueprints? {CLIGHT2}{CITALIC}(y/n): {CDARK2}").lower() in ["y", "yes"]:

        print(f"\n{CLIGHT1}{CBOLD}Combinator Blueprint: {CDARK2}{CITALIC}{comb.as_blueprint()}{CEND}", end="\n")
        print(f"\n{CLIGHT1}{CBOLD}Lamp Blueprint: {CDARK2}{CITALIC}{lamp_grid_to_blueprint(lamps)}{CEND}")
