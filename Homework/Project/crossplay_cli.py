"""
Crossplay CLI Game (approximation of NYT Crossplay)
===================================================

This module implements a command‑line word game inspired by the New York
Times game **Crossplay**.  The goal is to provide a simple environment for
two players (or bots) to play a Scrabble‑like game from a terminal.  It
implements many of the key features described for Crossplay, including
the letter distribution, scoring system, rack size, tile bag mechanics
and the special 40‑point bonus when a player uses all seven tiles in a
single move【872144310017704†L260-L303】.  It also enforces the "equal turn
rule" where both players get the same number of turns once the bag is
empty【872144310017704†L271-L273】.

**Important note about the board layout**
---------------------------------------

Crossplay’s official board uses a modern layout with premium squares
placed differently from Scrabble to encourage a more open game【872144310017704†L379-L389】.
Those exact coordinates are proprietary and unpublished.  This CLI
version instead uses the familiar Scrabble premium square pattern to
remain symmetrical and fair; the multipliers are the same types (double
letter, triple letter, double word, triple word) but their positions
mirror the classic board.  You can adjust the positions in the
`PREMIUM_SQUARES` dictionary to experiment with other layouts.

The game does **not** enforce a dictionary by default.  Any string of
letters that fits on the board and can be formed from the player’s rack
is accepted.  For serious play or training an AI, you can easily plug
in your own dictionary by implementing the `is_valid_word` function.

Usage
-----

Run the module directly with Python to start an interactive game:

```
python3 crossplay_cli.py
```

You will be prompted for player names and then alternately asked to
enter moves.  Moves use the syntax:

```
WORD ROW COL DIRECTION
```

Where `WORD` is the letters you wish to place (without spaces),
`ROW` and `COL` are zero‑based coordinates (0–14) for the starting
square, and `DIRECTION` is either `H` for horizontal or `V` for
vertical.  For example, the command `HELLO 7 7 H` places the word
“HELLO” starting at row 7, column 7 going horizontally to the right.

Players may also type `PASS` to skip their turn or `EXCHANGE LETTERS` to
swap tiles back into the bag.  When exchanging, separate the letters
with no spaces (e.g. `EXCHANGE ABC`).

Implementation overview
-----------------------

* **Letter distribution** and **scoring** come directly from NYT Crossplay
  documentation【872144310017704†L281-L297】.  There are 100 tiles including
  blanks (3), and letters like *K* and *V* are worth 6 points each while
  common vowels are worth only 1 point.
* A player’s **rack** always holds up to seven tiles; after playing
  tiles, new ones are drawn from the bag until it is empty【872144310017704†L260-L303】.
* When a player uses all seven tiles in a single move, a **40‑point
  bonus** is added after applying board multipliers【872144310017704†L299-L303】.
* The **equal turn rule** ensures both players get one final turn after
  the bag is emptied【872144310017704†L271-L273】.  The game ends either
  when both players pass consecutively or when both have had one move
  after the bag is empty and cannot play further.
* At the end of the game, any unplayed tiles remaining on a player’s
  rack are subtracted from that player’s score and added to their
  opponent’s score, similar to Scrabble’s rules.

This code is meant to be clear and modifiable.  Feel free to adjust
the board layout, scoring or rules to suit your training experiments.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Letter distribution and points for NYT Crossplay
# -----------------------------------------------------------------------------
# According to published sources【872144310017704†L281-L297】 there are 100 tiles with the
# following counts and point values.  Blank tiles have zero value but can
# represent any letter.  Tile counts sum to 100.

# Each entry maps a letter to a tuple (count, points)
LETTER_DISTRIBUTION: Dict[str, Tuple[int, int]] = {
    "*": (3, 0),  # blanks
    "A": (9, 1), "E": (12, 1), "I": (8, 1), "N": (5, 1), "O": (8, 1),
    "R": (6, 1), "S": (5, 1), "T": (6, 1),
    "D": (4, 2), "L": (4, 2), "U": (3, 2),
    "C": (2, 3), "H": (3, 3), "M": (2, 3), "P": (2, 3),
    "B": (2, 4), "F": (2, 4), "G": (3, 4), "Y": (2, 4),
    "W": (2, 5),
    "K": (1, 6), "V": (2, 6),
    "X": (1, 8),
    "J": (1, 10), "Q": (1, 10), "Z": (1, 10),
}


def generate_tile_bag() -> List[str]:
    """Create a shuffled list of tiles based on Crossplay distribution."""
    tiles: List[str] = []
    for letter, (count, _) in LETTER_DISTRIBUTION.items():
        tiles.extend([letter] * count)
    random.shuffle(tiles)
    return tiles


def tile_points(letter: str) -> int:
    """Return the point value for a given letter (blank '*' = 0)."""
    return LETTER_DISTRIBUTION[letter][1]


# -----------------------------------------------------------------------------
# Board configuration
# -----------------------------------------------------------------------------
# The board is 15x15.  Each cell may have a premium multiplier which applies
# only on the turn a tile is placed.  After a tile is on a premium square
# subsequent plays do not benefit from the multiplier.

class Premium:
    NONE = " "
    DOUBLE_LETTER = "DL"
    TRIPLE_LETTER = "TL"
    DOUBLE_WORD = "DW"
    TRIPLE_WORD = "TW"


# Define the premium squares.  We use the classic Scrabble layout to avoid
# proprietary patterns.  Coordinates are zero‑based (row, col).  The board is
# symmetrical so we list only a subset and mirror the rest.  Feel free to
# modify this dictionary to experiment with different layouts.
_PREMIUM_TEMPLATE: Dict[Tuple[int, int], str] = {
    # Triple word squares
    (0, 0): Premium.TRIPLE_WORD, (0, 7): Premium.TRIPLE_WORD, (0, 14): Premium.TRIPLE_WORD,
    (7, 0): Premium.TRIPLE_WORD, (7, 14): Premium.TRIPLE_WORD,
    (14, 0): Premium.TRIPLE_WORD, (14, 7): Premium.TRIPLE_WORD, (14, 14): Premium.TRIPLE_WORD,
    # Double word squares
    (1, 1): Premium.DOUBLE_WORD, (2, 2): Premium.DOUBLE_WORD, (3, 3): Premium.DOUBLE_WORD,
    (4, 4): Premium.DOUBLE_WORD, (7, 7): Premium.DOUBLE_WORD,
    (10, 10): Premium.DOUBLE_WORD, (11, 11): Premium.DOUBLE_WORD,
    (12, 12): Premium.DOUBLE_WORD, (13, 13): Premium.DOUBLE_WORD,
    # Triple letter squares
    (1, 5): Premium.TRIPLE_LETTER, (1, 9): Premium.TRIPLE_LETTER,
    (5, 1): Premium.TRIPLE_LETTER, (5, 5): Premium.TRIPLE_LETTER,
    (5, 9): Premium.TRIPLE_LETTER, (5, 13): Premium.TRIPLE_LETTER,
    (9, 1): Premium.TRIPLE_LETTER, (9, 5): Premium.TRIPLE_LETTER,
    (9, 9): Premium.TRIPLE_LETTER, (9, 13): Premium.TRIPLE_LETTER,
    (13, 5): Premium.TRIPLE_LETTER, (13, 9): Premium.TRIPLE_LETTER,
    # Double letter squares
    (0, 3): Premium.DOUBLE_LETTER, (0, 11): Premium.DOUBLE_LETTER,
    (2, 6): Premium.DOUBLE_LETTER, (2, 8): Premium.DOUBLE_LETTER,
    (3, 0): Premium.DOUBLE_LETTER, (3, 7): Premium.DOUBLE_LETTER, (3, 14): Premium.DOUBLE_LETTER,
    (6, 2): Premium.DOUBLE_LETTER, (6, 6): Premium.DOUBLE_LETTER,
    (6, 8): Premium.DOUBLE_LETTER, (6, 12): Premium.DOUBLE_LETTER,
    (7, 3): Premium.DOUBLE_LETTER, (7, 11): Premium.DOUBLE_LETTER,
    (8, 2): Premium.DOUBLE_LETTER, (8, 6): Premium.DOUBLE_LETTER,
    (8, 8): Premium.DOUBLE_LETTER, (8, 12): Premium.DOUBLE_LETTER,
    (11, 0): Premium.DOUBLE_LETTER, (11, 7): Premium.DOUBLE_LETTER, (11, 14): Premium.DOUBLE_LETTER,
    (12, 6): Premium.DOUBLE_LETTER, (12, 8): Premium.DOUBLE_LETTER,
    (14, 3): Premium.DOUBLE_LETTER, (14, 11): Premium.DOUBLE_LETTER,
}


def build_premium_squares() -> Dict[Tuple[int, int], str]:
    """Return a complete dictionary of premium squares with symmetry applied."""
    premiums: Dict[Tuple[int, int], str] = {}
    for (r, c), val in _PREMIUM_TEMPLATE.items():
        premiums[(r, c)] = val
        premiums[(r, 14 - c)] = val
        premiums[(14 - r, c)] = val
        premiums[(14 - r, 14 - c)] = val
    return premiums


PREMIUM_SQUARES: Dict[Tuple[int, int], str] = build_premium_squares()


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class Square:
    """A square on the board, holding either a tile or empty."""
    letter: Optional[str] = None  # actual letter placed; '*' denotes blank
    is_blank: bool = False  # True if this letter was a blank tile
    premium: str = Premium.NONE  # The premium on this square (applies only when filled)

    def display(self) -> str:
        if self.letter is None:
            # Show premium codes on empty squares to help players choose positions
            return self.premium or " . "
        # Show letter; blanks display in lower case to distinguish
        return f" {self.letter.lower() if self.is_blank else self.letter} "


class Board:
    """Representation of the game board."""

    def __init__(self) -> None:
        # 15x15 grid of squares with initial premiums
        self.size = 15
        self._load_words()
        self.grid: List[List[Square]] = []
        for r in range(self.size):
            row: List[Square] = []
            for c in range(self.size):
                premium = PREMIUM_SQUARES.get((r, c), Premium.NONE)
                row.append(Square(letter=None, is_blank=False, premium=premium))
            self.grid.append(row)

    def _load_words(self):
        with open("Wordlist.txt") as f:
            self.words = f.read().splitlines()

    def is_empty(self) -> bool:
        """Return True if no tiles have been placed on the board."""
        for row in self.grid:
            for sq in row:
                if sq.letter is not None:
                    return False
        return True

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def get_square(self, r: int, c: int) -> Square:
        return self.grid[r][c]

    def place_move(self, word: str, r: int, c: int, direction: str,
                   rack: List[str]) -> Tuple[bool, int, List[str]]:
        """
        Attempt to place a word on the board.

        Parameters
        ----------
        word: str
            The uppercase word to place.  Blanks are represented by '?' and must
            be replaced by a real letter supplied after the command (e.g.
            'EXCHANGE' command).  In this simplified game, we assume players
            replace '?' in their word before calling this method.
        r, c: int
            Starting row and column (zero‑based).
        direction: str
            'H' for horizontal, 'V' for vertical.
        rack: list[str]
            The player's current rack.  Letters used will be removed.

        Returns
        -------
        success: bool
            True if the move is legal and was applied.
        score: int
            The total score obtained from this move, including cross words and
            bonuses.
        used_letters: list[str]
            The letters that were consumed from the player's rack (for
            replenishing if illegal move).
        """
        # Normalize inputs
        direction = direction.upper()
        if direction not in ("H", "V"):
            return False, 0, []
        word = word.upper()

        # Keep track of letters taken from the rack
        used_from_rack: List[str] = []
        placements: List[Tuple[int, int, str, bool]] = []  # row, col, letter, is_blank

        # Determine if the placement stays within bounds and does not conflict
        for index, letter in enumerate(word):
            rr = r + (index if direction == "V" else 0)
            cc = c + (index if direction == "H" else 0)
            if not self.in_bounds(rr, cc):
                print("Move goes out of bounds.")
                return False, 0, []
            square = self.get_square(rr, cc)
            if square.letter is None:
                # Need to use a tile from rack
                if letter == '?':
                    print("Please specify the actual letter for '?' blanks before placing.")
                    return False, 0, []
                if letter not in rack:
                    print(f"You don't have letter '{letter}' in your rack.")
                    return False, 0, []
                used_from_rack.append(letter)
                placements.append((rr, cc, letter, False))
            else:
                # The letter on the board must match
                if square.letter != letter:
                    print(f"Board has '{square.letter}' at {(rr, cc)}; cannot place '{letter}'.")
                    return False, 0, []
                # This tile is not taken from rack
        # First move must cover the centre square
        if self.is_empty():
            # The center is (7,7)
            covers_center = False
            for index in range(len(word)):
                rr = r + (index if direction == "V" else 0)
                cc = c + (index if direction == "H" else 0)
                if rr == 7 and cc == 7:
                    covers_center = True
                    break
            if not covers_center:
                print("The first word must cover the center square (7,7).")
                return False, 0, []
        else:
            # Later moves must connect to existing tiles
            touching = False
            for (rr, cc, letter, _) in placements:
                # Check adjacent squares for existing letters
                # Horizontal: check up/down; Vertical: check left/right
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if self.in_bounds(nr, nc) and self.get_square(nr, nc).letter is not None:
                        touching = True
                        break
                if touching:
                    break
            # Also check if any part of the word sits on existing letters
            for index, letter in enumerate(word):
                rr = r + (index if direction == "V" else 0)
                cc = c + (index if direction == "H" else 0)
                if self.get_square(rr, cc).letter is not None:
                    touching = True
                    break
            if not touching:
                print("New word must connect to existing tiles.")
                return False, 0, []
        # Compute score: main word plus cross words
        total_score = 0
        used_tiles_count = len(used_from_rack)
        # To calculate main word score, we build the full word with existing letters
        main_word_letters: List[Tuple[str, int, int, bool]] = []  # letter, row, col, is_new
        for index, letter in enumerate(word):
            rr = r + (index if direction == "V" else 0)
            cc = c + (index if direction == "H" else 0)
            square = self.get_square(rr, cc)
            is_new_tile = square.letter is None
            main_word_letters.append((letter, rr, cc, is_new_tile))

        # Score the main word
        word_score = 0
        word_multiplier = 1
        for letter, rr, cc, is_new_tile in main_word_letters:
            letter_value = tile_points(letter)
            if is_new_tile:
                premium = self.get_square(rr, cc).premium
                if premium == Premium.DOUBLE_LETTER:
                    letter_value *= 2
                elif premium == Premium.TRIPLE_LETTER:
                    letter_value *= 3
                elif premium == Premium.DOUBLE_WORD:
                    word_multiplier *= 2
                elif premium == Premium.TRIPLE_WORD:
                    word_multiplier *= 3
            word_score += letter_value
        word_score *= word_multiplier
        total_score += word_score

        # Score cross words for each new tile if applicable
        for letter, rr, cc, is_new_tile in main_word_letters:
            if not is_new_tile:
                continue
            # Determine the perpendicular direction
            if direction == "H":
                # vertical cross word
                cross_letters: List[Tuple[str, int, int, bool]] = []
                # find start of cross word
                sr = rr
                # move up
                while sr > 0 and self.get_square(sr - 1, cc).letter is not None:
                    sr -= 1
                # build cross word downward
                endr = rr
                while endr < self.size and self.get_square(endr, cc).letter is not None:
                    endr += 1
                # If cross word length > 1, compute score
                if endr - sr > 1:
                    cross_word_score = 0
                    cross_word_multiplier = 1
                    for tr in range(sr, endr):
                        sq = self.get_square(tr, cc)
                        ch = sq.letter if sq.letter is not None else letter
                        # If this position is the newly placed letter
                        if tr == rr:
                            ch_value = tile_points(letter)
                            prem = sq.premium
                            if prem == Premium.DOUBLE_LETTER:
                                ch_value *= 2
                            elif prem == Premium.TRIPLE_LETTER:
                                ch_value *= 3
                            elif prem == Premium.DOUBLE_WORD:
                                cross_word_multiplier *= 2
                            elif prem == Premium.TRIPLE_WORD:
                                cross_word_multiplier *= 3
                        else:
                            # existing letter; no premium
                            ch_value = tile_points(ch)
                        cross_word_score += ch_value
                    cross_word_score *= cross_word_multiplier
                    total_score += cross_word_score
            else:  # direction == "V"
                # horizontal cross word
                cross_letters = []
                sc = cc
                while sc > 0 and self.get_square(rr, sc - 1).letter is not None:
                    sc -= 1
                endc = cc
                while endc < self.size and self.get_square(rr, endc).letter is not None:
                    endc += 1
                if endc - sc > 1:
                    cross_word_score = 0
                    cross_word_multiplier = 1
                    for tc in range(sc, endc):
                        sq = self.get_square(rr, tc)
                        ch = sq.letter if sq.letter is not None else letter
                        if tc == cc:
                            ch_value = tile_points(letter)
                            prem = sq.premium
                            if prem == Premium.DOUBLE_LETTER:
                                ch_value *= 2
                            elif prem == Premium.TRIPLE_LETTER:
                                ch_value *= 3
                            elif prem == Premium.DOUBLE_WORD:
                                cross_word_multiplier *= 2
                            elif prem == Premium.TRIPLE_WORD:
                                cross_word_multiplier *= 3
                        else:
                            ch_value = tile_points(ch)
                        cross_word_score += ch_value
                    cross_word_score *= cross_word_multiplier
                    total_score += cross_word_score

        # 40‑point bonus for using all seven tiles【872144310017704†L299-L303】
        if used_tiles_count == 7:
            total_score += 40

        # If we reach here, the move is valid; apply it
        for (rr, cc, letter, is_blank) in placements:
            sq = self.get_square(rr, cc)
            sq.letter = letter
            sq.is_blank = is_blank
            # After use, premium no longer applies in future moves
            sq.premium = Premium.NONE
        # Remove used letters from rack
        for letter in used_from_rack:
            rack.remove(letter)
        return True, total_score, used_from_rack

    def render(self) -> None:
        """Print the current board to the console with coordinates and premiums."""
        # Header
        header = "    " + " ".join(f"{c:2d}" for c in range(self.size))
        print(header)
        for r in range(self.size):
            row_str = f"{r:2d} "
            for c in range(self.size):
                sq = self.get_square(r, c)
                if sq.letter is None:
                    if sq.premium == Premium.DOUBLE_LETTER:
                        row_str += " DL"
                    elif sq.premium == Premium.TRIPLE_LETTER:
                        row_str += " TL"
                    elif sq.premium == Premium.DOUBLE_WORD:
                        row_str += " DW"
                    elif sq.premium == Premium.TRIPLE_WORD:
                        row_str += " TW"
                    else:
                        row_str += "  ."
                else:
                    row_str += f"  {sq.letter.lower() if sq.is_blank else sq.letter}"
            print(row_str)


class Player:
    def __init__(self, name: str, human=True) -> None:
        self.name = name
        self.human = human
        self.rack: List[str] = []
        self.score: int = 0

        if not human:
            self._instance_AI()

    def draw_tiles(self, bag: List[str], n: int) -> None:
        """Draw up to n tiles from the bag into the player's rack."""
        for _ in range(n):
            if not bag:
                break
            self.rack.append(bag.pop())

    def rack_string(self) -> str:
        return " ".join(self.rack)

    def _instance_AI(self):
        pass

    def take_turn(self, board_layout):
        AI_input = []
        AI_input += self.rack
        # TODO implement the walkway between the AI
        return "PASS"


def is_valid_word(word: str, board) -> bool:
    """
    Placeholder for word validation.

    The current implementation accepts any sequence of letters A–Z or
    asterisks ('*') for blanks.  To use a real dictionary, replace the
    body of this function with a lookup into your preferred word list.
    """

    if board is None:
        return False
    if not word:
        return False
    for ch in word:
        if ch not in string.ascii_letters + '*':
            return False
    if word.lower() in board.words:
        return True
    else:
        return False


def replenish_rack(player: Player, bag: List[str]) -> None:
    """Refill a player's rack up to 7 tiles."""
    draw_count = 7 - len(player.rack)
    player.draw_tiles(bag, draw_count)


def subtract_remaining_tiles(p1: Player, p2: Player) -> None:
    """At the end of the game, adjust scores based on remaining tiles."""
    # Sum the points of remaining tiles on racks
    p1_sum = sum(tile_points(ch) for ch in p1.rack if ch != '*')
    p2_sum = sum(tile_points(ch) for ch in p2.rack if ch != '*')
    # Subtract from each player
    p1.score -= p1_sum
    p2.score -= p2_sum
    # Add opponents' leftover values to each other
    p1.score += p2_sum
    p2.score += p1_sum


def play_game() -> None:
    """Main game loop for interactive play."""
    print("Welcome to the Crossplay CLI!")
    print("This is an approximation of NYT's Crossplay using the standard Scrabble board.")
    board = Board()
    bag = generate_tile_bag()
    # Set up players
    p1_name = "Player1"
    p2_name = "Player2"
    player1 = Player(p1_name, True)
    player2 = Player(p2_name, True)
    players = [player1, player2]
    # Initial draw
    for p in players:
        p.draw_tiles(bag, 7)

    current_index = 0
    passes_in_row = 0
    bag_emptied_turns: Dict[int, int] = {0: -1, 1: -1}
    while True:
        player = players[current_index]
        # Show board and rack
        print("\nCurrent board:")
        board.render()
        print(f"\n{player.name}'s turn. Score: {player.score}")
        print(f"Your rack: {player.rack_string()}")
        # Check if bag is empty
        bag_empty = (len(bag) == 0)
        # If bag is empty and this player hasn't taken a turn since it emptied
        if bag_empty and bag_emptied_turns[current_index] == -1:
            bag_emptied_turns[current_index] = 0  # mark that this is their final opportunity
        # Prompt for move
        if player.human == True:
            cmd = input("Enter move (WORD ROW COL DIR) or PASS or EXCHANGE letters: ").strip()
        else:
            cmd = player.take_turn(board.grid)
            pass  #TODO allow the bot to take a turn
        if not cmd:
            cmd = "PASS"
        parts = cmd.split()
        if parts[0].upper() == "PASS":
            print(f"{player.name} passes.")
            passes_in_row += 1
            if bag_empty and bag_emptied_turns[current_index] == 0:
                bag_emptied_turns[current_index] = 1
            # If both players pass consecutively, game ends
            if passes_in_row >= 2:
                print("Both players and a human passed consecutively. Game over.")
                break
            # Switch to next player
            current_index = 1 - current_index
            continue
        elif parts[0].upper() == "EXCHANGE":
            # Exchange tiles back into the bag
            if len(parts) < 2:
                print("Specify the letters you wish to exchange, e.g. EXCHANGE ABC")
                continue
            exchange_letters = parts[1].upper()
            if len(bag) == 0:
                print("Cannot exchange after the bag is empty.")
                continue
            success = True
            returned_tiles: List[str] = []
            for ch in exchange_letters:
                if ch not in player.rack:
                    print(f"You don't have letter '{ch}' to exchange.")
                    success = False
                    break
                returned_tiles.append(ch)
            if not success:
                continue
            # Remove returned tiles from rack and put back into bag
            for ch in returned_tiles:
                player.rack.remove(ch)
                bag.append(ch)
            random.shuffle(bag)
            # Draw replacement tiles
            player.draw_tiles(bag, len(returned_tiles))
            print(f"Exchanged {len(returned_tiles)} tiles.")
            passes_in_row = 0
            current_index = 1 - current_index
            continue
        else:
            # Attempt to parse move
            if len(parts) != 4:
                print("Invalid move format. Use WORD ROW COL DIR or PASS.")
                continue
            word, row_str, col_str, direction = parts
            if not row_str.isdigit() or not col_str.isdigit():
                print("Row and column must be numbers between 0 and 14.")
                continue
            row = int(row_str)
            col = int(col_str)
            word = word.upper()
            if not is_valid_word(word, board):
                print("Invalid word. Words must consist of letters A-Z or '*' for blanks.")
                continue
            # In this simplified implementation we do not support blanks ('?') in input
            if '?' in word:
                print("This CLI does not support '?' placeholders. Replace them with the intended letter.")
                continue
            success, move_score, used_from_rack = board.place_move(word, row, col, direction, player.rack)
            if success:
                passes_in_row = 0
                player.score += move_score
                print(f"Move accepted for {move_score} points. Total: {player.score}")
                # Draw new tiles
                replenish_rack(player, bag)
                if bag_empty and bag_emptied_turns[current_index] == 0:
                    bag_emptied_turns[current_index] = 1
                # If both players have taken their final turn after bag is empty and cannot play, end game
                # We'll check after switching players
                current_index = 1 - current_index
            else:
                # If move invalid, return used tiles (not removed yet) implicitly, nothing to do
                print("Move was invalid. Try again.")
                continue
        # Check equal turn end condition
        if bag_empty and bag_emptied_turns[0] == 1 and bag_emptied_turns[1] == 1:
            # Both have taken final turn; end game if one more pass or invalid move happens
            # We'll allow them to continue placing moves until they can't
            # If next player has no possible tiles, the game will end naturally by passes
            pass
    # Game over - final scoring
    subtract_remaining_tiles(player1, player2)
    print("\nFinal board:")
    board.render()
    print(f"\n{player1.name}: {player1.score} points")
    print(f"{player2.name}: {player2.score} points")
    if player1.score > player2.score:
        print(f"{player1.name} wins!")
    elif player2.score > player1.score:
        print(f"{player2.name} wins!")
    else:
        print("It's a tie!")


import random


class env_wrapper():
    """Wrapper class for the environment."""

    def __init__(self):
        self.player1 = None
        self.player2 = None
        self.board = None
        self.bag = None

        self.reset(0)

    def reset(self, seed):
        random.seed(seed)
        self.player1 = Player("Champion", False)
        self.player2 = Player("Challenger", False)
        self.board = None
        self.board = Board
        self._play_game()


    def step(self):
        # here we're going to construct the observation


    def _play_game(self):
        """Main game loop for interactive play."""
        print("Welcome to the Crossplay CLI!")
        print("This is an approximation of NYT's Crossplay using the standard Scrabble board.")
        board = self.board
        bag = generate_tile_bag()
        # Set up players
        p1_name = "Player1"
        p2_name = "Player2"
        players = [self.player1, self.player2]
        # Initial draw
        for p in players:
            p.draw_tiles(bag, 7)

        current_index = 0
        passes_in_row = 0
        bag_emptied_turns: Dict[int, int] = {0: -1, 1: -1}
        while True:
            player = players[current_index]
            # Show board and rack
            # print("\nCurrent board:")
            # board.render()
            # print(f"\n{player.name}'s turn. Score: {player.score}")
            # print(f"Your rack: {player.rack_string()}")
            # Check if bag is empty
            bag_empty = (len(bag) == 0)
            # If bag is empty and this player hasn't taken a turn since it emptied
            if bag_empty and bag_emptied_turns[current_index] == -1:
                bag_emptied_turns[current_index] = 0  # mark that this is their final opportunity
            # Prompt for move
            # if player.human == True:
            #     cmd = input("Enter move (WORD ROW COL DIR) or PASS or EXCHANGE letters: ").strip()
            # else:
            cmd = player.take_turn(board.grid)
            parts = cmd.split()

            # Attempt to parse move
            if len(parts) != 4:
                print("Invalid move format. Use WORD ROW COL DIR or PASS.")
                continue
            word, row_str, col_str, direction = parts
            if not row_str.isdigit() or not col_str.isdigit():
                print("Row and column must be numbers between 0 and 14.")
                continue
            row = int(row_str)
            col = int(col_str)
            word = word.upper()
            if not is_valid_word(word, board):
                print("Invalid word. Words must consist of letters A-Z or '*' for blanks.")
                continue
            # In this simplified implementation we do not support blanks ('?') in input
            if '?' in word:
                print("This CLI does not support '?' placeholders. Replace them with the intended letter.")
                continue
            success, move_score, used_from_rack = board.place_move(word, row, col, direction, player.rack)
            if success:
                passes_in_row = 0
                player.score += move_score
                print(f"Move accepted for {move_score} points. Total: {player.score}")
                # Draw new tiles
                replenish_rack(player, bag)
                if bag_empty and bag_emptied_turns[current_index] == 0:
                    bag_emptied_turns[current_index] = 1
                # If both players have taken their final turn after bag is empty and cannot play, end game
                # We'll check after switching players
                current_index = 1 - current_index
            else:
                # If move invalid, return used tiles (not removed yet) implicitly, nothing to do
                print("Move was invalid. Try again.")
                passes_in_row += 1
                if passes_in_row > 5:
                    print("NO VALID MOVES FOR 5 TURNS: Game over.")
                    break
                continue
            # Check equal turn end condition
            if bag_empty and bag_emptied_turns[0] == 1 and bag_emptied_turns[1] == 1:
                # Both have taken final turn; end game if one more pass or invalid move happens
                # We'll allow them to continue placing moves until they can't
                # If next player has no possible tiles, the game will end naturally by passes
                pass
        # Game over - final scoring
        subtract_remaining_tiles(self.player1, self.player2)
        print("\nFinal board:")
        board.render()
        print(f"\n{self.player1.name}: {self.player1.score} points")
        print(f"{self.player2.name}: {self.player2.score} points")
        if self.player1.score > self.player2.score:
            print(f"{self.player1.name} wins!")
        elif self.player2.score > self.player1.score:
            print(f"{self.player2.name} wins!")
        else:
            print("It's a tie!")
        pass


if __name__ == "__main__":
    play_game()
