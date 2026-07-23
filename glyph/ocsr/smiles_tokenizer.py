"""Char-level SMILES tokenizer aligned to MolNexTR/MolScribe `vocab_chars.json`.

This tokenizer produces token IDs that match those upstream repos. The vocab
file is vendored under ``glyph/ocsr/vocab/vocab_chars.json`` (sha256
sanity-check available at import time) so we do not depend on the external
repos at runtime.

Special tokens follow the upstream contract:

- ``<pad>=0``, ``<sos>=1``, ``<eos>=2``, ``<unk>=3``, ``<mask>=4``.

The vocab covers SMILES punctuation, digits, lowercase + uppercase letters,
plus a handful of misc symbols, for a total of 101 entries. Chartok-coords
and edges are *not* implemented in this module; coord-bin ids would live at
``101..101+coord_bins`` (and again ``..+coord_bins`` if ``sep_xy`` is True).
We will add those when we wire the multitask training target.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
MASK = "<mask>"

# sha256 of the bytes of the vendored vocab file. If we ever update the
# vendored vocab, this constant must be updated too.
VENDORED_VOCAB_SHA256 = "74a1bea43c1488383498d68521cf4f5527aacfb084450a6e455583ed2c0cdfb1"


def _load_vendored_vocab() -> dict[str, int]:
    with resources.files("glyph.ocsr.vocab").joinpath("vocab_chars.json").open("rb") as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != VENDORED_VOCAB_SHA256:
        raise RuntimeError(
            "Vendored vocab_chars.json sha256 mismatch: "
            f"expected {VENDORED_VOCAB_SHA256}, got {digest}. "
            "If this change is intentional, update VENDORED_VOCAB_SHA256."
        )
    return json.loads(raw.decode("utf-8"))


class CharSmilesTokenizer:
    """Character-level tokenizer over the upstream `vocab_chars.json`."""

    PAD = PAD
    SOS = SOS
    EOS = EOS
    UNK = UNK
    MASK = MASK
    PAD_ID = PAD_ID
    SOS_ID = SOS_ID
    EOS_ID = EOS_ID
    UNK_ID = UNK_ID
    MASK_ID = MASK_ID

    def __init__(self, stoi: dict[str, int] | None = None):
        if stoi is None:
            stoi = _load_vendored_vocab()
        self._validate(stoi)
        self.stoi = dict(stoi)
        self.itos = {i: tok for tok, i in self.stoi.items()}

    @staticmethod
    def _validate(stoi: dict[str, int]) -> None:
        for tok, expected in [
            (PAD, PAD_ID),
            (SOS, SOS_ID),
            (EOS, EOS_ID),
            (UNK, UNK_ID),
            (MASK, MASK_ID),
        ]:
            if stoi.get(tok) != expected:
                raise ValueError(
                    f"Vocab inconsistent: {tok!r} expected id {expected}, got {stoi.get(tok)}"
                )
        # All ids must be a contiguous range starting from 0.
        ids = sorted(stoi.values())
        if ids != list(range(len(ids))):
            raise ValueError("Vocab ids must be a contiguous 0..N-1 range")

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, smiles: str, add_special: bool = True) -> list[int]:
        ids: list[int] = []
        if add_special:
            ids.append(SOS_ID)
        for ch in smiles:
            ids.append(self.stoi.get(ch, UNK_ID))
        if add_special:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        chars: list[str] = []
        for i in ids:
            if strip_special:
                if i == EOS_ID:
                    break
                if i in (PAD_ID, SOS_ID):
                    continue
            tok = self.itos.get(int(i), UNK)
            chars.append(tok)
        # Keep <unk> markers visible so a downstream RDKit parse will fail
        # cleanly rather than silently dropping characters.
        return "".join(chars)

    def encode_padded(self, smiles: str, max_len: int, add_special: bool = True) -> list[int]:
        ids = self.encode(smiles, add_special=add_special)
        if len(ids) > max_len:
            ids = [*ids[: max_len - 1], EOS_ID]
        return ids + [PAD_ID] * (max_len - len(ids))


def get_default_tokenizer() -> CharSmilesTokenizer:
    return CharSmilesTokenizer()


# ---- chartok_coords extension ----------------------------------


# A SMILES atom is one of: bracketed atom [...], two-char element (Cl, Br),
# single-char organic subset element, or aromatic single-char (lowercase).
_ATOM_TWO_CHAR = {"Cl", "Br"}
_ATOM_ORGANIC = set("BCNOSPFI")
_ATOM_AROMATIC = set("bcnops")


def _atom_tokens(smiles: str) -> list[tuple[int, int]]:
    """Return list of (start, end) char ranges for atom tokens in ``smiles``.

    Handles ``[...]`` bracketed atoms, ``Cl``/``Br`` two-char organic atoms,
    single-char organic atoms (uppercase), and lowercase aromatic atoms.
    Returns ranges in source order.
    """

    spans: list[tuple[int, int]] = []
    i = 0
    n = len(smiles)
    while i < n:
        ch = smiles[i]
        if ch == "[":
            j = smiles.find("]", i + 1)
            if j == -1:
                break
            spans.append((i, j + 1))
            i = j + 1
        elif ch.isalpha():
            if i + 1 < n and (smiles[i : i + 2] in _ATOM_TWO_CHAR):
                spans.append((i, i + 2))
                i += 2
            elif (ch.isupper() and ch in _ATOM_ORGANIC) or (ch.islower() and ch in _ATOM_AROMATIC):
                spans.append((i, i + 1))
                i += 1
            else:
                # Other letters (e.g., chirality H inside brackets are handled by [...])
                i += 1
        else:
            i += 1
    return spans


class ChartokCoordsTokenizer(CharSmilesTokenizer):
    """Tokenizer that interleaves x/y coordinate bins with SMILES chars.

    Layout per atom: ``[char1][char2]...[charN][x_id][y_id]``. Final sequence
    is ``[SOS]...interleaved...[EOS]`` with optional pad. Coord bins are at
    ids ``[offset, offset+coord_bins)`` for x, and (with sep_xy=True)
    ``[offset+coord_bins, offset+2*coord_bins)`` for y.
    """

    def __init__(self, coord_bins: int = 64, sep_xy: bool = True, stoi=None):
        super().__init__(stoi=stoi)
        self.coord_bins = coord_bins
        self.sep_xy = sep_xy
        self.offset = len(self.stoi)
        self.maxx = coord_bins
        self.maxy = coord_bins
        # Total vocab covers chars + 2*coord_bins (or 1*coord_bins if not sep_xy).
        self.coord_vocab = (2 if sep_xy else 1) * coord_bins
        self._total_vocab = self.offset + self.coord_vocab

    @property
    def total_vocab_size(self) -> int:
        return self._total_vocab

    # ---- coord <-> id helpers ----
    def x_to_id(self, x: float) -> int:
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y: float) -> int:
        if self.sep_xy:
            return self.offset + self.maxx + round(y * (self.maxy - 1))
        return self.offset + round(y * (self.maxy - 1))

    def id_to_x(self, idv: int) -> float:
        return (idv - self.offset) / (self.maxx - 1)

    def id_to_y(self, idv: int) -> float:
        if self.sep_xy:
            return (idv - self.offset - self.maxx) / (self.maxy - 1)
        return (idv - self.offset) / (self.maxy - 1)

    def is_x(self, idv: int) -> bool:
        return self.offset <= idv < self.offset + self.maxx

    def is_y(self, idv: int) -> bool:
        if self.sep_xy:
            return self.offset + self.maxx <= idv < self.offset + self.maxx + self.maxy
        return self.is_x(idv)

    def is_coord(self, idv: int) -> bool:
        return self.is_x(idv) or self.is_y(idv)

    # ---- encode/decode ----
    def encode_chartok_coords(
        self,
        smiles: str,
        coords: list[list[float]] | None,
        max_len: int,
    ) -> tuple[list[int], list[int]]:
        """Encode SMILES with optional coords. Returns (ids, atom_indices_in_seq).

        atom_indices_in_seq[k] = position in the *padded* output sequence of the
        last token belonging to atom k (i.e. the y-coord token if coords are
        provided, else the last char of the atom token).
        """

        ids: list[int] = [self.stoi[self.SOS]]
        atom_positions: list[int] = []
        spans = _atom_tokens(smiles)
        cursor = 0
        for atom_idx, (span_lo, span_hi) in enumerate(spans):
            # Emit non-atom characters between cursor and span_lo
            for ch in smiles[cursor:span_lo]:
                ids.append(self.stoi.get(ch, self.UNK_ID))
            # Emit atom token characters
            for ch in smiles[span_lo:span_hi]:
                ids.append(self.stoi.get(ch, self.UNK_ID))
            if coords is not None and atom_idx < len(coords):
                x, y = coords[atom_idx]
                ids.append(self.x_to_id(float(x)))
                ids.append(self.y_to_id(float(y)))
            atom_positions.append(len(ids) - 1)
            cursor = span_hi
        # Trailing non-atom tail
        for ch in smiles[cursor:]:
            ids.append(self.stoi.get(ch, self.UNK_ID))
        ids.append(self.stoi[self.EOS])
        if len(ids) > max_len:
            ids = [*ids[: max_len - 1], self.stoi[self.EOS]]
            atom_positions = [p for p in atom_positions if p < max_len]
        ids = ids + [self.stoi[self.PAD]] * (max_len - len(ids))
        return ids, atom_positions

    def decode_chartok_coords(self, ids: list[int]) -> str:
        """Decode by skipping coord-id tokens and stopping at EOS."""

        chars: list[str] = []
        for i in ids:
            if i == self.EOS_ID:
                break
            if i in (self.PAD_ID, self.SOS_ID):
                continue
            if self.is_coord(i):
                continue
            chars.append(self.itos.get(int(i), self.UNK))
        return "".join(chars)
