"""Tests for ChartokCoordsTokenizer interleaving."""

from glyph.ocsr.smiles_tokenizer import ChartokCoordsTokenizer


def test_chartok_smiles_no_coords_round_trips():
    tok = ChartokCoordsTokenizer(coord_bins=64, sep_xy=True)
    smi = "CCO"
    ids, atom_pos = tok.encode_chartok_coords(smi, coords=None, max_len=16)
    assert ids[0] == tok.SOS_ID
    assert tok.EOS_ID in ids
    out = tok.decode_chartok_coords(ids)
    assert out == smi
    # atom positions point to last char of each atom (no coords inserted)
    assert len(atom_pos) == 3


def test_chartok_with_coords_interleaves_correctly():
    tok = ChartokCoordsTokenizer(coord_bins=64, sep_xy=True)
    smi = "CCO"
    coords = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]]
    ids, atom_pos = tok.encode_chartok_coords(smi, coords=coords, max_len=32)
    # Decoded SMILES strips coord tokens
    out = tok.decode_chartok_coords(ids)
    assert out == smi
    # Each atom appended 2 coord ids
    assert len(atom_pos) == 3
    for k in range(3):
        # atom_pos[k] points at a y-coord token
        idv = ids[atom_pos[k]]
        assert tok.is_y(idv)
        # the previous id at atom_pos[k]-1 must be an x-coord
        assert tok.is_x(ids[atom_pos[k] - 1])


def test_chartok_handles_two_char_atoms_and_aromatic():
    tok = ChartokCoordsTokenizer()
    smi = "Clc1ccccc1Br"
    coords = [[0.0, 0.0]] * 8  # 8 atoms: Cl, c, c, c, c, c, c, Br
    ids, atom_pos = tok.encode_chartok_coords(smi, coords, max_len=64)
    out = tok.decode_chartok_coords(ids)
    assert out == smi
    assert len(atom_pos) == 8


def test_chartok_total_vocab_size_default_is_229():
    tok = ChartokCoordsTokenizer(coord_bins=64, sep_xy=True)
    assert tok.total_vocab_size == 101 + 64 + 64  # 229


def test_chartok_truncation_and_padding():
    tok = ChartokCoordsTokenizer()
    long_smi = "C" * 80
    ids, _ = tok.encode_chartok_coords(long_smi, coords=None, max_len=20)
    assert len(ids) == 20
    # last non-pad token is EOS
    nonpad = [i for i in ids if i != tok.PAD_ID]
    assert nonpad[-1] == tok.EOS_ID
