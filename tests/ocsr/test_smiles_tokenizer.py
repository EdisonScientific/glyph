"""Tests for the upstream-aligned SMILES char tokenizer."""

import pytest
from rdkit import Chem, RDLogger

from glyph.ocsr.smiles_tokenizer import (
    EOS_ID,
    MASK_ID,
    PAD_ID,
    SOS_ID,
    UNK_ID,
    CharSmilesTokenizer,
)

RDLogger.DisableLog("rdApp.*")


def test_special_token_ids_match_upstream_contract():
    tok = CharSmilesTokenizer()
    assert tok.stoi[CharSmilesTokenizer.PAD] == PAD_ID == 0
    assert tok.stoi[CharSmilesTokenizer.SOS] == SOS_ID == 1
    assert tok.stoi[CharSmilesTokenizer.EOS] == EOS_ID == 2
    assert tok.stoi[CharSmilesTokenizer.UNK] == UNK_ID == 3
    assert tok.stoi[CharSmilesTokenizer.MASK] == MASK_ID == 4


def test_vocab_size_is_101():
    tok = CharSmilesTokenizer()
    assert tok.vocab_size == 101


def test_encode_decode_roundtrip_on_real_smiles():
    tok = CharSmilesTokenizer()
    for smi in ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "[Na+].[Cl-]"]:
        ids = tok.encode(smi)
        assert ids[0] == SOS_ID and ids[-1] == EOS_ID
        # No <unk> for canonical SMILES characters
        assert UNK_ID not in ids
        out = tok.decode(ids)
        assert out == smi
        # Make sure the decoded SMILES still parses with RDKit.
        assert Chem.MolFromSmiles(out) is not None


def test_unknown_character_maps_to_unk():
    tok = CharSmilesTokenizer()
    # Use a character not in the 101-entry vocab.
    weird = "CéC"  # 'é' is not in vocab
    ids = tok.encode(weird, add_special=False)
    assert UNK_ID in ids


def test_padded_encode_and_truncation():
    tok = CharSmilesTokenizer()
    ids = tok.encode_padded("CCO", max_len=8)
    assert len(ids) == 8
    assert ids[0] == SOS_ID
    # last non-pad token must be EOS
    last_non_pad = next(i for i in reversed(ids) if i != PAD_ID)
    assert last_non_pad == EOS_ID
    # truncation forces EOS as the final non-pad token even at max_len boundary
    long_smi = "C" * 32
    ids2 = tok.encode_padded(long_smi, max_len=10)
    assert len(ids2) == 10
    assert ids2[-1] == EOS_ID


def test_decode_strips_specials_until_eos():
    tok = CharSmilesTokenizer()
    # SOS, C, C, O, EOS, then garbage that must be ignored by default
    ids = [SOS_ID, tok.stoi["C"], tok.stoi["C"], tok.stoi["O"], EOS_ID, PAD_ID, PAD_ID]
    assert tok.decode(ids) == "CCO"


def test_validate_rejects_inconsistent_vocab():
    bad = {"<pad>": 9, "<sos>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4}
    with pytest.raises(ValueError):
        CharSmilesTokenizer(stoi=bad)
