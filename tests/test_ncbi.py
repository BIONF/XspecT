"""Test the NCBIHandler class."""

import os
import pytest
from xspect.handlers.ncbi import NCBIHandler, AssemblyLevel, AssemblySource

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="module")
def ncbi_handler():
    """Fixture for the NCBI class."""

    ncbi_api_key = os.environ.get("NCBI_API_KEY")

    return NCBIHandler(api_key=ncbi_api_key)


def test_get_genus_taxon_id(ncbi_handler):
    """Test the get_genus_taxon_id method of the NCBI class."""
    genus = "Escherichia"
    taxon_id = ncbi_handler.get_genus_taxon_id(genus)
    assert taxon_id == 561


def test_get_genus_taxon_id_invalid(ncbi_handler):
    """Test the get_genus_taxon_id method of the NCBI class with an invalid genus."""
    genus = "InvalidGenus"
    with pytest.raises(ValueError):
        ncbi_handler.get_genus_taxon_id(genus)


def test_get_genus_taxon_wrong_rank(ncbi_handler):
    """Test the get_genus_taxon_id method of the NCBI class with the wrong input rank."""
    genus = "Acinetobacter baumannii"
    with pytest.raises(ValueError):
        ncbi_handler.get_genus_taxon_id(genus)


def test_get_genus_taxon_id_no_bacteria(ncbi_handler):
    """Test the get_genus_taxon_id method of the NCBI class with a genus that is not a bacterium."""
    genus = "Arabidopsis"
    with pytest.raises(ValueError):
        ncbi_handler.get_genus_taxon_id(genus)


def test_get_species(ncbi_handler):
    """Test the get_species method of the NCBI class."""
    genus_id = 469
    species_ids = ncbi_handler.get_species(genus_id)
    assert isinstance(species_ids, list)
    assert len(species_ids) > 0
    assert 40216 in species_ids  # radioresistens
    assert 470 in species_ids  # baumannii, part of the calcoaceticus/baumannii complex
    assert 909768 not in species_ids  # calcoaceticus/baumannii complex (species group)


def test_get_taxon_names(ncbi_handler):
    """Test the get_taxon_names method of the NCBI class."""
    taxon_ids = [470, 471]
    taxon_names = ncbi_handler.get_taxon_names(taxon_ids)
    assert isinstance(taxon_names, dict)
    assert len(taxon_names) == 2
    assert taxon_names[470] == "Acinetobacter baumannii"
    assert taxon_names[471] == "Acinetobacter calcoaceticus"


def test_get_accessions(
    ncbi_handler,
):
    """Test the get_accessions method of the NCBI class."""
    taxon_id = 470
    accessions = ncbi_handler.get_accessions(
        taxon_id, AssemblyLevel.REFERENCE, AssemblySource.REFSEQ, 1, 10000, True, False
    )
    assert isinstance(accessions, list)
    assert len(accessions) > 0
    assert "GCF_009035845.1" in accessions


def test_get_accessions_inconclusive(ncbi_handler):
    """Test the get_accessions method with a genus with an inconclusive taxonomy check status."""
    taxon_id = 955
    min_n50 = 10000
    exclude_atypical = True
    allow_inconclusive = False

    accessions = ncbi_handler.get_accessions(
        taxon_id,
        AssemblyLevel.CHROMOSOME,
        AssemblySource.REFSEQ,
        1,
        min_n50,
        exclude_atypical,
        allow_inconclusive,
    )
    assert isinstance(accessions, list)
    assert len(accessions) == 0

    allow_inconclusive = True

    accessions_inconclusive = ncbi_handler.get_accessions(
        taxon_id,
        AssemblyLevel.REFERENCE,
        AssemblySource.REFSEQ,
        1,
        min_n50,
        exclude_atypical,
        allow_inconclusive,
    )
    assert isinstance(accessions_inconclusive, list)
    assert len(accessions_inconclusive) > 0


def test_get_highest_quality_accessions(ncbi_handler):
    """Test the get_highest_quality_accessions method of the NCBI class."""
    taxon_id = 470
    num_accessions = 2
    min_n50 = 10000
    exclude_atypical = True
    allow_inconclusive = False
    accessions = ncbi_handler.get_highest_quality_accessions(
        taxon_id,
        AssemblySource.REFSEQ,
        num_accessions,
        min_n50,
        exclude_atypical,
        allow_inconclusive,
    )
    assert isinstance(accessions, list)
    assert len(accessions) == num_accessions
    assert "GCF_009035845.1" in accessions


def test_download_assemblies(ncbi_handler, tmp_path):
    """Test the download_assemblies method of the NCBI class."""
    accessions = ["GCF_009035845.1", "GCF_024749785.1"]
    ncbi_handler.download_assemblies(accessions, tmp_path)

    downloaded_file = tmp_path / "ncbi_dataset.zip"
    assert downloaded_file.is_file()
    assert downloaded_file.stat().st_size > 0


def test_download_reference_genome(ncbi_handler, tmp_path):
    """Test the download_assemblies method of the NCBI class."""
    tax_id = 470
    file_path = ncbi_handler.download_reference_genome(tax_id, tmp_path)

    assert file_path.is_file()
