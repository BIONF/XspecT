"""NCBI handler for the NCBI Datasets API."""

import shutil
from enum import Enum
from pathlib import Path
import time
from loguru import logger
import requests
import zipfile

# pylint: disable=line-too-long


class AssemblyLevel(Enum):
    """Enum for the assembly level."""

    REFERENCE = "reference"
    COMPLETE_GENOME = "complete_genome"
    CHROMOSOME = "chromosome"
    SCAFFOLD = "scaffold"
    CONTIG = "contig"


class AssemblySource(Enum):
    """Enum for the assembly source."""

    REFSEQ = "refseq"
    GENBANK = "genbank"


class NCBIHandler:
    """
    This class uses the NCBI Datasets API to get data about taxa and their assemblies.

    It provides methods to get taxon IDs, species, names, accessions, and download assemblies.
    It also enforces rate limiting to comply with NCBI's API usage policies.
    """

    def __init__(
        self,
        api_key: str | None = None,
    ):
        """
        Initialise the NCBI handler.

        This method sets up the base URL for the NCBI Datasets API and initializes the rate limiting parameters.

        Args:
            api_key (str | None): The NCBI API key. If None, the handler will use the public API without an API key.
        """
        self.api_key = api_key
        self.base_url = "https://api.ncbi.nlm.nih.gov/datasets/v2"
        self.last_request_time = 0.0
        self.min_interval = (
            1 / 10 if api_key else 1 / 5
        )  # NCBI allows 10 requests per second with if an API key, otherwise 5 requests per second

    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting for the NCBI Datasets API.

        This method ensures that the requests to the API are limited to 5 requests per second
        without an API key and 10 requests per second with an API key.
        It uses a simple time-based approach to enforce the rate limit.
        """
        now = time.time()
        elapsed_time = now - self.last_request_time
        if elapsed_time < self.min_interval:
            time.sleep(self.min_interval - elapsed_time)
        self.last_request_time = now

    def _make_request(self, endpoint: str, timeout: int = 15) -> dict:
        """
        Make a request to the NCBI Datasets API.

        This method constructs the full URL for the API endpoint, adds the necessary headers (including the API key if provided),
        and makes a GET request to the API. It also enforces rate limiting before making the request.

        Args:
            endpoint (str): The endpoint to make the request to.
            timeout (int, optional): The timeout for the request in seconds. Defaults to 10.

        Returns:
            dict: The response from the API.
        """
        self._enforce_rate_limit()

        endpoint = endpoint if endpoint.startswith("/") else "/" + endpoint
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        response = requests.get(
            self.base_url + endpoint, headers=headers, timeout=timeout
        )
        if response.status_code != 200:
            response.raise_for_status()

        return response.json()

    def get_genus_taxon_id(self, genus: str) -> int:
        """
        Get the taxon id for a given genus name.

        This function checks if the genus name is valid by making a request to the NCBI Datasets API.
        If the genus name is valid, it returns the taxon id.
        If the genus name is not valid, it raises an exception.

        Args:
            genus (str): The genus name to validate.

        Returns:
            int: The taxon id for the given genus name.

        Raises:
            ValueError: If the genus name is not valid.
        """
        endpoint = f"/taxonomy/taxon/{genus}"
        response = self._make_request(endpoint)

        try:
            taxonomy = response["taxonomy_nodes"][0]["taxonomy"]

            taxon_id = taxonomy["tax_id"]
            rank = taxonomy["rank"]
            lineage = taxonomy["lineage"]

            if rank != "GENUS":
                raise ValueError(f"Genus name {genus} is not a genus.")
            if lineage[2] != 2:
                raise ValueError(f"Genus name {genus} does not belong to bacteria.")

            return taxon_id
        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid genus name: {genus}") from e

    def get_species(self, genus_id: int) -> list[int]:
        """
        Get species ids for a given genus id.

        This function makes a request to the NCBI Datasets API to get the species for a given genus id.
        Children with the rank "SPECIES_GROUP" (e.g. species complexes) are traversed recursively,
        so that the species classified below them are included as well.
        Only taxa with the rank "SPECIES" are returned.

        Args:
            genus_id (int): The genus id to get the species for.

        Returns:
            list[int]: A list containing the species taxnomy ids.
        """
        try:
            return self._get_species_in_subtree(genus_id)
        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid genus id: {genus_id}") from e

    def _get_species_in_subtree(self, taxon_id: int) -> list[int]:
        """
        Get the species ids in the subtree of a given taxon id.

        Species groups are traversed recursively. Children with other ranks are ignored.

        Args:
            taxon_id (int): The taxon id whose subtree should be searched.

        Returns:
            list[int]: A list containing the species taxonomy ids.
        """
        endpoint = f"/taxonomy/taxon/{taxon_id}/filtered_subtree"
        edges = self._make_request(endpoint)["edges"]

        species_ids = []
        for child_id in edges[str(taxon_id)]["visible_children"]:
            rank = edges.get(str(child_id), {}).get("rank")
            if rank == "SPECIES":
                species_ids.append(child_id)
            elif rank == "SPECIES_GROUP":
                species_ids.extend(self._get_species_in_subtree(child_id))
        return list(dict.fromkeys(species_ids))

    def get_taxon_names(self, taxon_ids: list[int]) -> dict[int, str]:
        """
        Get the names for a given list of taxon ids.

        This function makes a request to the NCBI Datasets API to get the names for a given list of taxon ids.
        It returns a dictionary with the taxon ids as keys and the names as values.

        Args:
            taxon_ids (list[int]): The list of taxon ids to get the names for.

        Returns:
            dict[int, str]: A dictionary containing the taxon ids and their corresponding names.
        """
        if len(taxon_ids) > 1000:
            raise ValueError("Maximum number of taxon ids is 1000.")
        if len(taxon_ids) < 1:
            raise ValueError("At least one taxon id is required.")

        endpoint = f"/taxonomy/taxon/{','.join(map(str, taxon_ids))}?page_size=1000"
        response = self._make_request(endpoint)

        try:
            taxon_names = {
                int(taxonomy_node["taxonomy"]["tax_id"]): taxonomy_node["taxonomy"][
                    "organism_name"
                ]
                for taxonomy_node in response["taxonomy_nodes"]
            }
            if len(taxon_names) != len(taxon_ids):
                raise ValueError("Not all taxon ids were found.")
        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid taxon ids: {taxon_ids}") from e

        return taxon_names

    def get_accessions(
        self,
        taxon_id: int,
        assembly_level: AssemblyLevel,
        assembly_source: AssemblySource,
        count: int,
        min_n50: int,
        exclude_atypical: bool,
        allow_inconclusive: bool,
        exclude_paired_reports: bool = True,
        current_version_only: bool = True,
    ) -> list[str]:
        """
        Get the accessions for a given taxon id.

        This function makes a request to the NCBI Datasets API to get the accessions for a given taxon id.
        It filters the accessions based on the assembly level, assembly source, and other parameters.
        It returns a list with the respective accessions.

        Args:
            taxon_id int: The taxon id to get the accessions for.
            assembly_level (AssemblyLevel): The assembly level to get the accessions for.
            assembly_source (AssemblySource): The assembly source to get the accessions for.
            count (int): The number of accessions to get.
            min_n50 (int): The minimum contig n50 to filter the accessions.
            exclude_atypical (bool): Whether to exclude atypical accessions.
            allow_inconclusive (bool): Whether to allow accessions with an inconclusive taxonomy check status.
            exclude_paired_reports (bool, optional): Whether to exclude paired reports. Defaults to True.
            current_version_only (bool, optional): Whether to get only the current version of the accessions. Defaults to True.


        Returns:
            list[str]: A list containing the accessions.
        """
        endpoint = (
            f"/genome/taxon/{taxon_id}/dataset_report?"
            f"filters.tax_exact_match=false&"
            f"filters.assembly_source={assembly_source.value}&"
            f"filters.exclude_atypical={exclude_atypical}&"
            f"filters.exclude_paired_reports={exclude_paired_reports}&"
            f"filters.current_version_only={current_version_only}&"
            f"page_size={count * 2}&"  # to avoid having less than count if n50 or ANI is not met
        )
        endpoint += (
            "filters.reference_only=true&"
            if assembly_level == AssemblyLevel.REFERENCE
            else f"filters.assembly_level={assembly_level.value}"
        )

        response = self._make_request(endpoint)
        try:
            accessions = [
                report["accession"]
                for report in response["reports"]
                if report["assembly_stats"]["contig_n50"] >= min_n50
                and (
                    allow_inconclusive
                    or report["average_nucleotide_identity"]["taxonomy_check_status"]
                    == "OK"
                )
            ]
        except (IndexError, KeyError, TypeError):
            logger.debug(
                f"Could not get {assembly_level.value} accessions for taxon with ID: {taxon_id}. Skipping."
            )
            return []
        return accessions[:count]  # Limit to count

    def get_highest_quality_accessions(
        self,
        taxon_id: int,
        assembly_source: AssemblySource,
        count: int,
        min_n50: int,
        exclude_atypical: bool,
        allow_inconclusive: bool,
    ) -> list[str]:
        """
        Get the highest quality accessions for a given taxon id (based on the assembly level).

        This function iterates through the assembly levels in order of quality and retrieves accessions
        until the specified count is reached. It ensures that the accessions are unique and sorted by quality.

        Args:
            taxon_id (int): The taxon id to get the accessions for.
            assembly_source (AssemblySource): The assembly source to get the accessions for.
            count (int): The number of accessions to get.
            min_n50 (int): The minimum contig n50 to filter the accessions.
            exclude_atypical (bool): Whether to exclude atypical accessions.
            allow_inconclusive (bool): Whether to allow accessions with an inconclusive taxonomy check status.

        Returns:
            list[str]: A list containing the highest quality accessions.
        """
        accessions = []
        for assembly_level in list(AssemblyLevel):
            accessions += self.get_accessions(
                taxon_id,
                assembly_level,
                assembly_source,
                count,
                min_n50=min_n50,
                exclude_atypical=exclude_atypical,
                allow_inconclusive=allow_inconclusive,
            )
            if len(set(accessions)) >= count:
                break
        return list(set(accessions))[:count]  # Remove duplicates and limit to count

    def download_assemblies(self, accessions: list[str], output_dir: Path) -> None:
        """
        Download assemblies for a list of accessions.

        This function makes a request to the NCBI Datasets API to download the assemblies for the given accessions.
        It saves the downloaded assemblies as a zip file in the specified output directory.

        Args:
            accessions (list[str]): A list of accessions to download.
            output_dir (Path): The directory where the downloaded assemblies will be saved.
        """
        endpoint = f"/genome/accession/{','.join(accessions)}/download?include_annotation_type=GENOME_FASTA"

        self._enforce_rate_limit()

        response = requests.get(self.base_url + endpoint, stream=True, timeout=15)
        if response.status_code != 200:
            response.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "ncbi_dataset.zip", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def download_reference_genome(self, taxon_id: int, output_dir: Path) -> Path | None:
        """
        Notes:
        Developed by Oemer Cetin as part of a Bsc thesis at Goethe University Frankfurt am Main (2025).
        (An Integration of Alignment-Free and Alignment-Based Approaches for Bacterial Taxon Assignment)

        Downloads the reference genome from the RefSeq-DB for a given taxon ID.

        This function queries the NCBI Datasets API for the reference genome and downloads it.

        Args:
            taxon_id (int): The taxonomy ID of the species.
            output_dir (Path): Directory where the genome will be saved.

        Returns:
            Path: Path to the downloaded ZIP file.
        """
        accessions = self.get_accessions(
            taxon_id=taxon_id,
            assembly_level=AssemblyLevel.REFERENCE,
            assembly_source=AssemblySource.REFSEQ,
            count=1,  # only one reference exists
            min_n50=0,
            exclude_atypical=True,
            allow_inconclusive=False,
        )

        if not accessions:
            return None

        logger.info(
            f"Downloading reference genome for taxon {taxon_id}: {accessions[0]}"
        )
        self.download_assemblies(accessions, output_dir)

        zip_path = output_dir / "ncbi_dataset.zip"

        fna_file = ""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith(".fna"):
                    extracted_path = zip_ref.extract(file, path=output_dir)
                    fna_file = output_dir / f"{taxon_id}.fna"
                    Path(extracted_path).rename(
                        fna_file
                    )  # consistent file name (tax_id)
                    logger.info(f"Extracted reference genome to {fna_file}")
                    break

        # clean up
        zip_path.unlink()
        shutil.rmtree(output_dir / "ncbi_dataset")

        return fna_file
