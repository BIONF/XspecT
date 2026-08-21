"""Base probabilistic filter model for sequence data"""

# pylint: disable=no-name-in-module, too-many-instance-attributes

import json
from math import ceil
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from rbloom import Bloom
from xxhash import xxh3_64_intdigest
from xspect.models.probabilistic_filter_model import ProbabilisticFilterModel
from xspect.file_io import get_record_iterator


class ProbabilisticSingleFilterModel(ProbabilisticFilterModel):
    """
    Probabilistic filter model for sequence data, with a single filter

    This model uses a Bloom filter to store k-mers from the training sequences. It is designed to
    be used with a single filter, which is suitable e. g. for genus-level classification.
    """

    def __init__(
        self,
        k: int,
        model_display_name: str,
        author: str | None,
        author_email: str | None,
        model_type: str,
        base_path: Path,
        fpr: float = 0.01,
        training_accessions: list[str] | None = None,
    ) -> None:
        """Initialize probabilistic single filter model.

        This model uses a Bloom filter to store k-mers from the training sequences. It is designed
        to be used with a single filter, which is suitable e.g. for genus-level classification.

        Args:
            k (int): Length of the k-mers to use for filtering
            model_display_name (str): Display name of the model
            author (str | None): Author of the model
            author_email (str | None): Email of the author
            model_type (str): Type of the model, e.g. "probabilistic_single_filter"
            base_path (Path): Base path where the model will be saved
            fpr (float): False positive rate for the Bloom filter, default is 0.01
            training_accessions (list[str] | None): List of accessions used for training
        """
        super().__init__(
            k=k,
            model_display_name=model_display_name,
            author=author,
            author_email=author_email,
            model_type=model_type,
            base_path=base_path,
            fpr=fpr,
            num_hashes=1,
            training_accessions=training_accessions,
        )
        self.bf = None

    def fit(
        self,
        file_path: Path,
        display_name: str,
        training_accessions: list[str] | None = None,
    ) -> None:
        """
        Fit the bloom filter to the sequences.

        Trains the model by reading sequences from the provided file path,
        generating k-mers, and adding them to the Bloom filter.

        Args:
            file_path (Path): Path to the file containing sequences in FASTA format
            display_name (str): Display name for the model
            training_accessions (list[str] | None): List of accessions used for training
        """
        self.training_accessions = training_accessions

        # estimate number of kmers
        total_length = 0
        for record in get_record_iterator(file_path):
            total_length += len(record.seq)
        num_kmers = total_length - self.k + 1

        self.bf = Bloom(num_kmers, self.fpr, hash_func=xxh3_64_intdigest)
        for record in get_record_iterator(file_path):
            for kmer in self._generate_kmers(record.seq):
                self.bf.add(kmer.encode())
        self.display_names[file_path.stem] = display_name

        bloom_path = self.base_path / self.slug() / "filter.bloom"
        bloom_path.parent.mkdir(parents=True, exist_ok=True)
        self.bf.save(str(bloom_path))

    def calculate_hits(
        self, sequence: Seq | SeqRecord, exclude_ids=None, step: int = 1
    ) -> dict:
        """
        Calculate the hits for the sequence

        Calculates the number of k-mers in the sequence that are present in the Bloom filter.

        Args:
            sequence (Seq | SeqRecord): Sequence to calculate hits for
            exclude_ids (list[str] | None): List of IDs to exclude, default is None
            step (int): Step size for generating k-mers, default is 1
        Returns:
            dict: Dictionary with the display name as key and the number of hits as value
        """
        if isinstance(sequence, SeqRecord):
            sequence = sequence.seq

        if not isinstance(sequence, Seq):
            raise ValueError("Invalid sequence, must be a Bio.Seq object")

        if not len(sequence) > self.k:
            raise ValueError("Invalid sequence, must be longer than k")

        num_hits = sum(
            1
            for kmer in self._generate_kmers(sequence, step=step)
            if kmer.encode() in self.bf
        )
        return {next(iter(self.display_names)): num_hits}

    @staticmethod
    def load(path: Path) -> "ProbabilisticSingleFilterModel":
        """
        Load the model from disk

        This method reads the model's JSON file and the associated Bloom filter file,
        reconstructing the model instance.

        Args:
            path (Path): Path to the model directory containing the JSON file
        Returns:
            ProbabilisticSingleFilterModel: An instance of the model loaded from disk
        """
        with open(path, "r", encoding="utf-8") as file:
            json_object = file.read()
            model_json = json.loads(json_object)
            model = ProbabilisticSingleFilterModel(
                model_json["k"],
                model_json["model_display_name"],
                model_json["author"],
                model_json["author_email"],
                model_json["model_type"],
                path.parent,
                fpr=model_json["fpr"],
                training_accessions=model_json["training_accessions"],
            )
            model.display_names = model_json["display_names"]
            bloom_path = model.base_path / model.slug() / "filter.bloom"
            model.bf = Bloom.load(
                str(bloom_path),
                hash_func=xxh3_64_intdigest,
            )
            return model

    def _generate_kmers(self, sequence: Seq, step: int = 1):
        """
        Generate kmers from the sequence

        Generates k-mers from the sequence, considering both the forward and reverse complement
        strands.

        Args:
            sequence (Seq): Sequence to generate k-mers from
            step (int): Step size for generating k-mers, default is 1
        Yields:
            str: The minimizer k-mer (the lexicographically smallest k-mer between the forward and
                reverse complement)
        """
        num_kmers = ceil((len(sequence) - self.k + 1) / step)
        for i in range(num_kmers):
            start_pos = i * step
            kmer = sequence[start_pos : start_pos + self.k]
            minimizer = min(kmer, str(kmer.reverse_complement()))
            yield str(minimizer)
