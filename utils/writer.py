import csv
import os
import warnings
import numpy as np
from typing import Dict, List, Optional, TypeAlias, TypeVar

from core import ChannelResults, ResultsBase, Metrics, sort_channel_results_by_metric
from utils.reader import read_csv_to_channel_results
from visualization.barcode import generate_combined_barcode, generate_comparison_barcodes
from core.config import ComparisonConfig

warnings.filterwarnings("ignore")


R = TypeVar("R", bound=ResultsBase)

ExtraColumns: TypeAlias = Dict[str, List[str]]


def _common_source_mode(results: List[ChannelResults]):
    """The AnalysisMode every row was read back under, or None if they disagree.

    Aggregation pools CSVs the caller chose, and nothing stops those being a 2D run and a
    volumetric one. Mixing them is already meaningless -- the same column is an area in
    one and a volume in the other -- so return None and let the writer fall back to the
    legacy layout rather than silently stamping one mode's schema onto the other's rows.
    """
    modes = {getattr(r, "source_mode", None) for r in results}
    if len(modes) != 1:
        if len(modes) > 1:
            names = sorted(m.key if m else "unknown" for m in modes)
            print(
                f"Warning: aggregating results from more than one analysis mode "
                f"({', '.join(names)}). Size columns do not mean the same thing across "
                f"modes -- an area in one is a volume in another.",
                flush=True,
            )
        return None
    return modes.pop()


def _families_present(results: List[ChannelResults]) -> Dict[str, bool]:
    """Which optional families these rows actually carry, by the same test the writer uses."""
    from core.results import OPTIONAL_FAMILIES

    return {
        f.switch: any(
            getattr(r, f.attribute, None) is not None
            and getattr(r, f.attribute).is_populated()
            for r in results
        )
        for f in OPTIONAL_FAMILIES
    }


def results_to_csv(
    results: List[R],
    output_filepath: str,
    extra_columns: Optional[ExtraColumns] = None,
    physical_units: bool = False,
    **kwargs,
) -> None:
    """Write homogeneous results to a CSV file."""
    assert len(results) > 0, "Results list cannot be empty."

    if extra_columns:
        for col_name, values in extra_columns.items():
            assert len(values) == len(
                results
            ), f"Extra column '{col_name}' length ({len(values)}) must match results length ({len(results)})."

    # All results must be the same type
    expected_type = type(results[0])
    for i, result in enumerate(results[1:], 1):
        assert (
            type(result) == expected_type
        ), f"All results must be the same type. Result {i} is {type(result).__name__}, expected {expected_type.__name__}"

    # The requested representation must actually have been computed.
    #
    # This used to demand that the fractional value be NaN exactly when physical units
    # were requested -- true of the 2D branch, which fills either the fractional or the
    # quantity fields and leaves the other NaN. The volumetric branch fills BOTH, since a
    # fraction of the analysed volume and a size in um^3 are both meaningful there, so the
    # old form rejected every 3D run with more than one row and made the um^3 columns
    # unwritable. Those are the only crop-independent size numbers a volumetric run has.
    #
    # The real invariant is weaker: whichever representation is being written must carry
    # data whenever the other one does. That still catches the mismatch the check existed
    # for -- asking for physical units from a run that only computed fractions -- while
    # accepting a branch that populates both, and without firing on a legitimately empty
    # result where every size is NaN.
    quantified = physical_units

    for i, result in enumerate(results):
        fractional_nan = np.isnan(result.binarization.get_data()[2])
        physical_nan = np.isnan(result.binarization.get_physical_data()[2])
        requested_nan, other_nan = (
            (physical_nan, fractional_nan) if quantified
            else (fractional_nan, physical_nan)
        )
        assert not (requested_nan and not other_nan), (
            f"Result {i} has no "
            f"{'physical (um^2 / um^3)' if quantified else 'fractional'} size data, but "
            f"the other representation is populated, so the requested columns would be "
            f"empty. Check enable_physical_units against the branch that produced these "
            f"results."
        )

    # Ensure the directory exists
    assert os.path.exists(
        os.path.dirname(output_filepath)
    ), "Output directory does not exist."

    # The analysis mode decides which metric families this CSV carries. Mesh columns
    # stay conditional on data actually being present, so an xyzt run that skipped
    # meshing does not emit nine empty columns.
    # An optional family is emitted only when some row actually carries data for it,
    # so a run that skipped meshing does not produce a block of empty columns. Driven by
    # the registry, so the writer and the barcode cannot disagree about which families
    # are present -- the last bug here was a CSV gaining columns the barcode did not
    # render.
    from core.results import OPTIONAL_FAMILIES

    for family in OPTIONAL_FAMILIES:
        if family.switch in kwargs or not hasattr(results[0], family.attribute):
            continue
        populated = any(
            getattr(r, family.attribute, None) is not None
            and getattr(r, family.attribute).is_populated() for r in results)
        kwargs = dict(kwargs, **{family.switch: populated})

    headers = results[0].get_physical_headers(**kwargs) if quantified else results[0].get_headers(**kwargs)
    if extra_columns:
        headers = list(extra_columns.keys()) + headers

    with open(output_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for i, result in enumerate(results):
            row = []
            if extra_columns:
                for col_name in extra_columns.keys():
                    row.append(extra_columns[col_name][i])
            new_row = result.get_physical_data(**kwargs) if quantified else result.get_data(**kwargs)
            row.extend(new_row)
            writer.writerow(row)
    return quantified


def generate_aggregate_csv(
    csv_files: List[str],
    output_csv: str,
    gen_barcode: bool = False,
    sort_metric: Optional[str] = None,
    separate_channels: bool = True,
    metrics_to_visualize: List[bool] = None,
) -> None:
    """
    Clean version of aggregate CSV generation using structured data.

    Args:
        csv_files: List of CSV file paths to aggregate
        output_csv: Output path for the aggregate CSV
        gen_barcode: Whether to generate barcode visualization
        sort_metric: Optional metric name to sort by (e.g. "Mean Speed")
        separate_channels: Whether to create separate barcode figures per channel
    """

    if not csv_files:
        return

    all_results = []

    # Read each CSV file back into ChannelResults
    for csv_file in csv_files:
        try:
            results = read_csv_to_channel_results(csv_file)
            all_results.extend(results)
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
            continue

    if not all_results:
        print("No valid data found in CSV files")
        return

    # Sort if requested
    if sort_metric:
        sort_channel_results_by_metric(all_results, sort_metric)

    mode = _common_source_mode(all_results)

    if not (len(csv_files) == 1 and csv_files[0] == output_csv):
        # Write aggregate CSV using the clean writer, under the SCHEMA THE ROWS CAME IN.
        # Without `mode` the writer falls back to the 2D layout, so aggregating xyzt CSVs
        # produced a header saying "Maximum Island Area" (um^2 in the physical variant)
        # over values that are um^3 volumes, and an xyz aggregate regained seven
        # Speed/Divergence/Curl columns that mode never computes. It also let every
        # optional family through unfiltered, giving a header no mode can produce -- which
        # then failed the reader's own `headers in accepted_headers` assert, so the
        # aggregate could not be read back at all.
        quantified = results_to_csv(all_results, output_csv, just_metrics=False, mode=mode)
    else:
        quantified = bool(np.isnan(all_results[0].binarization.get_data()[2]) and (not 
                      np.isnan(all_results[0].binarization.get_physical_data()[2])))

    # Generate barcode if requested
    if gen_barcode:
        barcode_path = output_csv.replace(".csv", " Barcode")
        # A mask built positionally against the mode-less 25 metrics cannot match the
        # column count the renderer derives from the families these results actually
        # carry, and the renderer refuses a mask of the wrong length -- so aggregating
        # volumetric CSVs raised ValueError out of an un-caught call, losing the barcode
        # after the CSV had already been written. Drop a mask that does not fit and show
        # everything, saying so, rather than failing: the mask only trims the picture.
        shown = metrics_to_visualize or None
        if shown is not None:
            expected = len(ChannelResults.get_metrics(
                just_metrics=True, mode=mode, **_families_present(all_results)))
            if len(shown) != expected:
                print(
                    f"Note: the saved barcode metric selection has {len(shown)} entries "
                    f"but these results carry {expected} metrics, so it does not apply "
                    f"to them; showing all columns.",
                    flush=True,
                )
                shown = None
        generate_combined_barcode(
            all_results, barcode_path,
            separate_channels=separate_channels,
            physical_units = quantified,
            metrics_to_visualize= shown,
            mode=mode,
        )

def compare_multiple_csvs(
    csv_files: List[str],
    sort_metric: Optional[str] = None,
    separate_channels: bool = False,
) -> None:
    """
    Clean version of aggregate CSV generation using structured data.

    Args:
        csv_files: List of CSV file paths to aggregate
        sort_metric: Optional metric name to sort by (e.g. "Mean Speed")
        separate_channels: Whether to create separate barcode figures per channel
    """

    if not csv_files:
        return

    all_results = []

    # Read each CSV file back into ChannelResults
    for csv_file in csv_files:
        try:
            results = read_csv_to_channel_results(csv_file)
            if sort_metric:
                sort_channel_results_by_metric(results, sort_metric)
            all_results.append(results)
            quantified = bool(np.isnan(all_results[0][0].binarization.get_data()[2]) and (not 
                      np.isnan(all_results[0][0].binarization.get_physical_data()[2])))
            assert ((np.isnan(results[0].binarization.get_data()[2]) and (not 
                      np.isnan(results[0].binarization.get_physical_data()[2]))) == quantified
        ), f"All results must have the same headers. Result headers do not match."
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
            continue

    if not all_results:
        print("No valid data found in CSV files")
        return
    
    barcode_list = [csv_path.replace(".csv", " Barcode") for csv_path in csv_files]
    
    generate_comparison_barcodes(all_results, barcode_list, separate_channels)

def create_metric_comparison(
    compare_config: ComparisonConfig
) -> None:
    csv_file = compare_config.csv_location
    output_file = compare_config.output_location
    first_metric = compare_config.first_comparison_metric
    second_metric = compare_config.second_comparison_metric
    if not (csv_file and output_file):
        return
    results = read_csv_to_channel_results(csv_file)
    # Built for the mode and families these rows carry. With the mode-less list a
    # volumetric metric name simply was not in `metrics`, and the lookups below index
    # [0] into the empty match -- so comparing any 3D metric died with a bare IndexError
    # naming nothing.
    mode = _common_source_mode(results)
    families = _families_present(results)
    metrics = ChannelResults.get_metrics(mode=mode, **families)
    file_metric = metrics[0]

    def find(name):
        match = [metric for metric in metrics if metric.value == name]
        if not match:
            raise ValueError(
                f"{name!r} is not a metric of this CSV. It carries: "
                f"{', '.join(m.value for m in metrics[1:])}."
            )
        return match[0]

    first_metric, second_metric = find(first_metric), find(second_metric)
    dict_data = [result.get_dict_data(mode=mode, **families) for result in results]
    files = [row[file_metric] for row in dict_data]
    param1 = [row[first_metric] for row in dict_data]
    param2 = [row[second_metric] for row in dict_data]
    headers = ["File", first_metric.value, second_metric.value]
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for file, p1, p2 in zip(files, param1, param2):
            if p1 == np.nan or p2 == np.nan:
                continue
            writer.writerow([file, p1, p2])
