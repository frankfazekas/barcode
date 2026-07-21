import os, functools, builtins
from itertools import pairwise
import nd2, av
import imageio.v3 as iio
import numpy as np
from utils import vprint
from core import BarcodeConfig, InputConfig, ChannelResults, BinarizationResults, IntensityResults, FlowResults
from core.results import MeshResults

def check_first_frame_dim(file):
    min_intensity = np.min(file[0])
    mean_intensity = np.mean(file[0])
    return 2 * np.exp(-1) * mean_intensity <= min_intensity

def read_file(filepath, count_list, config: BarcodeConfig = None, in_config: InputConfig = None, accept_dim: bool = False, allow_large_files = True):
    print = functools.partial(builtins.print, flush=True)
    
    if count_list[1] != 1:    
        print(f'File {count_list[0]} of {count_list[1]}')
        print(filepath)
        count_list[0] += 1

    file_size = os.path.getsize(filepath)
    file_size_gb = file_size / (1024 ** 3)
    if file_size_gb > 5 and not allow_large_files:
        print("File size is too large -- this program does not process files larger than 5 GB.")
        return None
    if filepath.endswith(('.avi', '.mp4')):
        frames = []
        container = av.open(filepath)
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='gray'))
        file = np.array(frames)
        file = np.reshape(file, (file.shape + (1,))) if len(file.shape) == 3 else file
    if filepath.endswith(('.tif', '.tiff')):
        file = iio.imread(filepath)
        file = np.reshape(file, (file.shape + (1,))) if len(file.shape) == 3 else file
        if file.shape[3] != min(file.shape):
            file = np.swapaxes(np.swapaxes(file, 1, 2), 2, 3)
    elif filepath.endswith('.nd2'):
        with nd2.ND2File(filepath) as ndfile:
            if len(ndfile.sizes) >= 5:
                count_list[0] += 1
                raise TypeError("Incorrect file dimensions: file must be time series data with 1+ channels (4 dimensions total)")
            if "Z" in ndfile.sizes:
                count_list[0] += 1
                raise TypeError('Z-stack identified, skipping to next file...')
            if 'T' not in ndfile.sizes or len(ndfile.shape) <= 2 or ndfile.sizes['T'] <= 5:
                count_list[0] += 1
                raise TypeError('Too few frames, unable to capture dynamics, skipping to next file...')
            if ndfile == None:
                raise TypeError('Unable to read file, skipping to next file...')
            file = ndfile.asarray()
            if 'C' not in ndfile.sizes:
                file = np.expand_dims(file, axis=1)
            file = np.swapaxes(np.swapaxes(file, 1, 2), 2, 3)
            try:
                times = ndfile.events(orient="list")["Time [s]"]
                frame_interval = np.array([y - x for x, y in pairwise(times)]).mean()
                micron_pix_ratio = ndfile.voxel_size()[0]
                config.reader.exposure_time = float(frame_interval / in_config.time)
                config.reader.um_pixel_ratio = micron_pix_ratio / in_config.length
                vprint(f"Extracted ND2 metadata: frame_interval={frame_interval:.4f}s, micron_pixel_ratio={micron_pix_ratio:.2f}")
            except Exception as e:
                config.reader.exposure_time = 1
                config.reader.um_pixel_ratio = 1
                vprint(f"Warning: Could not extract ND2 metadata: {e}")
    if (file == 0).all():
        print('Empty file: can not process, skipping to next file...')
        return None
    
    if accept_dim == False and check_first_frame_dim(file) == True:
        print(filepath + 'is too dim, skipping to next file...')
        return None
    else:
        return file
    
def read_csv_to_channel_results(filepath: str) -> list[ChannelResults]:
    """Read results from a CSV file into a list of ChannelResults."""

    def get_value(value_str: str) -> float:
        """Convert string to float, handling empty strings as NaN."""
        if value_str == "" or value_str.lower() == "nan":
            return np.nan
        try:
            return float(value_str)
        except ValueError:
            # If conversion fails, return NaN
            return np.nan

    expected_headers = ChannelResults.get_headers(just_metrics=False)
    expected_physical_headers = ChannelResults.get_physical_headers(just_metrics=False)
    expected_v1_headers = expected_headers[:10] + expected_headers[15:-1]
    # Volumetric runs append the mesh/curvature family; see core.results.MeshResults.
    expected_mesh_headers = ChannelResults.get_headers(just_metrics=False, include_mesh=True)
    expected_mesh_physical_headers = ChannelResults.get_physical_headers(
        just_metrics=False, include_mesh=True)

    # Every mode's header set, derived from the registry rather than hard-coded, so
    # adding a mode cannot leave this check behind. A header list the reader does not
    # recognise used to drop every row silently.
    from core.modes import MODES

    accepted_headers = [expected_headers, expected_physical_headers, expected_v1_headers,
                        expected_mesh_headers, expected_mesh_physical_headers]
    # Every combination of mode and optional family, generated from the registry
    # rather than hard-coded: a header list this loop does not know about used to make
    # the reader drop every row in silence.
    from itertools import product

    from core.results import OPTIONAL_FAMILIES

    for _mode in MODES.values():
        for _combo in product((False, True), repeat=len(OPTIONAL_FAMILIES)):
            _base = {f.switch: on for f, on in zip(OPTIONAL_FAMILIES, _combo)}
            # include_flow varies too: a static z-stack drops its flow columns, so its
            # volumetric CSV must still read back. For 2D modes this changes nothing
            # (flow_is_populated always keeps them), so the extra entries never match.
            for _flow in (True, False):
                _switches = dict(_base, include_flow=_flow)
                accepted_headers.append(ChannelResults.get_headers(
                    just_metrics=False, mode=_mode, **_switches))
                accepted_headers.append(ChannelResults.get_physical_headers(
                    just_metrics=False, mode=_mode, **_switches))

    v1_header_length = 18 # Channel, 7 Image_Binarization, 6 Intensity_Distribution, 4 Optical_Flow
    v2_header_length = 26 # Channel, 12 Image_Binarization, 6 Intensity_Distribution, 7 Optical_Flow
    # Derived, never hardcoded: this slices the mesh family off the end of a row, so a
    # stale literal here would silently mis-assign every column in the block.
    mesh_block_length = len(MeshResults.get_metrics())
    v3_header_length = v2_header_length + mesh_block_length

    import csv

    results = []
    with open(filepath, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

        assert headers in accepted_headers, (
            f"CSV headers {headers} do not match any BARCODE header set "
            f"(modes: {', '.join(MODES)})"
        )

        # Identify the layout once, rather than inferring it per row from a length.
        layout = _identify_layout(headers)

        for row in reader:
            filename = row[0]
            flags = row.pop(2)
            data = [get_value(value) for value in row[1:]]
            if np.isnan(data[0]):
                raise ValueError(f"Invalid channel in row: {row}")

            # A recognised mode layout is parsed from its own metric lists. This runs
            # before the legacy handling below, which would otherwise strip the mesh
            # block off the row first and leave nothing for the mesh fields.
            if layout is not None:
                results.append(_build_from_layout(filename, flags, data, layout))
                continue

            mesh_values = None
            if len(data) == v3_header_length:
                mesh_values = data[-mesh_block_length:]
                data = data[:-mesh_block_length]

            # The branches below dispatch on the header list, so compare against the
            # headers with any mesh block removed -- otherwise a volumetric CSV matches
            # neither the base nor the physical set and the row is silently dropped.
            base_headers = (
                headers[:-mesh_block_length] if mesh_values is not None else headers
            )
            if len(data) == v1_header_length:
                results.append(
                    ChannelResults(
                        filepath = filename,
                        channel=int(data[0]),
                        total_flags=flags,
                        binarization=BinarizationResults(
                            connectivity=data[1],
                            max_island_size=data[2],
                            max_void_size=data[3],
                            max_island_percent_change=data[4],
                            max_void_percent_change=data[5],
                            island_size_initial=data[6],
                            island_size_initial2=data[7],
                        ),
                        intensity=IntensityResults(
                            max_kurtosis=data[8],
                            max_median_skew=data[9],
                            max_mode_skew=data[10],
                            kurtosis_diff=data[11],
                            median_skew_diff=data[12],
                            mode_skew_diff=data[13],
                        ),
                        flow=FlowResults(
                            mean_speed=data[14],
                            delta_speed=data[15],
                            mean_theta=data[16],
                            mean_sigma_theta=data[17],
                        ),
                    )
                )
            elif len(data) == v2_header_length:
                if base_headers == expected_headers:
                    results.append(ChannelResults(
                        filepath = filename,
                        channel = int(data[0]),
                        total_flags=flags,
                        binarization=BinarizationResults(
                            connectivity=data[1],
                            max_island_size=data[2],
                            max_void_size=data[3],
                            max_island_percent_change=data[4],
                            max_void_percent_change=data[5],
                            island_size_initial=data[6],
                            island_size_initial2=data[7],
                            island_anisotropy = data[8],
                            mean_island_size = data[9],
                            total_island_size = data[10],
                            mean_island_separation = data[11],
                            island_correlation_length = data[12],
                        ),
                        intensity=IntensityResults(
                            max_kurtosis=data[13],
                            max_median_skew=data[14],
                            max_mode_skew=data[15],
                            kurtosis_diff=data[16],
                            median_skew_diff=data[17],
                            mode_skew_diff=data[18],
                        ),
                        flow=FlowResults(
                            mean_speed=data[19],
                            delta_speed=data[20],
                            mean_theta=data[21],
                            mean_sigma_theta=data[22],
                            velocity_correlation_length=data[23],
                            divergence=data[24],
                            curl=data[25]
                        )
                    ))
                elif base_headers == expected_physical_headers:
                    results.append(
                        ChannelResults(
                            filepath=filename,
                            channel=int(data[0]),
                            total_flags=flags,
                            binarization=BinarizationResults(
                                connectivity=data[1],
                                max_island_size_quantity=data[2],
                                max_void_size_quantity=data[3],
                                max_island_percent_change=data[4],
                                max_void_percent_change=data[5],
                                island_size_initial_quantity=data[6],
                                island_size_initial2_quantity=data[7],
                                island_anisotropy = data[8],
                                mean_island_size_quantity = data[9],
                                total_island_size_quantity = data[10],
                                mean_island_separation = data[11],
                                island_correlation_length = data[12],
                            ),
                            intensity=IntensityResults(
                                max_kurtosis=data[13],
                                max_median_skew=data[14],
                                max_mode_skew=data[15],
                                kurtosis_diff=data[16],
                                median_skew_diff=data[17],
                                mode_skew_diff=data[18],
                            ),
                            flow=FlowResults(
                                mean_speed=data[19],
                                delta_speed=data[20],
                                mean_theta=data[21],
                                mean_sigma_theta=data[22],
                                velocity_correlation_length=data[23],
                                divergence=data[24],
                                curl=data[25]
                            )
                        )
                    )
            if mesh_values is not None and results:
                results[-1].mesh = MeshResults.from_values(mesh_values)
    return results


def _identify_layout(headers):
    """Match ``headers`` against each mode's header sets.

    Returns ``(mode, physical, include_mesh)`` or None when this is one of the legacy
    layouts, which the length-based branches below still handle.
    """
    from core.modes import MODES

    from itertools import product

    from core.results import OPTIONAL_FAMILIES

    for mode in MODES.values():
        for combo in product((False, True), repeat=len(OPTIONAL_FAMILIES)):
            base = {f.switch: on for f, on in zip(OPTIONAL_FAMILIES, combo)}
            # include_flow must be part of the matched layout: a flow-suppressed CSV has
            # seven fewer columns, and the returned switches carry include_flow so the row
            # is rebuilt over the right block. Try flow-present first, the common case.
            for flow in (True, False):
                switches = dict(base, include_flow=flow)
                kw = dict(just_metrics=False, mode=mode, **switches)
                if headers == ChannelResults.get_headers(**kw):
                    return (mode, False, switches)
                if headers == ChannelResults.get_physical_headers(**kw):
                    return (mode, True, switches)
    return None


def _build_from_layout(filename, flags, data, layout):
    """Rebuild a ChannelResults by position, using the layout's own metric lists.

    Field order here mirrors each results class's ``get_data``; taking the counts from
    ``get_metrics`` rather than hard-coding them means a family gaining a metric cannot
    silently shift everything after it.
    """
    from core.results import ComponentResults, MeshResults

    mode, physical, switches = layout
    channel = int(data[0])
    values = data[1:]

    n_bin = len(BinarizationResults.get_metrics(mode))
    n_int = len(IntensityResults.get_metrics(mode))
    # Respect the layout's flow switch, not just the mode: a static z-stack's CSV has no
    # flow columns, so reading n_flow off mode.supports_flow alone would slice seven values
    # that are not there and shift every family after it.
    with_flow = mode.supports_flow and switches.get("include_flow", True)
    n_flow = len(FlowResults.get_metrics()) if with_flow else 0
    from core.results import OPTIONAL_FAMILIES

    binar = values[:n_bin]
    inten = values[n_bin:n_bin + n_int]
    flow = values[n_bin + n_int:n_bin + n_int + n_flow]

    # Optional families follow in registry order; slice each off in turn so adding a
    # family cannot shift the ones after it.
    cursor = n_bin + n_int + n_flow
    family_values = {}
    for family in OPTIONAL_FAMILIES:
        if not switches.get(family.switch):
            continue
        width = len(family.results_cls.get_metrics(mode))
        family_values[family.attribute] = values[cursor:cursor + width]
        cursor += width

    size_kwargs = (
        dict(max_island_size_quantity=binar[1], max_void_size_quantity=binar[2],
             island_size_initial_quantity=binar[5], island_size_initial2_quantity=binar[6],
             mean_island_size_quantity=binar[8], total_island_size_quantity=binar[9])
        if physical else
        dict(max_island_size=binar[1], max_void_size=binar[2],
             island_size_initial=binar[5], island_size_initial2=binar[6],
             mean_island_size=binar[8], total_island_size=binar[9])
    )

    result = ChannelResults(
        filepath=filename,
        channel=channel,
        total_flags=flags,
        # Carry the layout forward so aggregation and comparison, which re-write results
        # they did not compute, can write them back under the schema they were read in.
        source_mode=mode,
        binarization=BinarizationResults(
            connectivity=binar[0],
            max_island_percent_change=binar[3],
            max_void_percent_change=binar[4],
            island_anisotropy=binar[7],
            mean_island_separation=binar[10],
            island_correlation_length=binar[11],
            **size_kwargs,
        ),
        intensity=IntensityResults(
            max_kurtosis=inten[0], max_median_skew=inten[1], max_mode_skew=inten[2],
            kurtosis_diff=inten[3], median_skew_diff=inten[4], mode_skew_diff=inten[5],
        ),
    )
    if n_flow:
        result.flow = FlowResults(
            mean_speed=flow[0], delta_speed=flow[1], mean_theta=flow[2],
            mean_sigma_theta=flow[3], velocity_correlation_length=flow[4],
            divergence=flow[5], curl=flow[6],
        )

    # Most families' dataclass fields are in the same order as their get_data(), so the
    # values zip straight back on without naming them here -- a family gaining a field
    # then needs no change in the reader.
    #
    # A family with a DERIVED column does not have that property. MeshResults writes
    # Concavity (= 1 - solidity) from `_CSV_FIELDS` with no backing field, so its block is
    # 12 wide against 11 fields; zip truncates silently and shifts every column from
    # Concavity onward, which read Mean Curvature <H> back as 1 - Solidity -- a positive
    # dimensionless number in a column declared 1/um, where the real value is signed.
    # `from_values` is the class's own inverse of `get_data`, so prefer it wherever it
    # exists, and require an exact width match on the generic path so the next family to
    # grow a derived column fails loudly instead of silently re-labelling itself.
    for family in OPTIONAL_FAMILIES:
        block = family_values.get(family.attribute)
        if not block:
            continue
        results_cls = family.results_cls
        from_values = getattr(results_cls, "from_values", None)
        if from_values is not None:
            setattr(result, family.attribute, from_values(block))
            continue
        fields = list(results_cls.__dataclass_fields__)
        if len(fields) != len(block):
            raise ValueError(
                f"{results_cls.__name__} writes {len(block)} CSV columns but has "
                f"{len(fields)} fields, so reading them back by position would shift "
                f"every column after the mismatch. Give the class a `from_values` "
                f"classmethod naming its CSV order, as MeshResults does."
            )
        setattr(result, family.attribute,
                results_cls(**dict(zip(fields, block))))
    return result
