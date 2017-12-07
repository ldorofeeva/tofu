"""Universal projection-based reconstruction for tomographic/laminographic cone/parallel beam data
sets.
"""
import itertools
import logging
import numpy as np
from gi.repository import Ufo
from lamino import prepare_angular_arguments
from preprocess import create_preprocessing_pipeline
from util import get_reconstructed_cube_shape, get_reconstruction_regions
from tasks import get_task, get_writer


LOG = logging.getLogger(__name__)
DTYPE_CL_SIZE = {'float': 4,
                 'double': 8,
                 'half': 2,
                 'uchar': 1,
                 'ushort': 2,
                 'uint': 4}


def unireco(args):
    scheduler = Ufo.FixedScheduler()
    gpus = scheduler.get_resources().get_gpu_nodes()
    duration = 0
    for i, gpu in enumerate(gpus):
        print 'Max mem for {}: {:.2f} GB'.format(i, gpu.get_info(0) / 2 ** 30)

    prepare_angular_arguments(args)
    _convert_angles_to_rad(args)
    _set_projection_filter_scale(args)
    x_region, y_region, z_region = get_reconstruction_regions(args)
    runs = make_runs(gpus, x_region, y_region, z_region,
                     DTYPE_CL_SIZE[args.store_type],
                     slices_per_device=args.slices_per_device,
                     slice_memory_coeff=args.slice_memory_coeff,
                     data_splitting_policy=args.data_splitting_policy)
    LOG.info('Number of passes: %d', len(runs))

    for i, regions in enumerate(runs):
        for gpu, region in regions:
            LOG.debug('Pass %d: device %d, region: %s', i, gpu, region)
        duration += _run(args, x_region, y_region, regions, i)

    LOG.debug('Duration: %.2f s', duration)


def make_runs(gpus, x_region, y_region, z_region, bpp, slices_per_device=None,
              slice_memory_coeff=0.8, data_splitting_policy='one'):
    def _add_region(runs, gpu_index, current, to_process, z_start, z_step):
        z_end = z_start + current * z_step
        runs[-1].append((gpu_index, [z_start, z_end, z_step]))
        to_process -= current
        z_start = z_end

        return z_start, z_end, to_process

    z_start, z_stop, z_step = z_region
    y_start, y_stop, y_step = y_region
    x_start, x_stop, x_step = x_region
    slice_width, slice_height, num_slices = get_reconstructed_cube_shape(x_region, y_region,
                                                                         z_region)

    if slices_per_device:
        slices_per_device = [slices_per_device for i in range(len(gpus))]
    else:
        slices_per_device = get_num_slices_per_gpu(gpus, slice_width, slice_height, bpp,
                                                   slice_memory_coeff=slice_memory_coeff)

    max_slices_per_pass = sum(slices_per_device)
    if not max_slices_per_pass:
        raise RuntimeError('None of the available devices has enough memory to store any slices')
    num_full_passes = num_slices / max_slices_per_pass
    LOG.debug('Number of slices: %d', num_slices)
    LOG.debug('Slices per device %s', slices_per_device)
    LOG.debug('Maximum slices on all GPUs per pass: %d', max_slices_per_pass)
    LOG.debug('Number of passes with full workload: %d', num_slices / max_slices_per_pass)
    sorted_indices = np.argsort(slices_per_device)[-np.count_nonzero(slices_per_device):]
    runs = []
    z_start = z_region[0]
    to_process = num_slices

    # Create passes where all GPUs are fully loaded
    for j in range(num_full_passes):
        runs.append([])
        for i in sorted_indices:
            z_start, z_end, to_process = _add_region(runs, i, slices_per_device[i], to_process,
                                                     z_start, z_step)

    if to_process:
        if data_splitting_policy == 'one':
            # Fill the last pass by maximizing the workload per GPU
            runs.append([])
            for i in sorted_indices[::-1]:
                if not to_process:
                    break
                current = min(slices_per_device[i], to_process)
                z_start, z_end, to_process = _add_region(runs, i, current, to_process,
                                                         z_start, z_step)
        else:
            # Fill the last pass by maximizing the number of GPUs which will work
            num_gpus = len(sorted_indices)
            runs.append([])
            for j, i in enumerate(sorted_indices):
                # Current GPU will either process the maximum number of slices it can. If the number
                # of slices per GPU based on even division between them cannot saturate the GPU, use
                # this number. This way the work will be split evenly between the GPUs.
                current = max(min(slices_per_device[i], (to_process - 1) / (num_gpus - j) + 1), 1)
                z_start, z_end, to_process = _add_region(runs, i, current, to_process,
                                                         z_start, z_step)
                if not to_process:
                    break

    return runs


def get_num_slices_per_gpu(gpus, width, height, bpp, slice_memory_coeff=0.8):
    num_slices = []
    slice_size = width * height * bpp

    for i, gpu in enumerate(gpus):
        max_mem = gpu.get_info(Ufo.GpuNodeInfo.GLOBAL_MEM_SIZE)
        num_slices.append(int(np.floor(max_mem * slice_memory_coeff / slice_size)))

    return num_slices


def _run(args, x_region, y_region, regions, run_number):
    """Execute one pass on all possible GPUs with slice ranges given by *regions*."""
    graph = Ufo.TaskGraph()
    scheduler = Ufo.FixedScheduler()
    if hasattr(scheduler.props, 'enable_tracing'):
        LOG.debug("Use tracing: {}".format(args.enable_tracing))
        scheduler.props.enable_tracing = args.enable_tracing
    # TODO: are the gpus in the same order as the ones for which we computed the sizes?
    gpus = scheduler.get_resources().get_gpu_nodes()

    broadcast = Ufo.CopyTask()
    source = _setup_source(args, graph)
    graph.connect_nodes(source, broadcast)

    for j, gpu_and_region in enumerate(regions):
        gpu_index, region = gpu_and_region
        region_index = run_number * len(gpus) + j
        _setup_graph(args, graph, region_index, x_region, y_region, region,
                     broadcast, gpu=gpus[gpu_index])
        LOG.debug('Pass: %d, device: %d, region: %s', run_number + 1, gpu_index, region)

    scheduler.run(graph)
    duration = scheduler.props.time
    LOG.debug('Pass %d duration: %.2f s', run_number + 1, duration)

    return duration


def _setup_graph(args, graph, index, x_region, y_region, region, source, gpu=None):
    backproject = get_task('general-backproject', processing_node=gpu)

    if args.dry_run:
        sink = get_task('null', processing_node=gpu, download=True)
    else:
        sink = get_writer(args)
        sink.props.filename = '{}-{:>03}-%04i.tif'.format(args.output, index)

    backproject.props.parameter = args.z_parameter
    if args.burst:
        backproject.props.burst = args.burst
    backproject.props.z = args.z
    backproject.props.region = region
    backproject.props.x_region = x_region
    backproject.props.y_region = y_region
    backproject.props.center_x = args.center_x or [args.width / 2. + (args.width % 2) * 0.5]
    backproject.props.center_z = args.center_z or [args.height / 2. + (args.height % 2) * 0.5]
    backproject.props.source_position_x = args.source_position_x
    backproject.props.source_position_y = args.source_position_y
    backproject.props.source_position_z = args.source_position_z
    backproject.props.detector_position_x = args.detector_position_x
    backproject.props.detector_position_y = args.detector_position_y
    backproject.props.detector_position_z = args.detector_position_z
    backproject.props.detector_angle_x = args.detector_angle_x
    backproject.props.detector_angle_y = args.detector_angle_y
    backproject.props.detector_angle_z = args.detector_angle_z
    backproject.props.axis_angle_x = args.axis_angle_x
    backproject.props.axis_angle_y = args.axis_angle_y
    backproject.props.axis_angle_z = args.axis_angle_z
    backproject.props.volume_angle_x = args.volume_angle_x
    backproject.props.volume_angle_y = args.volume_angle_y
    backproject.props.volume_angle_z = args.volume_angle_z
    backproject.props.num_projections = args.number
    backproject.props.compute_type = args.compute_type
    backproject.props.result_type = args.result_type
    backproject.props.store_type = args.store_type
    backproject.props.overall_angle = args.overall_angle
    backproject.props.addressing_mode = args.unireco_padding_mode
    backproject.props.gray_map_min = args.slice_gray_map[0]
    backproject.props.gray_map_max = args.slice_gray_map[1]

    if args.only_bp:
        first = backproject
        graph.connect_nodes(source, backproject)
    else:
        first = create_preprocessing_pipeline(args, graph, source=source,
                                              processing_node=gpu,
                                              cone_beam_weight=not args.disable_cone_beam_weight)
        graph.connect_nodes(first, backproject)

    graph.connect_nodes(backproject, sink)

    return first


def _setup_source(args, graph):
    from tofu.preprocess import create_flat_correct_pipeline
    from tofu.util import set_node_props, setup_read_task
    if args.dry_run:
        source = get_task('dummy-data', number=args.number, width=args.width, height=args.height)
    elif args.darks and args.flats:
        source = create_flat_correct_pipeline(args, graph)
    else:
        source = get_task('read')
        set_node_props(source, args)
        setup_read_task(source, args.projections, args)

    return source


def _set_projection_filter_scale(args):
    is_parallel = np.all(np.isinf(args.source_position_y))
    magnification = (args.source_position_y[0] - args.detector_position_y[0]) / \
        args.source_position_y[0]

    args.projection_filter_scale = 1.
    if is_parallel:
        if np.any(np.array(args.axis_angle_x)):
            LOG.debug('Adjusting filter for parallel beam laminography')
            args.projection_filter_scale = 0.5 * np.cos(args.axis_angle_x[0])
    else:
        args.projection_filter_scale = 0.5
        args.projection_filter_scale /= magnification ** 2
        if np.all(np.array(args.axis_angle_x) == 0):
            LOG.debug('Adjusting filter for cone beam tomography')
            args.projection_filter_scale /= magnification


def _convert_angles_to_rad(args):
    names = ['detector_angle', 'axis_angle', 'volume_angle']
    coords = ['x', 'y', 'z']
    angular_z_params = map(lambda x: x[0].replace('_', '-') + '-' + x[1],
                           itertools.product(names, coords))
    args.overall_angle = np.deg2rad(args.overall_angle)
    if args.z_parameter in angular_z_params:
        LOG.debug('Converting z parameter values to radians')
        args.region = _convert_list_to_rad(args.region)

    for name in names:
        for coord in coords:
            full_name = name + '_' + coord
            values = getattr(args, full_name)
            setattr(args, full_name, _convert_list_to_rad(values))


def _convert_list_to_rad(values):
    return np.deg2rad(np.array(values)).tolist()
