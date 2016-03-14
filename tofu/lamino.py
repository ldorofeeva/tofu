"""Laminographic reconstruction."""
import logging
import numpy as np
from multiprocessing import Queue, Process
from tofu.util import determine_shape, get_filenames


LOG = logging.getLogger(__name__)


def lamino(params):
    """Laminographic reconstruction utilizing all GPUs."""
    LOG.info('Z parameter: {}'.format(params.z_parameter))
    if not params.overall_angle:
        params.overall_angle = 360.
        LOG.info('Overall angle not specified, using 360 deg')
    if not params.angle:
        num_files = len(get_filenames(params.input))
        if not num_files:
            raise RuntimeError("No files found in `{}'".format(params.input))
        params.angle = params.overall_angle / num_files * params.step
        LOG.info('Angle not specified, calculating from ' +
                 '{} projections and step {}: {} deg'.format(num_files, params.step,
                                                             params.angle))
    if not (params.width and params.height):
        proj_width, proj_height = determine_shape(params)
        if not proj_width:
            raise RuntimeError("Could not determine width from the input")
    if not params.number:
        params.number = int(np.round(np.abs(params.overall_angle / params.angle)))
    if not params.width:
        params.width = proj_width
    if not params.height:
        params.height = proj_height - params.y
    if params.dry_run:
        LOG.info('Dummy data W x H x N: {} x {} x {}'.format(params.width,
                                                             params.height,
                                                             params.number))

    # For now we need to make a workaround for the memory leak, which means we need to execute the
    # passes in separate processes to clean up the low level code. For that we also need to call the
    # region-splitting in a separate function.
    # TODO: Simplify after the memory leak fix!
    queue = Queue()
    proc = Process(target=_create_runs, args=(params, queue,))
    proc.start()
    proc.join()
    x_region, y_region, regions, num_gpus = queue.get()

    for i in range(0, len(regions), num_gpus):
        z_subregion = regions[i:min(i + num_gpus, len(regions))]
        LOG.info('Computing slices {}..{}'.format(z_subregion[0][0], z_subregion[-1][1]))
        proc = Process(target=_run, args=(params, x_region, y_region, z_subregion, i / num_gpus))
        proc.start()
        proc.join()


def _create_runs(params, queue):
    """Workaround function to get the number of gpus and compute regions. gi.repository must always
    be called in a separate process, otherwise the resources return None gpus.
    """
    #TODO: remove the whole function after memory leak fix!
    from gi.repository import Ufo

    scheduler = Ufo.FixedScheduler()
    gpus = scheduler.get_resources().get_gpu_nodes()
    num_gpus = len(gpus)
    x_region, y_region, regions = _split_regions(params, gpus)
    LOG.info('Using {} GPUs in {} passes'.format(min(len(regions), num_gpus), len(regions)))

    queue.put((x_region, y_region, regions, num_gpus))


def _run(params, x_region, y_region, regions, index):
    """Execute one pass on all possible GPUs with slice ranges given by *regions*."""
    from gi.repository import Ufo
    from tofu.flatcorrect import create_pipeline as create_flat_pipeline
    from tofu.util import set_node_props

    pm = Ufo.PluginManager()
    graph = Ufo.TaskGraph()
    scheduler = Ufo.FixedScheduler()
    gpus = scheduler.get_resources().get_gpu_nodes()
    num_gpus = len(gpus)

    broadcast = Ufo.CopyTask()
    if params.dry_run:
        source = pm.get_task('dummy-data')
        source.props.number = params.number
        source.props.width = params.width
        source.props.height = params.height
    elif params.darks and params.flats:
        source = create_flat_pipeline(params, graph)
    else:
        source = pm.get_task('read')
        source.props.path = params.input
        set_node_props(source, params)
    graph.connect_nodes(source, broadcast)

    for i, region in enumerate(regions):
        subindex = index * num_gpus + i
        pad = _setup_graph(pm, graph, subindex, gpus[i], x_region, y_region, region, params)
        graph.connect_nodes(broadcast, pad)

    scheduler.run(graph)
    duration = scheduler.props.time
    LOG.info('Execution time: {} s'.format(duration))

    return duration


def _setup_graph(pm, graph, index, gpu, x_region, y_region, region, params):
    from tofu.reco import setup_padding
    pad = pm.get_task('pad')
    crop = pm.get_task('crop')
    fft = pm.get_task('fft')
    ifft = pm.get_task('ifft')
    fltr = pm.get_task('filter')
    backproject = pm.get_task('anka-backproject')
    slicer = pm.get_task('slice')
    if params.dry_run:
        writer = pm.get_task('null')
        writer.props.force_download = True
    else:
        writer = pm.get_task('write')
        writer.props.filename = '{}-{:>03}-%04i.tif'.format(params.output, index)

    # parameters
    setup_padding(pad, crop, params.width, params.height)
    backproject.props.num_projections = params.number
    backproject.props.overall_angle = np.deg2rad(params.overall_angle)
    backproject.props.lamino_angle = np.deg2rad(params.lamino_angle)
    backproject.props.roll_angle = np.deg2rad(params.roll_angle)
    backproject.props.x_region = x_region
    backproject.props.y_region = y_region
    backproject.props.z = params.z
    if params.z_parameter in ['lamino-angle', 'roll-angle']:
        region = [np.deg2rad(reg) for reg in region]
    backproject.props.region = region
    backproject.props.parameter = params.z_parameter
    backproject.props.center = params.axis
    fft.props.dimensions = 1
    ifft.props.dimensions = 1
    fltr.props.scale = np.sin(backproject.props.lamino_angle)

    pad.set_proc_node(gpu)
    crop.set_proc_node(gpu)
    fft.set_proc_node(gpu)
    ifft.set_proc_node(gpu)
    fltr.set_proc_node(gpu)
    backproject.set_proc_node(gpu)

    graph.connect_nodes(pad, fft)
    graph.connect_nodes(fft, fltr)
    graph.connect_nodes(fltr, ifft)
    graph.connect_nodes(ifft, crop)
    graph.connect_nodes(crop, backproject)
    graph.connect_nodes(backproject, slicer)
    graph.connect_nodes(slicer, writer)

    return pad


def _split_regions(params, gpus):
    """Split processing between *gpus* by specifying the number of slices processed per GPU."""
    if params.x_region[1] == -1:
        x_region = _make_region(params.width)
    else:
        x_region = params.x_region
    if params.y_region[1] == -1:
        y_region = _make_region(params.width)
    else:
        y_region = params.y_region
    if params.region[1] == -1:
        region = _make_region(params.height)
    else:
        region = params.region
    LOG.info('X region: {}'.format(x_region))
    LOG.info('Y region: {}'.format(y_region))
    LOG.info('Parameter region: {}'.format(region))

    z_start, z_stop, z_step = region
    y_start, y_stop, y_step = y_region
    x_start, x_stop, x_step = x_region

    num_slices = len(np.arange(z_start, z_stop, z_step))
    slice_height = len(range(y_start, y_stop, y_step))
    slice_width = len(range(x_start, x_stop, x_step))

    if params.slices_per_device:
        num_slices_per_gpu = params.slices_per_device
    else:
        num_slices_per_gpu = _compute_num_slices(gpus, slice_width, slice_height)
    if num_slices_per_gpu > num_slices:
        num_slices_per_gpu = num_slices
    LOG.info('Using {} slices per GPU'.format(num_slices_per_gpu))

    z_starts = np.arange(z_start, z_stop, z_step * num_slices_per_gpu)
    regions = []
    for start in z_starts:
        regions.append((start, min(z_stop, start + z_step * num_slices_per_gpu), z_step))

    return x_region, y_region, regions


def _make_region(n):
    return (-(n / 2), n / 2 + n % 2, 1)


def _compute_num_slices(gpus, width, height):
    """Determine number of slices which can be calculated per-device based on *gpus*, slice *width*
    and *height*.
    """
    from gi.repository import Ufo

    # Make sure the double buffering works with room for intermediate steps
    # TODO: compute this precisely
    safety_coeff = 3.
    # Use the weakest one, if heterogenous systems emerge, measure the performance and
    # reconsider
    memories = [gpu.get_info(Ufo.GpuNodeInfo.GLOBAL_MEM_SIZE) for gpu in gpus]
    i = np.argmin(memories)
    max_allocatable = gpus[i].get_info(Ufo.GpuNodeInfo.MAX_MEM_ALLOC_SIZE)
    if max_allocatable * safety_coeff <= memories[i]:
        # Don't waste resources
        max_memory = max_allocatable
    else:
        max_memory = memories[i] / safety_coeff

    if max_memory > 2 ** 32:
        # Current NVIDIA implementation allows only 4 GB
        max_memory = 2 ** 32
    max_memory /= safety_coeff
    num_slices = int(np.floor(max_memory / (width * height * 4)))
    LOG.info('GPU memory used per GPU: {:.2f} GB'.format(max_memory / 2. ** 30))

    return num_slices