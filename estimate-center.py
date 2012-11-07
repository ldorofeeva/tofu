import argparse
import numpy as np
from gi.repository import Ufo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-directory', metavar='PATH', type=str, default='.',
                        help="Location with sinograms")
    parser.add_argument('-n', '--num-sinograms', metavar='N', type=int, default=None,
                        help="Number of sinograms to estimate")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--angle-step', metavar='F', type=float,
                       help="Angle step between projections")
    group.add_argument('-p', '--num-projections', metavar='N', type=int,
                       help="Number of projections")

    args = parser.parse_args()

    # create nodes
    pm = Ufo.PluginManager()
    sino_reader = pm.get_filter('reader')
    cor = pm.get_filter('centerofrotation')

    # configure nodes
    sino_reader.set_properties(path=args.input_directory)

    if args.num_sinograms:
        sino_reader.set_properties(count=args.num_sinograms)

    angle_step = args.angle_step if args.angle_step else np.pi / args.num_projections
    cor.set_properties(angle_step=angle_step)

    centers = []

    def print_center(cor, prop):
        center = cor.props.center
        print 'Calculated center: %f' % center
        centers.append(center)

    cor.connect('notify::center', print_center)

    g = Ufo.Graph()
    g.connect_filters(sino_reader, cor)

    s = Ufo.Scheduler()
    s.run(g)

    print 'Mean center: %f' % np.mean(centers)
