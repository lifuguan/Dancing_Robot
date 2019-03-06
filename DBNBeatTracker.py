#!C:\Python27amd64\python.exe
# encoding: utf-8
"""
DBNBeatTracker beat tracking algorithm.

"""

from __future__ import absolute_import, division, print_function

import argparse

from madmom.audio import SignalProcessor
from madmom.features import (ActivationsProcessor, DBNBeatTrackingProcessor,
                             RNNBeatProcessor)
from madmom.io import write_beats
from madmom.ml.nn import NeuralNetworkEnsemble
from madmom.processors import IOProcessor, io_arguments


def main():
    """DBNBeatTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The DBNBeatTracker program detects all beats in an audio file according to
    the method described in:

    "A Multi-Model Approach to Beat Tracking Considering Heterogeneous Music
     Styles"
    Sebastian Böck, Florian Krebs and Gerhard Widmer.
    Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.

    It does not use the multi-model (Section 2.2.) and selection stage (Section
    2.3), i.e. this version corresponds to the pure DBN version of the
    algorithm for which results are given in Table 2.

    Instead of the originally proposed state space and transition model for the
    DBN, the following is used:

    "An Efficient State Space Model for Joint Tempo and Meter Tracking"
    Florian Krebs, Sebastian Böck and Gerhard Widmer.
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.

    This program can be run in 'single' file mode to process a single audio
    file and write the detected beats to STDOUT or the given output file.

      $ DBNBeatTracker single INFILE [-o OUTFILE]

    If multiple audio files should be processed, the program can also be run
    in 'batch' mode to save the detected beats to files with the given suffix.

      $ DBNBeatTracker batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] FILES

    If no output directory is given, the program writes the files with the
    detected beats to the same location as the audio files.

    The 'pickle' mode can be used to store the used parameters to be able to
    exactly reproduce experiments.

    ''')
    # version
    p.add_argument('--version', action='version',
                   version='DBNBeatTracker.2016')
    # input/output options
    io_arguments(p, output_suffix='.beats.txt', online=True)
    ActivationsProcessor.add_arguments(p)
    # signal processing arguments
    SignalProcessor.add_arguments(p, norm=False, gain=0)
    # peak picking arguments
    DBNBeatTrackingProcessor.add_arguments(p)
    NeuralNetworkEnsemble.add_arguments(p, nn_files=None)

    # parse arguments
    args = p.parse_args()

    # set immutable arguments
    args.fps = 100
    
    # print arguments
    if args.verbose:
        print(args)
    print("The following mesxxsage shows the args :\n", args,  "\n\n\nshows the vars(args) message:\n", vars(args))
    ''' 
    The args's message:

    Namespace(correct=True, fps=100, func=<function process_online at 0x00000000055E9AC8>, 
    gain=0, infile=None, load=False, max_bpm=215.0, min_bpm=55.0, nn_files=None, norm=False,
     num_frames=1, num_tempi=None, num_threads=1, observation_lambda=16, online=True, 
     origin='stream', outfile=<open file '<stdout>', mode 'w' at 0x0000000002DFB0C0>,
      save=False, sep=None, threshold=0, transition_lambda=100, verbose=None)
    '''

    # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # use a RNN to predict the beats
        in_processor = RNNBeatProcessor(**vars(args))

    # output processor
    if args.save:
        # save the RNN beat activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
       
        # track the beats with a DBN and output them
        beat_processor = DBNBeatTrackingProcessor(**vars(args))
        out_processor = [beat_processor, write_beats]
        print("ok")

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
