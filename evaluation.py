import argparse
import torch
from PIL import Image
from sav_dataset.utils.endo_sav_benchmark import benchmark

def evaluate(args):
    benchmark(
        [args.gt_root],
        [args.output_mask_dir],
        args.strict,
        args.num_processes,
        verbose=not args.quiet,
        skip_first_and_last=not args.do_not_skip_first_and_last_frame,
        epoch=0,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--gt_root",
        required=True,
        help="Path to the GT folder. For SA-V, it's sav_val/Annotations_6fps or sav_test/Annotations_6fps",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Quietly run evaluation without printing the information out",
        action="store_true",
    )
    parser.add_argument(
        "--do_not_skip_first_and_last_frame",
        help="In SA-V val and test, we skip the first and the last annotated frames in evaluation. "
             "Set this to true for evaluation on settings that doen't skip first and last frames",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--strict",
        help="Make sure every video in the gt_root folder has a corresponding video in the prediction",
        action="store_true",
    )
    parser.add_argument(
        "-n", "--num_processes", default=16, type=int, help="Number of concurrent processes"
    )

    args = parser.parse_args()
    evaluate(args)
