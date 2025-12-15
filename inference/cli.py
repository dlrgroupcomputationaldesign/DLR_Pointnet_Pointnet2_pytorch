import argparse
from .inference import run_inference

def parse_args():
    p = argparse.ArgumentParser("Model")
    p.add_argument("--model", type=str, default="pointnet")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--num_point", type=int, default=4096)
    p.add_argument("--log_dir", type=str, required=True)
    p.add_argument("--visual", action="store_true", default=False)
    p.add_argument("--test_project", type=str, default="MorrisCollege_Pinson")
    p.add_argument("--num_votes", type=int, default=3)
    p.add_argument("--data_type", type=str, default="clustered")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--label_path", type=str, required=True)
    p.add_argument("--trained_model", type=str, required=True)
    p.add_argument("--output_csv", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    run_inference(
        model=args.model,
        batch_size=args.batch_size,
        gpu=args.gpu,
        num_point=args.num_point,
        log_dir=args.log_dir,
        visual=args.visual,
        test_project=args.test_project,
        num_votes=args.num_votes,
        data_type=args.data_type,
        data_dir=args.data_dir,
        label_path=args.label_path,
        trained_model=args.trained_model,
        output_csv=args.output_csv,
    )

if __name__ == "__main__":
    main()
