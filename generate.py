import torch

import util
from util import VariantUtil


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = util.shared_args()
    parser.add_argument("-o", "--out", type=str, default="out/0",
                        help="Output folder. Will be created if it doesn't exist.")
    parser.add_argument("-n", "--num-images", type=int, required=True, help="Number of images to generate.")
    args = parser.parse_args()
    args.good_text = args.good_text.split("|")
    args.bad_text = args.bad_text.split("|")
    print(f"{len(args.good_text)} good texts, {len(args.bad_text)} bad texts")

    image = util.get_initial_image(args)
    ctr = util.get_ctr(args)

    processor = VariantUtil(args.good_text, args.bad_text, args.clip_positive_scale, args.clip_negative_scale,
                            args.opt_repeats, args.guidance_scale, args.inference_steps, height=args.height,
                            width=args.width)

    for i in range(args.num_images):
        print(f"Generating image {i+1} / {args.num_images}")
        save_path = util.output_path(args.out, ctr)
        image = processor.variants(image, 1)[0]
        image.save(save_path)
        ctr += 1
