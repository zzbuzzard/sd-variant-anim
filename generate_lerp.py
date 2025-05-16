import torch
import os
from os.path import join

import util
from util import VariantUtil


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = util.shared_args()
    parser.add_argument("-o", "--out", type=str, default="out/0",
                        help="Output folder. Will be created if it doesn't exist.")
    parser.add_argument("-n", "--num-images", type=int, required=True, help="Number of images to generate.")
    parser.add_argument("-f", "--fps", type=int, default=20, help="Frames per second.")
    parser.add_argument("-in", "--intermediate", type=int, default=8, help="Number of intermediate frames between each variant.")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("-p", "--noise-shift-prop", type=float, default=0.4,
                        help="Speed at which noise changes (0 is no change, 1 means each variant has independent noise). Lower values lead to smoother transitions")
    args = parser.parse_args()
    args.good_text = args.good_text.split("|")
    args.bad_text = args.bad_text.split("|")
    print(f"{len(args.good_text)} good texts, {len(args.bad_text)} bad texts")
    args.overwrite = True

    os.makedirs(args.out, exist_ok=True)
    out_file = join(args.out, "video.mp4")
    writer = util.AvVideoWriter(args.height, args.width, out_file, args.fps)

    image = util.get_initial_image(args)
    ctr = 0

    processor = VariantUtil(args.good_text, args.bad_text, args.clip_positive_scale, args.clip_negative_scale,
                            args.opt_repeats, args.guidance_scale, args.inference_steps, height=args.height,
                            width=args.width)

    print("Generating initial image...")
    image_emb = processor.get_image_emb(image)
    noise = torch.randn(processor.latent_shape, dtype=image_emb.dtype, device=image_emb.device)
    image = processor.variants(image, count=1, noise=noise)[0]
    image.save(util.output_path(args.out, ctr))
    ctr += 1

    writer.write(image)

    for i in range(args.num_images):

        print(f"Generating image batch {i+1} / {args.num_images}")

        all_images, image_emb, noise = processor.variant_lerp(image_emb, noise, image, args.intermediate, args.batch_size,
                                                              noise_shift_factor=args.noise_shift_prop)

        for img in all_images:
            img.save(util.output_path(args.out, ctr))
            ctr += 1
            writer.write(img)

        image = all_images[-1]

    writer.close()
