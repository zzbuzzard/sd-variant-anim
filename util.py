import torch
import numpy as np
import os
from os.path import join
from PIL import Image
from typing import List, Tuple
import av
import torchvision.transforms.functional as trf
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
from diffusers.schedulers import DPMSolverMultistepScheduler
import gc
import argparse

from pipeline import StableDiffusionImageVariationPipeline


def shared_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input-image", type=str, default=None, help="Initial image as input to the first variant generation.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite existing files in output directory. If not set (default), "
                             "generation will continue from the last image, extending the animation.")
    parser.add_argument("-gs", "--guidance-scale", type=float, default=4, help="Classifier-free guidance scale.")
    parser.add_argument("-s", "--inference-steps", type=int, default=20, help="Number of inference steps (DPM++ scheduler)")
    parser.add_argument("--good-text", type=str,
                        default="realistic HD photograph|professional photograph|3D realistic materials|complex image|intricate detailed",
                        help="List of text prompts to move TOWARDS with CLIP guidance. Separated by '|' character.")
    parser.add_argument("--bad-text", type=str,
                        default="simple texture|abstract image|repeating pattern|abstract shapes|cartoon|human face|oversaturated|blurry photograph|smooth low contrast",
                        help="List of text prompts to move AWAY from with CLIP guidance. Separated by '|' character.")
    parser.add_argument("-cp", "--clip-positive-scale", type=float, default=30, help="CLIP guidance scale (for 'good' text).")
    parser.add_argument("-cn", "--clip-negative-scale", type=float, default=30, help="CLIP guidance scale (for 'bad' text).")
    parser.add_argument("--opt-repeats", type=int, default=10,
                        help="Number of steps to optimise image latent with CLIP guidance for.")
    return parser


def output_path(out: str, ctr: int) -> str:
    return join(out, f"{ctr:07d}.jpg")


def get_ctr(args):
    os.makedirs(args.out, exist_ok=True)
    if args.overwrite:
        return 0
    ctr = 0
    while os.path.isfile(output_path(args.out, ctr)):
        ctr += 1
    return ctr


def get_initial_image(args) -> Image.Image:
    ctr = get_ctr(args)
    if ctr == 0:
        assert args.input_image is not None, f"Must specify `--input-image` if starting a new animation."
        assert os.path.isfile(args.input_image), f"Image at '{args.input_image}' not found."
        image = Image.open(args.input_image)
    else:
        # Load last image in directory to continue from
        image = Image.open(output_path(args.out, ctr - 1))

    return image.convert("RGB").resize((512, 512))


# Path w files in format 000000.png 000001.png ...
def make_video_ffmpeg(path, fps, gif=False, out_path=None):
    if out_path is None:
        ext = "gif" if gif else "mp4"
        out_path = os.path.join(path, f"vid.{ext}")
    if gif:
        os.system(f"ffmpeg -r {fps} -i {path}/%07d.jpg {out_path}")
    else:
        os.system(f"ffmpeg -r {fps} -i {path}/%07d.jpg -vcodec libx264 -y {out_path} -qp 0")


# Uses PyAV and writes as MP4
class AvVideoWriter:
    def __init__(self, height, width, out_path, fps, crf=20):
        self.file = open(out_path, "wb")
        self.output = av.open(self.file, 'w', format="mp4")
        self.stream = self.output.add_stream('h264', str(fps))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv420p'  # not yuv444p
        self.stream.options = {'crf': str(crf)}  # higher value = more compression

    def write(self, img):
        if isinstance(img, Image.Image):
            arr = np.array(img)
        elif isinstance(img, np.ndarray):
            arr = img
        elif isinstance(img, torch.Tensor):
            arr = np.array(trf.to_pil_image(img))

        frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
        packet = self.stream.encode(frame)
        self.output.mux(packet)

    def close(self):
        packet = self.stream.encode(None)
        self.output.mux(packet)
        self.output.close()
        self.file.close()


def load_model(device="cuda:0", dtype=torch.float16) -> Tuple[StableDiffusionImageVariationPipeline, CLIPModel, CLIPProcessor]:
    clip_device = device
    clip_dtype = dtype

    ### Load CLIP model (and remove text emb)
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    del clip.vision_model
    clip = clip.to(device=device, dtype=torch.float16)

    ### Load SD model
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
        torch_dtype=dtype,
        safety_checker=None,
    )
    clip.vision_model = sd_pipe.image_encoder.vision_model
    sd_pipe = sd_pipe.to(device)

    ### If on different devices, the CLIP image encoder should be on the CLIP model's device
    if clip_device != device or dtype != clip_dtype:
        sd_pipe.image_encoder.to(device=clip_device, dtype=clip_dtype)
    del sd_pipe.safety_checker
    sd_pipe.safety_checker = None
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)

    gc.collect()
    torch.cuda.empty_cache()

    return sd_pipe, clip, clip_preprocess


tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])


class VariantUtil:
    def __init__(self, good_text: List[str], bad_text: List[str], clip_pos_mul: float, clip_neg_mul: float,
                 step_repeats: int, guidance_scale: float, num_inference_steps: int = 10, device="cuda:0",
                 dtype=torch.float16):
        self.sd_pipe, self.clip, self.clip_preprocess = load_model(device, dtype)

        self.good_text = good_text
        self.bad_text = bad_text

        clip_device = device
        clip_dtype = dtype

        all_text = self.good_text + self.bad_text
        tok = self.clip_preprocess.tokenizer(all_text, return_tensors="pt", padding=True)
        input_ids = tok.input_ids.to(clip_device)
        attention_mask = tok.attention_mask.to(clip_device).to(clip_dtype)
        text_embeds = self.clip.text_projection(self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)[1])
        self.text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # No longer needed
        del self.clip.text_model
        del self.clip.text_projection

        self.clip_pos_mul = clip_pos_mul
        self.clip_neg_mul = clip_neg_mul
        self.step_repeats = step_repeats

        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def shift_emb(self, image_emb: torch.Tensor) -> torch.Tensor:
        """
        Numerically optimise image_emb to increase its CLIP score (maximise good logits, minimise bad logits).
        This could be likely be done numerically but this method is more flexible (if you want to incorporate other
        losses).
        """
        start_emb = image_emb.clone()

        for _ in range(self.step_repeats):
            image_emb.grad = None
            with torch.enable_grad():
                image_emb.requires_grad_(True)

                # emb = clip.visual_projection(image_emb)  # its already been projected
                emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
                # logit_scale = self.clip.logit_scale.exp()
                logits_per_text = torch.matmul(self.text_embeds, emb.t())  # n x 1

                maximise = torch.mean(logits_per_text[:len(self.good_text)])  # +ve
                minimise = torch.mean(logits_per_text[len(self.good_text):])  # -ve

                # we want positive similarity to be HIGH, negative similarity to be LOW
                loss = minimise * self.clip_neg_mul - maximise * self.clip_pos_mul  # minimise = maximise pos, minimise neg
                loss.backward()

            image_emb = image_emb - image_emb.grad

        print("CLIP guidance shift magnitude:", torch.linalg.norm(image_emb - start_emb).item())
        return image_emb

    def variants(self, image: Image.Image, count: int = 1, callback=None):
        preprocessed = tform(image).to("cuda").unsqueeze(0)
        image_emb = self.sd_pipe.encode_image(preprocessed)
        image_emb = self.shift_emb(image_emb)

        images = self.sd_pipe(
            image_embeddings=image_emb,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=count,
            callback=callback
        )["images"]

        return images
