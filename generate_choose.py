import threading
import queue
import os
from PIL import Image, ImageTk
import tkinter as tk
from typing import List
import torch

import util
from util import VariantUtil


class GenerationManager(threading.Thread):
    def __init__(self, args, callback):
        super().__init__(daemon=True)
        self.callback = callback        # called as callback(branch, variants)
        self.tasks = queue.Queue()
        self.cache = {}                 # keys: ('ui',None) or ('spec',branch)
        self.cancel_except = None

        self.processor = VariantUtil(args.good_text, args.bad_text, args.clip_positive_scale, args.clip_negative_scale,
                                     args.opt_repeats, args.guidance_scale, args.inference_steps, height=args.height,
                                     width=args.width)

        self.start()

    def generate(self, branch: int, image: Image.Image) -> List[Image.Image]:
        # callback allows interruption. only relevant for speculative generation
        def callback(*args, **kwargs):
            if (branch is not None and   # speculative mode
                    self.cancel_except is not None and
                    branch != self.cancel_except):
                print("Interrupting speculative generation on branch", branch)
                self.processor.sd_pipe.interrupt = True

        print("Starting generation for branch =", branch)
        return self.processor.variants(image, args.batch_size, callback)

    def has_tasks(self):
        return self.tasks.qsize() > 0

    def run(self):
        torch.set_grad_enabled(False)

        while True:
            if self.tasks.qsize() == 0: continue
            task = self.tasks.get()
            if task is None:
                break
            if task == "reset":
                self.cancel_except = None
                continue

            branch, image = task

            # When cancel_except is set, we cancel all other jobs until "reset" is reached.
            # (so we skip to the correct branch, if not started yet, or until "reset" if already done)
            if self.cancel_except is not None:
                if branch != self.cancel_except:
                    print("Skipping branch", branch)
                    continue

            variants = self.generate(branch, image)

            # Run was interrupted (it was speculative and ended up being unnecessary)
            if self.processor.sd_pipe.interrupt:
                self.processor.sd_pipe.interrupt = False
            else:
                # only cache if we completed an un‐interrupted run
                key = ('ui', None) if branch is None else ('spec', branch)
                self.cache[key] = variants

                # Display immediately if:
                #  1) Non-speculative (branch is None)
                #  2) Speculative and became concrete during generation (branch = cancel_except)
                if branch is None or (self.cancel_except is not None and branch == self.cancel_except):
                    print(branch, "ready for immediate display")
                    self.callback(variants)

    def add_task(self, branch: int, image: Image.Image):
        """
        branch=None => to be displayed immediately
        branch=int => speculative with given index
        """
        self.tasks.put((branch, image))

    def cancel_others(self, chosen_branch: int):
        if self.has_tasks():
            self.cancel_except = chosen_branch
            self.tasks.put("reset")  # after this, turn off cancel mode

    def shutdown(self):
        self.tasks.put(None)


class VariantSelectorApp(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.title("Variant Generator")
        self.manager = GenerationManager(args, self.display_variants)
        self.frames = []
        self.img_refs = []

        self.out_dir = args.out
        self.ctr = util.get_ctr(args)
        print(f"Saving to {util.output_path(self.out_dir, self.ctr)} and above...")

        self.speculative = args.speculative

        # container frame
        self.row = tk.Frame(self)
        self.row.pack(padx=10, pady=10)

        # Start first generation round:
        initial_image = util.get_initial_image(args)
        self.manager.add_task(None, initial_image)

    def clear_display(self):
        for f in self.frames:
            f.destroy()
        self.frames.clear()
        self.img_refs.clear()

    def display_variants(self, variants: List[Image.Image]):
        self.clear_display()
        self.manager.cache.clear()

        for idx, img in enumerate(variants):
            frame = tk.Frame(self.row, bd=2, relief='ridge')
            frame.pack(side='left', padx=5)
            self.frames.append(frame)

            # resize to thumbnail
            thumb = img.copy().resize((200, 200))
            photo = ImageTk.PhotoImage(thumb)
            self.img_refs.append(photo)

            btn = tk.Button(frame, image=photo, command=lambda i=idx, im=img: self.on_select(i, im))
            btn.pack()

        # queue speculative generations
        if self.speculative:
            for idx, img in enumerate(variants):
                self.manager.add_task(idx, img)

    def on_select(self, idx, selected_image):
        """
        User picked an image. Save it, cancel speculative generation, and
        either show the already‐computed next images or start generation.
        """
        # Save to disk
        selected_image.save(util.output_path(self.out_dir, self.ctr))
        self.ctr += 1

        if self.speculative:
            # Kill unnecessary speculative tasks
            self.manager.cancel_others(idx)

            # Load generation if it's been done already
            # (if not, it'll run next due to cancel_others, and the then display_variants will be called)
            key = ('spec', idx)
            if key in self.manager.cache:
                # generated already
                next_variants = self.manager.cache[key]
                self.display_variants(next_variants)
            else:
                self.clear_display()
        else:
            # No speculative generation: just start next gen and wait
            self.manager.add_task(None, selected_image)
            self.clear_display()

    def on_close(self):
        self.manager.shutdown()
        self.destroy()


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = util.shared_args()
    parser.add_argument("-o", "--out", type=str, default="selected/0",
                        help="Output folder. Will be created if it doesn't exist.")
    parser.add_argument("-b", "--batch-size", type=int, default=2,
                        help="Batch size (i.e. number of images you get to choose from at each iteration).")
    parser.add_argument("--speculative", action="store_true", help="Speculative generation.")
    args = parser.parse_args()
    args.good_text = args.good_text.split("|")
    args.bad_text = args.bad_text.split("|")
    print(f"{len(args.good_text)} good texts, {len(args.bad_text)} bad texts")

    app = VariantSelectorApp(args)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
