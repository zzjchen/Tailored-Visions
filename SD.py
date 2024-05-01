import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from diffusers import ModelMixin

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

@torch.no_grad()
def text2img(pipe,prompt,batch_size=4,save=False,save_path=None,**kwargs):
    image = pipe(prompt,num_images_per_prompt=batch_size,**kwargs).images
    if save and save_path is not None:
        grid=image_grid(image,1,batch_size)
        grid.save(save_path)
    return image

if __name__=="__main__":
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    prompt="an astronaut riding a horse"
    text2img(pipe,prompt,save=True,save_path="astronaut_rides_horse.png")

    #image.save("astronaut_rides_horse.png")