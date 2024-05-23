from glob import glob
import argparse


from apps.utils import load_model, RMBG
from apps.mv_models import GenMVImage

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./eval_data", help="Path to the input data",)
    parser.add_argument("--output", type=str, default="./eval_outputs", help="Path to the inference results",)
    parser.add_argument("--model", type=str, default="", help="Path to the image-to-shape diffusion model",)
    parser.add_argument("--mv_model", type=str, default="crm", help="Path to the multi-view images model",)
    parser.add_argument("--scheduler_name", type=str, default="crm", help="Path to the multi-view images model",)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    
    # load the multi-view images model
    if args.model == "":
        ckpt_path = hf_hub_download(repo_id="wyysf/CraftsMan", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6/model.ckpt", repo_type="model")
        config_path = hf_hub_download(repo_id="wyysf/CraftsMan", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6/config.yaml", repo_type="model")
    else:
        ckpt_path = f"{args.model}/model.ckpt"
        config_path = f"{args.model}/config.yaml"    
    model = load_model(ckpt_path, config_path, device)
    
    # read the input images
    if os.path.isdir(args.input):
        image_files = glob(os.path.join(args.input, "*.png"))
    else:
        image_files = [args.input]
    
    for image_file in image_files:
        pass
    
    rmbg = RMBG(device)
    gen_mvimg = GenMVImage(device)

    for image_file in image_files:
        image = Image.open(image_file)
        import pdb; pdb.set_trace()
        # image = rmbg.rmbg_sam(image, foreground_ratio=0.5)
        # mv_img = gen_mvimg.gen_image_from_crm(image)
        # for i, img in enumerate(mv_img):
        #     img.save(f"{args.output}/mv_img_{i}.png")
        # print(f"Multi-view images saved to {args.output}")