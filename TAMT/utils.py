import numpy as np
import os
import glob
import argparse
import network.resnet as resnet
import network.VideoMAE as VideoMAE
import torch
import random
from weight_loaders import weight_loader_fn_dict

model_dict = dict(
    ResNet10=resnet.ResNet10,
    ResNet12=resnet.ResNet12,
    ResNet18=resnet.ResNet18,
    ResNet34=resnet.ResNet34,
    ResNet34s=resnet.ResNet34s,
    ResNet50=resnet.ResNet50,
    ResNet101=resnet.ResNet101,
    VideoMAEB=VideoMAE.vit_base_patch16_224,
    VideoMAES=VideoMAE.vit_small_patch16_112,
    VideoMAES2=VideoMAE.vit_small_patch16_224,
    VideoMAEGiant=VideoMAE.vit_giant_patch14_224
    )


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    print(best_file)
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


# def set_gpu(args):
#     if args.gpu == '-1':
#         gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
#     else:
#         gpu_list = [int(x) for x in args.gpu.split(',')]
#         print('use gpu:', gpu_list)
#         os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     return gpu_list.__len__()

def set_gpu(params):
    if params.gpu == -1:  # chạy CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)
        gpu_list = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x]
        return len(gpu_list)
    

def load_model(model, ckpt_path):
    import torch
    import os

    if not os.path.isfile(ckpt_path):
        print(f"⚠️ Checkpoint {ckpt_path} không tồn tại, bỏ qua pretrain!")
        return model

    # Try loading in several ways to handle PyTorch 2.6+ weights_only behavior
    ckpt = None
    try:
        # default (may raise WeightsUnpicklingError on torch>=2.6 if file contains non-weight globals)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    except Exception as e:
        print("⚠️ torch.load mặc định thất bại:", e)
        try:
            # try allowing execution (unsafe) for trusted checkpoints
            print("Thử lại với weights_only=False...")
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        except Exception as e2:
            print("⚠️ torch.load with weights_only=False thất bại:", e2)
            try:
                # add safe global for numpy scalar then retry (useful for some checkpoints)
                print("Thêm numpy.core.multiarray.scalar vào safe globals và thử lại...")
                try:
                    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                except Exception:
                    # older/newer torch API differences: ignore if not available
                    pass
                ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
            except Exception as e3:
                print("❌ Không thể load checkpoint bằng các phương thức thử: ", e3)
                raise

    # Kiểm tra các key phổ biến
    if isinstance(ckpt, dict):
        if 'state' in ckpt:
            state_dict = ckpt['state']
        elif 'module' in ckpt:
            state_dict = ckpt['module']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt  # Nếu ckpt là state_dict hoặc tương tự
    else:
        state_dict = ckpt

    # Lấy state_dict của model hiện tại
    model_dict = model.state_dict()

    # Giữ lại các key trùng khớp
    matched_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # Cập nhật weight
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"✅ Loaded pretrained model từ {ckpt_path} với {len(matched_dict)} layers khớp")
    return model


    # load pretrained model of SSL
    if dir =='/hd1/wyl/model/112vit-s-woSupervisecheckpoint-399.pth':
        model_dict = model.state_dict()
        file_dict = torch.load(dir)['model']
        model.feature.load_state_dict(file_dict, strict=False)
        return model

    # load pretrained model of SL
    if dir == '/hd1/wyl/model/112112vit-s-140epoch.pt' or dir =='/hd1/wyl/model/vit-s-120epoch.pt' or dir =='/hd1/wyl/model/112112vit-s-120epoch.pt':
        model_dict = model.state_dict()
        file_dict = torch.load(dir)['module']
        model.feature.load_state_dict(file_dict, strict=False)
        return model

    # load finetuned model
    model_dict = model.state_dict()
    file_dict = torch.load(dir)['state']
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_pretrained_model(model, ckpt_path):
    import torch
    if not os.path.isfile(ckpt_path):
        print(f"⚠️ Checkpoint {ckpt_path} không tồn tại, bỏ qua pretrain.")
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Lấy state_dict phù hợp
    if "model" in ckpt:
        ckpt = ckpt["model"]
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "module" in ckpt:
        ckpt = ckpt["module"]

    # Remap key nếu thiếu prefix "feature."
    def remap_checkpoint_keys(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("feature."):
                new_k = "feature." + k
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict

    ckpt = remap_checkpoint_keys(ckpt)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("⚠️ Missing keys:", missing)
    print("⚠️ Unexpected keys:", unexpected)
    return model
