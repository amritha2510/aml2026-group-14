from torchvision import transforms


def get_train_transforms(image_size=224, preserve_aspect_ratio=False):
    resize_ops = (
        [transforms.Resize(image_size), transforms.RandomCrop(image_size)]
        if preserve_aspect_ratio
        else [transforms.Resize((image_size, image_size))]
    )
    return transforms.Compose([
        *resize_ops,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size=224, preserve_aspect_ratio=False):
    resize_ops = (
        [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
        if preserve_aspect_ratio
        else [transforms.Resize((image_size, image_size))]
    )
    return transforms.Compose([
        *resize_ops,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])