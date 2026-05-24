import timm


def build_vit_model(
    model_name="vit_tiny_patch16_224",
    pretrained=True,
    num_classes=3,
    freeze_backbone=False,
    dropout=0.1,
):
    """
    Replaces get_normal_vit_baseline() from main.py.

    Key differences:
    - Configurable model variant
    - Optional backbone freezing (for 2-phase training)
    - Configurable dropout to combat overfitting on small dataset
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        attn_drop_rate=dropout,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[ViT] {model_name} | trainable: {trainable:,} / {total:,} params")
    print(f"[ViT] Backbone frozen: {freeze_backbone} | Dropout: {dropout}")

    return model